"""Coordinate-level stereochemistry constraints and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem

ChiralVolumeConstraint = Tuple[int, int, int, int, float, float]


@dataclass(frozen=True)
class TetrahedralStereoReference:
    """Reference geometry for one defined tetrahedral stereocenter."""

    center: int
    neighbors: Tuple[int, int, int]
    priority_neighbors: Tuple[int, ...]
    priority_ranks: Tuple[int, ...]
    label: str
    sign: float
    volume: float
    margin: float


def chiral_volume(coords: np.ndarray, center: int, i: int, j: int, k: int) -> float:
    """Return signed tetrahedral volume for ``center`` and three neighbours."""
    return float(
        (coords[i] - coords[center])
        @ np.cross(coords[j] - coords[center], coords[k] - coords[center])
    )


def _cip_ranked_neighbors(atom: Chem.Atom) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return explicit neighbours sorted by descending RDKit CIP rank."""
    ranked = []
    for neighbor in atom.GetNeighbors():
        if not neighbor.HasProp("_CIPRank"):
            continue
        ranked.append((neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank"))))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return (
        tuple(idx for idx, _ in ranked),
        tuple(rank for _, rank in ranked),
    )


def _unmapped_stereo_copy(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy whose atom-map labels cannot affect CIP ranking."""
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        atom.SetAtomMapNum(0)
    return copied


def tetrahedral_stereo_references(
    mol: Chem.Mol,
    coords: np.ndarray | None = None,
) -> List[TetrahedralStereoReference]:
    """
    Build reference labels and signed volumes for defined tetrahedral centers.

    ``mol`` must have a conformer, unless ``coords`` is supplied directly, whose
    coordinates already satisfy the molecule's intended stereochemistry. The
    signed volume is computed from the three highest-priority explicit
    substituents by RDKit ``_CIPRank`` on a map-stripped copy, so atom-map labels
    used for reaction bookkeeping cannot alter chemical priority. Unspecified,
    unranked, or near-planar centers are skipped.
    """
    stereo_mol = _unmapped_stereo_copy(mol)
    Chem.AssignStereochemistry(stereo_mol, force=True, cleanIt=True)
    positions = (
        np.asarray(coords, dtype=float)
        if coords is not None
        else np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    )
    references: List[TetrahedralStereoReference] = []
    for atom in stereo_mol.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        if not atom.HasProp("_CIPCode"):
            continue
        priority_neighbors, priority_ranks = _cip_ranked_neighbors(atom)
        if len(priority_neighbors) < 3:
            continue
        c = atom.GetIdx()
        i, j, k = priority_neighbors[:3]
        vol = chiral_volume(positions, c, i, j, k)
        if abs(vol) < 1e-6:
            continue
        references.append(
            TetrahedralStereoReference(
                center=c,
                neighbors=(i, j, k),
                priority_neighbors=priority_neighbors,
                priority_ranks=priority_ranks,
                label=atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",
                sign=1.0 if vol > 0.0 else -1.0,
                volume=vol,
                margin=0.5 * abs(vol),
            )
        )
    return references


def chiral_constraints_from_references(
    references: Iterable[TetrahedralStereoReference],
) -> List[ChiralVolumeConstraint]:
    """Convert tetrahedral stereo references to optimizer penalty constraints."""
    return [
        (ref.center, ref.neighbors[0], ref.neighbors[1], ref.neighbors[2], ref.sign, ref.margin)
        for ref in references
    ]


def stereo_mismatches(
    coords: np.ndarray,
    references: Sequence[TetrahedralStereoReference],
    require_margin: bool = False,
) -> List[TetrahedralStereoReference]:
    """Return references whose signed volume is wrong in ``coords``."""
    mismatches = []
    for ref in references:
        i, j, k = ref.neighbors
        vol = chiral_volume(coords, ref.center, i, j, k)
        threshold = ref.margin if require_margin else 0.0
        if ref.sign * vol <= threshold:
            mismatches.append(ref)
    return mismatches


def stereo_matches(
    coords: np.ndarray,
    references: Sequence[TetrahedralStereoReference],
    require_margin: bool = False,
) -> bool:
    """Return True when all referenced tetrahedral centers have the target sign."""
    return not stereo_mismatches(coords, references, require_margin=require_margin)


def stereo_debug_enabled(owner=None) -> bool:
    """Return True when stereochemistry diagnostics are explicitly enabled."""
    if owner is not None and getattr(owner, "stereo_debug", False):
        return True
    raw = os.environ.get("SPARTA_STEREO_DEBUG", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def format_stereo_state(
    label: str,
    coords: np.ndarray,
    references: Sequence[TetrahedralStereoReference],
) -> str:
    """Format tetrahedral reference-vs-current signed volumes for debugging."""
    lines = [f"[SPARTA stereo] {label}: {len(references)} tetrahedral reference(s)"]
    for ref in references:
        i, j, k = ref.neighbors
        vol = chiral_volume(coords, ref.center, i, j, k)
        signed = ref.sign * vol
        status = "OK" if signed > 0.0 else "MISMATCH"
        lines.append(
            "[SPARTA stereo] "
            f"  atom={ref.center} target={ref.label or '?'} "
            f"volume_neighbors={ref.neighbors} "
            f"priority_neighbors={ref.priority_neighbors} "
            f"priority_ranks={ref.priority_ranks} ref_sign={ref.sign:+.0f} "
            f"ref_vol={ref.volume:+.4f} cur_vol={vol:+.4f} "
            f"signed={signed:+.4f} {status}"
        )
    return "\n".join(lines)


def log_stereo_state(
    owner,
    label: str,
    coords: np.ndarray,
    references: Sequence[TetrahedralStereoReference],
) -> None:
    """Print stereo diagnostics only when explicitly enabled."""
    if stereo_debug_enabled(owner):
        print(format_stereo_state(label, coords, references))


def stereocenter_flipped(
    coords: np.ndarray,
    references: Sequence[TetrahedralStereoReference],
) -> bool:
    """Return True if any recorded stereocenter changes signed-volume sign."""
    return not stereo_matches(coords, references, require_margin=False)


def reflect_atoms_across_plane(
    coords: np.ndarray,
    atom_indices: Iterable[int],
    plane_points: Tuple[int, int, int],
) -> np.ndarray:
    """Reflect selected atoms across the plane defined by three atom indices."""
    p0, p1, p2 = (coords[idx] for idx in plane_points)
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return coords.copy()
    normal = normal / norm
    new_coords = coords.copy()
    selected = list(atom_indices)
    displacement = new_coords[selected] - p0
    new_coords[selected] -= 2.0 * (displacement @ normal)[:, None] * normal
    return new_coords


def _substituent_atoms(mol: Chem.Mol, center: int, neighbor: int) -> set[int]:
    """Atoms reachable from ``neighbor`` without crossing ``center``."""
    adjacency = Chem.GetAdjacencyMatrix(mol)
    seen: set[int] = set()
    stack = [neighbor]
    while stack:
        atom_idx = stack.pop()
        if atom_idx in seen or atom_idx == center:
            continue
        seen.add(atom_idx)
        for next_idx, bonded in enumerate(adjacency[atom_idx]):
            if bonded and next_idx not in seen and next_idx != center:
                stack.append(int(next_idx))
    return seen


def _movable_substituent(
    mol: Chem.Mol,
    ref: TetrahedralStereoReference,
) -> Tuple[int, set[int]] | None:
    """Choose a substituent side that can be reflected without crossing a ring."""
    all_neighbors = set(ref.priority_neighbors)
    candidates = []
    for neighbor in ref.neighbors:
        branch = _substituent_atoms(mol, ref.center, neighbor)
        if not branch:
            continue
        if branch & (all_neighbors - {neighbor}):
            continue
        candidates.append((neighbor, branch))
    if not candidates:
        return None
    return min(candidates, key=lambda item: len(item[1]))


def correct_tetrahedral_stereochemistry(
    coords: np.ndarray,
    mol: Chem.Mol,
    references: Sequence[TetrahedralStereoReference],
    max_passes: int = 4,
) -> np.ndarray:
    """
    Reflect clean substituent branches until tetrahedral signs match references.

    This is intentionally conservative: centers whose substituents are tied
    together through a ring are left for the optimizer/validator rather than
    applying a broad mirror operation that could scramble nearby stereochemistry.
    """
    fixed = np.asarray(coords, dtype=float).copy()
    for _ in range(max_passes):
        mismatches = stereo_mismatches(fixed, references)
        if not mismatches:
            break
        changed = False
        for ref in mismatches:
            movable = _movable_substituent(mol, ref)
            if movable is None:
                continue
            moving_neighbor, branch = movable
            plane_neighbors = [idx for idx in ref.neighbors if idx != moving_neighbor]
            if len(plane_neighbors) < 2:
                continue
            fixed = reflect_atoms_across_plane(
                fixed,
                branch,
                (ref.center, plane_neighbors[0], plane_neighbors[1]),
            )
            changed = True
        if not changed:
            break
    return fixed


def tetrahedral_chiral_constraints(ref_mol: Chem.Mol) -> List[ChiralVolumeConstraint]:
    """Backward-compatible wrapper for tetrahedral optimizer constraints."""
    return chiral_constraints_from_references(tetrahedral_stereo_references(ref_mol))


def stereocenter_targets(mol: Chem.Mol, coords: np.ndarray) -> List[TetrahedralStereoReference]:
    """Backward-compatible wrapper for endpoint stereo references."""
    return tetrahedral_stereo_references(mol, coords)
