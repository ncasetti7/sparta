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


@dataclass(frozen=True)
class AlkeneStereoReference:
    """Reference geometry for one defined E/Z alkene."""

    bond: Tuple[int, int]
    substituents: Tuple[int, int]
    priority_ranks: Tuple[int, int]
    label: str
    sign: float
    value: float
    margin: float


@dataclass(frozen=True)
class AtropisomerStereoReference:
    """Reference geometry for one defined atropisomeric axis."""

    bond_index: int
    axis: Tuple[int, int]
    controlling_atoms: Tuple[int, int, int, int]
    descriptor_atoms: Tuple[int, int]
    label: str
    sign: float
    value: float
    angle: float
    margin: float


def chiral_volume(coords: np.ndarray, center: int, i: int, j: int, k: int) -> float:
    """Return signed tetrahedral volume for ``center`` and three neighbours."""
    return float(
        (coords[i] - coords[center])
        @ np.cross(coords[j] - coords[center], coords[k] - coords[center])
    )


def alkene_side_value(coords: np.ndarray, a: int, b: int, i: int, j: int) -> float:
    """Return side descriptor for substituents i/j around double bond a=b."""
    positions = np.asarray(coords, dtype=float)
    axis = positions[b] - positions[a]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return 0.0
    axis = axis / axis_norm
    vi = positions[i] - positions[a]
    vj = positions[j] - positions[b]
    vi = vi - np.dot(vi, axis) * axis
    vj = vj - np.dot(vj, axis) * axis
    ni = np.linalg.norm(vi)
    nj = np.linalg.norm(vj)
    if ni < 1e-10 or nj < 1e-10:
        return 0.0
    return float(np.dot(vi / ni, vj / nj))


def _dihedral_angle(coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Return signed dihedral angle in radians for i-j-k-l."""
    positions = np.asarray(coords, dtype=float)
    b0 = positions[i] - positions[j]
    b1 = positions[k] - positions[j]
    b2 = positions[l] - positions[k]
    norm = np.linalg.norm(b1)
    if norm < 1e-10:
        return 0.0
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-10 or nw < 1e-10:
        return 0.0
    v = v / nv
    w = w / nw
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def atropisomer_axis_value(coords: np.ndarray, a: int, b: int, i: int, j: int) -> float:
    """Return signed axial descriptor for substituents i/j around axis a-b."""
    return float(np.sin(_dihedral_angle(coords, i, a, b, j)))


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


def _highest_priority_neighbor(atom: Chem.Atom, exclude: int) -> Tuple[int, int] | None:
    """Return highest-CIP-rank explicit neighbor other than ``exclude``."""
    ranked = []
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == exclude or not neighbor.HasProp("_CIPRank"):
            continue
        ranked.append((neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank"))))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[0]


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


def alkene_stereo_references(
    mol: Chem.Mol,
    coords: np.ndarray | None = None,
) -> List[AlkeneStereoReference]:
    """
    Build reference labels and side descriptors for defined E/Z alkenes.

    Atom maps are stripped before RDKit assigns stereochemistry and CIP ranks, so
    reaction-map bookkeeping cannot alter chemical priority. Original atom
    indices are retained for coordinate descriptors.
    """
    stereo_mol = _unmapped_stereo_copy(mol)
    Chem.AssignStereochemistry(stereo_mol, force=True, cleanIt=True)
    positions = (
        np.asarray(coords, dtype=float)
        if coords is not None
        else np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    )
    references: List[AlkeneStereoReference] = []
    for bond in stereo_mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        left = _highest_priority_neighbor(stereo_mol.GetAtomWithIdx(a), b)
        right = _highest_priority_neighbor(stereo_mol.GetAtomWithIdx(b), a)
        if left is None or right is None:
            continue
        i, rank_i = left
        j, rank_j = right
        value = alkene_side_value(positions, a, b, i, j)
        if abs(value) < 1e-6:
            continue
        references.append(
            AlkeneStereoReference(
                bond=(a, b),
                substituents=(i, j),
                priority_ranks=(rank_i, rank_j),
                label="E" if stereo == Chem.BondStereo.STEREOE else "Z",
                sign=1.0 if value > 0.0 else -1.0,
                value=value,
                margin=0.5 * abs(value),
            )
        )
    return references


def atropisomer_stereo_references(
    mol: Chem.Mol,
    coords: np.ndarray | None = None,
) -> List[AtropisomerStereoReference]:
    """
    Build reference labels and signed torsion descriptors for atropisomer axes.

    RDKit represents atropisomerism as bond stereo, usually serialized in
    CXSMILES as ``|wU:...|`` or ``|wD:...|`` fields. The graph is copied and atom
    maps are stripped before stereo perception, while original atom indices are
    retained for coordinate descriptors.
    """
    stereo_mol = _unmapped_stereo_copy(mol)
    Chem.AssignStereochemistry(stereo_mol, force=True, cleanIt=True)
    try:
        stereo_infos = Chem.FindPotentialStereo(stereo_mol, cleanIt=True, flagPossible=True)
    except RuntimeError:
        stereo_infos = []
    positions = (
        np.asarray(coords, dtype=float)
        if coords is not None
        else np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    )
    references: List[AtropisomerStereoReference] = []
    for info in stereo_infos:
        if info.type != Chem.StereoType.Bond_Atropisomer:
            continue
        if info.specified != Chem.StereoSpecified.Specified:
            continue
        bond_idx = int(info.centeredOn)
        if bond_idx < 0 or bond_idx >= stereo_mol.GetNumBonds():
            continue
        bond = stereo_mol.GetBondWithIdx(bond_idx)
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW):
            continue
        controlling = tuple(int(idx) for idx in info.controllingAtoms)
        if len(controlling) != 4:
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        left = [idx for idx in controlling if stereo_mol.GetBondBetweenAtoms(a, idx) is not None]
        right = [idx for idx in controlling if stereo_mol.GetBondBetweenAtoms(b, idx) is not None]
        if not left or not right:
            continue
        i, j = left[0], right[0]
        angle = _dihedral_angle(positions, i, a, b, j)
        value = float(np.sin(angle))
        if abs(value) < 1e-6:
            continue
        references.append(
            AtropisomerStereoReference(
                bond_index=bond_idx,
                axis=(a, b),
                controlling_atoms=controlling,
                descriptor_atoms=(i, j),
                label="CW" if stereo == Chem.BondStereo.STEREOATROPCW else "CCW",
                sign=1.0 if value > 0.0 else -1.0,
                value=value,
                angle=angle,
                margin=0.5 * abs(value),
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


def alkene_stereo_mismatches(
    coords: np.ndarray,
    references: Sequence[AlkeneStereoReference],
    require_margin: bool = False,
) -> List[AlkeneStereoReference]:
    """Return E/Z references whose side descriptor is wrong in ``coords``."""
    mismatches = []
    for ref in references:
        a, b = ref.bond
        i, j = ref.substituents
        value = alkene_side_value(coords, a, b, i, j)
        threshold = ref.margin if require_margin else 0.0
        if ref.sign * value <= threshold:
            mismatches.append(ref)
    return mismatches


def alkene_stereo_matches(
    coords: np.ndarray,
    references: Sequence[AlkeneStereoReference],
    require_margin: bool = False,
) -> bool:
    """Return True when all referenced E/Z alkenes have the target side sign."""
    return not alkene_stereo_mismatches(coords, references, require_margin=require_margin)


def atropisomer_stereo_mismatches(
    coords: np.ndarray,
    references: Sequence[AtropisomerStereoReference],
    require_margin: bool = False,
) -> List[AtropisomerStereoReference]:
    """Return atropisomer references whose signed axis descriptor is wrong."""
    mismatches = []
    for ref in references:
        a, b = ref.axis
        i, j = ref.descriptor_atoms
        value = atropisomer_axis_value(coords, a, b, i, j)
        threshold = ref.margin if require_margin else 0.0
        if ref.sign * value <= threshold:
            mismatches.append(ref)
    return mismatches


def atropisomer_stereo_matches(
    coords: np.ndarray,
    references: Sequence[AtropisomerStereoReference],
    require_margin: bool = False,
) -> bool:
    """Return True when all referenced atropisomer axes have the target sign."""
    return not atropisomer_stereo_mismatches(coords, references, require_margin=require_margin)


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


def format_alkene_state(
    label: str,
    coords: np.ndarray,
    references: Sequence[AlkeneStereoReference],
) -> str:
    """Format E/Z reference-vs-current side descriptors for debugging."""
    lines = [f"[SPARTA stereo] {label}: {len(references)} alkene E/Z reference(s)"]
    for ref in references:
        a, b = ref.bond
        i, j = ref.substituents
        value = alkene_side_value(coords, a, b, i, j)
        signed = ref.sign * value
        status = "OK" if signed > 0.0 else "MISMATCH"
        lines.append(
            "[SPARTA stereo] "
            f"  bond={ref.bond} target={ref.label or '?'} "
            f"substituents={ref.substituents} "
            f"priority_ranks={ref.priority_ranks} ref_sign={ref.sign:+.0f} "
            f"ref_value={ref.value:+.4f} cur_value={value:+.4f} "
            f"signed={signed:+.4f} {status}"
        )
    return "\n".join(lines)


def format_atropisomer_state(
    label: str,
    coords: np.ndarray,
    references: Sequence[AtropisomerStereoReference],
) -> str:
    """Format atropisomer reference-vs-current signed descriptors for debugging."""
    lines = [f"[SPARTA stereo] {label}: {len(references)} atropisomer reference(s)"]
    for ref in references:
        a, b = ref.axis
        i, j = ref.descriptor_atoms
        value = atropisomer_axis_value(coords, a, b, i, j)
        signed = ref.sign * value
        status = "OK" if signed > 0.0 else "MISMATCH"
        lines.append(
            "[SPARTA stereo] "
            f"  bond={ref.axis} target={ref.label or '?'} "
            f"controlling_atoms={ref.controlling_atoms} "
            f"descriptor_atoms={ref.descriptor_atoms} ref_sign={ref.sign:+.0f} "
            f"ref_value={ref.value:+.4f} cur_value={value:+.4f} "
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


def log_alkene_state(
    owner,
    label: str,
    coords: np.ndarray,
    references: Sequence[AlkeneStereoReference],
) -> None:
    """Print E/Z diagnostics only when explicitly enabled."""
    if stereo_debug_enabled(owner):
        print(format_alkene_state(label, coords, references))


def log_atropisomer_state(
    owner,
    label: str,
    coords: np.ndarray,
    references: Sequence[AtropisomerStereoReference],
) -> None:
    """Print atropisomer diagnostics only when explicitly enabled."""
    if stereo_debug_enabled(owner):
        print(format_atropisomer_state(label, coords, references))


def stereocenter_flipped(
    coords: np.ndarray,
    references: Sequence[TetrahedralStereoReference],
) -> bool:
    """Return True if any recorded stereocenter changes signed-volume sign."""
    return not stereo_matches(coords, references, require_margin=False)


def alkene_stereo_flipped(
    coords: np.ndarray,
    references: Sequence[AlkeneStereoReference],
) -> bool:
    """Return True if any recorded E/Z alkene changes side-descriptor sign."""
    return not alkene_stereo_matches(coords, references, require_margin=False)


def atropisomer_stereo_flipped(
    coords: np.ndarray,
    references: Sequence[AtropisomerStereoReference],
) -> bool:
    """Return True if any recorded atropisomer axis changes descriptor sign."""
    return not atropisomer_stereo_matches(coords, references, require_margin=False)


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


def rotate_atoms_around_axis(
    coords: np.ndarray,
    atom_indices: Iterable[int],
    axis_points: Tuple[int, int],
    angle: float,
) -> np.ndarray:
    """Rotate selected atoms around the axis defined by two atom indices."""
    new_coords = np.asarray(coords, dtype=float).copy()
    p0, p1 = (new_coords[idx] for idx in axis_points)
    axis = p1 - p0
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return new_coords
    axis = axis / norm
    selected = list(atom_indices)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    rel = new_coords[selected] - p0
    rotated = (
        rel * cos_a
        + np.cross(axis, rel) * sin_a
        + axis * (rel @ axis)[:, None] * (1.0 - cos_a)
    )
    new_coords[selected] = p0 + rotated
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
    """
    Choose a substituent side that can be reflected without crossing a ring.

    A tetrahedral flip needs an improper (reflection) operation, which would
    invert any *other* stereocenter living inside the reflected branch. Branches
    that carry their own defined stereochemistry are therefore rejected, so a
    neighbouring center is never collateral damage; if that leaves no clean
    branch, the corrector declines and the optimizer's per-center chiral-volume
    penalty (which acts only on this center) is left to enforce the sign.
    """
    all_neighbors = set(ref.priority_neighbors)
    candidates = []
    for neighbor in ref.neighbors:
        branch = _substituent_atoms(mol, ref.center, neighbor)
        if not branch:
            continue
        if branch & (all_neighbors - {neighbor}):
            continue
        if _branch_has_internal_stereo(mol, branch):
            continue
        candidates.append((neighbor, branch))
    if not candidates:
        return None
    return min(candidates, key=lambda item: len(item[1]))


def _movable_alkene_substituent(
    mol: Chem.Mol,
    ref: AlkeneStereoReference,
) -> Tuple[int, set[int]] | None:
    """Choose a clean alkene substituent branch to rotate around the double bond."""
    a, b = ref.bond
    i, j = ref.substituents
    candidates = []
    for center, neighbor, blocked in ((a, i, b), (b, j, a)):
        branch = _substituent_atoms(mol, center, neighbor)
        if not branch:
            continue
        if blocked in branch:
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


def correct_alkene_stereochemistry(
    coords: np.ndarray,
    mol: Chem.Mol,
    references: Sequence[AlkeneStereoReference],
    max_passes: int = 4,
) -> np.ndarray:
    """
    Rotate clean substituent branches until E/Z side signs match references.

    This is a coordinate repair for generated product conformers, not a physical
    alkene isomerization model. Ring-tied or otherwise non-separable substituent
    branches are left for validation/rejection.
    """
    fixed = np.asarray(coords, dtype=float).copy()
    for _ in range(max_passes):
        mismatches = alkene_stereo_mismatches(fixed, references)
        if not mismatches:
            break
        changed = False
        for ref in mismatches:
            movable = _movable_alkene_substituent(mol, ref)
            if movable is None:
                continue
            _, branch = movable
            fixed = rotate_atoms_around_axis(fixed, branch, ref.bond, np.pi)
            changed = True
        if not changed:
            break
    return fixed


def _branch_has_internal_stereo(mol: Chem.Mol, branch: set[int]) -> bool:
    """
    True if ``branch`` carries a defined *chiral* stereo element of its own — a
    tetrahedral center or an atropisomeric axis. Reflection (an improper
    operation) inverts those, so a corrector that reflects such a branch would
    silently corrupt them and must rotate instead / decline. E/Z double bonds are
    deliberately excluded: reflection preserves cis/trans, so an alkene inside the
    branch is not a reason to avoid reflecting it.
    """
    for idx in branch:
        if mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    reflection_sensitive = (
        Chem.BondStereo.STEREOATROPCW,
        Chem.BondStereo.STEREOATROPCCW,
    )
    for bond in mol.GetBonds():
        if (
            bond.GetBeginAtomIdx() in branch
            and bond.GetEndAtomIdx() in branch
            and bond.GetStereo() in reflection_sensitive
        ):
            return True
    return False


def _movable_atropisomer_substituent(
    mol: Chem.Mol,
    ref: AtropisomerStereoReference,
) -> Tuple[set[int], Tuple[int, int, int]] | None:
    """
    Choose a clean branch on one side of the axis to move, plus the three atoms
    defining the reflection plane that inverts the axial sign.

    The axis is ``a-b``; ``i``/``j`` are the controlling atoms bonded to ``a``/``b``.
    Reflecting the ``b``-side branch across plane ``(i, a, b)`` sends the signed
    dihedral ``i-a-b-j`` to its negative (the true mirror-image atropisomer),
    and symmetrically for the ``a``-side branch across ``(j, b, a)``. The two
    axis atoms lie in either plane, so they never move. Branches that loop back
    across the axis (a fused/bridged system) are rejected, mirroring the
    conservative stance of the tetrahedral and alkene correctors.
    """
    a, b = ref.axis
    i, j = ref.descriptor_atoms
    candidates = []
    # (branch-root neighbour, opposite axis atom, blocked atom, plane points)
    for root, other_axis, blocked, plane in (
        (b, a, i, (i, a, b)),
        (a, b, j, (j, b, a)),
    ):
        branch = _substituent_atoms(mol, other_axis, root)
        if not branch or blocked in branch or other_axis in branch:
            continue
        candidates.append((branch, plane))
    if not candidates:
        return None
    return min(candidates, key=lambda item: len(item[0]))


def correct_atropisomer_stereochemistry(
    coords: np.ndarray,
    mol: Chem.Mol,
    references: Sequence[AtropisomerStereoReference],
    max_passes: int = 4,
) -> np.ndarray:
    """
    Flip mis-set atropisomeric axes onto their reference sign, in place on a
    copy of ``coords``.

    For each mismatched axis the corrector moves the smaller clean branch on one
    side of the axis. It **reflects** that branch across the plane through the
    axis and the opposite side's controlling atom, which lands the branch on the
    true mirror-image geometry (dihedral ``theta -> -theta``): an isometry that
    preserves the branch's internal geometry and matches the intended-isomer
    reference/bounds, so it introduces no new strain on the pushforward. When the
    branch carries its own defined stereochemistry, reflection would invert it, so
    the corrector falls back to a ``pi`` rotation about the axis (a proper
    operation: ``sin(theta + pi) = -sin(theta)`` flips the axial sign while
    preserving every internal center, at the cost of landing on a different
    rotamer that the subsequent relaxation must resolve).

    Axes whose branches are tied together through a ring are left for
    validation/rejection.
    """
    fixed = np.asarray(coords, dtype=float).copy()
    for _ in range(max_passes):
        mismatches = atropisomer_stereo_mismatches(fixed, references)
        if not mismatches:
            break
        changed = False
        for ref in mismatches:
            movable = _movable_atropisomer_substituent(mol, ref)
            if movable is None:
                continue
            branch, plane = movable
            if _branch_has_internal_stereo(mol, branch):
                fixed = rotate_atoms_around_axis(fixed, branch, ref.axis, np.pi)
            else:
                fixed = reflect_atoms_across_plane(fixed, branch, plane)
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
