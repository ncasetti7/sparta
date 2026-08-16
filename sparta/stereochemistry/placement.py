"""Reaction-aware placement of fragments that create new stereochemistry.

The coordinate descriptors and validation rules live in ``constraints.py``.
This module deliberately reuses those rules and only changes the reactant pose
when a newly specified product stereo element spans two reactant fragments.
Intramolecular elements are not classified here and therefore continue through
the legacy product-coordinate correction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence, Tuple

import numpy as np
from rdkit import Chem

from sparta.stereochemistry.constraints import (
    AlkeneStereoReference,
    AtropisomerStereoReference,
    TetrahedralStereoReference,
    alkene_stereo_mismatches,
    atropisomer_stereo_mismatches,
    rotate_atoms_around_axis,
    stereo_mismatches,
)


@dataclass(frozen=True)
class IntermolecularStereoReferences:
    """New product stereo elements whose defining atoms span reactant fragments."""

    tetrahedral: Tuple[TetrahedralStereoReference, ...] = ()
    alkene: Tuple[AlkeneStereoReference, ...] = ()
    atropisomer: Tuple[AtropisomerStereoReference, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.tetrahedral or self.alkene or self.atropisomer)


def _fragment_lookup(mol: Chem.Mol) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    fragments = tuple(tuple(int(idx) for idx in frag) for frag in Chem.GetMolFrags(mol))
    atom_to_fragment = {
        atom_idx: fragment_idx
        for fragment_idx, fragment in enumerate(fragments)
        for atom_idx in fragment
    }
    fragment_atoms = {idx: fragment for idx, fragment in enumerate(fragments)}
    return atom_to_fragment, fragment_atoms


def intermolecular_stereo_references(
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    reactant_tetrahedral: Sequence[TetrahedralStereoReference],
    product_tetrahedral: Sequence[TetrahedralStereoReference],
    reactant_alkene: Sequence[AlkeneStereoReference],
    product_alkene: Sequence[AlkeneStereoReference],
    reactant_atropisomer: Sequence[AtropisomerStereoReference],
    product_atropisomer: Sequence[AtropisomerStereoReference],
) -> IntermolecularStereoReferences:
    """Identify newly specified product stereo elements formed between fragments.

    Identity is based only on mapped atom indices.  Labels are intentionally not
    compared because CIP labels may change when the endpoint graph changes.
    """
    atom_to_fragment, _ = _fragment_lookup(reactant_mol)

    reactant_centers = {ref.center for ref in reactant_tetrahedral}
    tetrahedral = []
    for ref in product_tetrahedral:
        if ref.center in reactant_centers:
            continue
        center_fragment = atom_to_fragment[ref.center]
        product_neighbors = (
            neighbor.GetIdx()
            for neighbor in product_mol.GetAtomWithIdx(ref.center).GetNeighbors()
        )
        if any(atom_to_fragment[idx] != center_fragment for idx in product_neighbors):
            tetrahedral.append(ref)

    reactant_alkene_bonds = {tuple(sorted(ref.bond)) for ref in reactant_alkene}
    alkene = [
        ref
        for ref in product_alkene
        if tuple(sorted(ref.bond)) not in reactant_alkene_bonds
        and _axis_has_interfragment_product_bond(
            product_mol, ref.bond, atom_to_fragment
        )
    ]

    reactant_axes = {tuple(sorted(ref.axis)) for ref in reactant_atropisomer}
    atropisomer = [
        ref
        for ref in product_atropisomer
        if tuple(sorted(ref.axis)) not in reactant_axes
        and _axis_has_interfragment_product_bond(
            product_mol, ref.axis, atom_to_fragment
        )
    ]

    return IntermolecularStereoReferences(
        tetrahedral=tuple(tetrahedral),
        alkene=tuple(alkene),
        atropisomer=tuple(atropisomer),
    )


def _axis_has_interfragment_product_bond(
    product_mol: Chem.Mol,
    axis: tuple[int, int],
    atom_to_fragment: dict[int, int],
) -> bool:
    """Whether forming the axis or one of its substituents joins fragments."""
    a, b = axis
    for center in (a, b):
        center_fragment = atom_to_fragment[center]
        for neighbor in product_mol.GetAtomWithIdx(center).GetNeighbors():
            if atom_to_fragment[neighbor.GetIdx()] != center_fragment:
                return True
    return False


def _best_plane_neighbors(
    coords: np.ndarray,
    center: int,
    candidates: Sequence[int],
) -> tuple[int, int] | None:
    """Choose the numerically best-defined center/neighbor/neighbor plane."""
    best = None
    best_area = 0.0
    for i, j in combinations(sorted(candidates), 2):
        area = float(
            np.linalg.norm(
                np.cross(coords[i] - coords[center], coords[j] - coords[center])
            )
        )
        if area > best_area:
            best = (i, j)
            best_area = area
    return best if best_area > 1e-10 else None


def _move_tetrahedral_incoming_fragment(
    coords: np.ndarray,
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    ref: TetrahedralStereoReference,
) -> tuple[np.ndarray, bool]:
    """Translate an incoming fragment to the opposite face of a forming center.

    Only the incoming attachment atom is mirrored geometrically.  Its complete
    reactant fragment receives the resulting translation, a proper rigid-body
    operation that preserves all internal stereochemistry and distances.
    """
    atom_to_fragment, fragment_atoms = _fragment_lookup(reactant_mol)
    center_fragment = atom_to_fragment[ref.center]
    product_neighbors = [
        neighbor.GetIdx()
        for neighbor in product_mol.GetAtomWithIdx(ref.center).GetNeighbors()
    ]
    incoming = [
        idx for idx in product_neighbors if atom_to_fragment[idx] != center_fragment
    ]
    stationary = [
        idx for idx in product_neighbors if atom_to_fragment[idx] == center_fragment
    ]
    if not incoming or len(stationary) < 2:
        return np.asarray(coords, dtype=float).copy(), False

    incoming_atom = min(
        incoming,
        key=lambda idx: (len(fragment_atoms[atom_to_fragment[idx]]), idx),
    )
    plane_neighbors = _best_plane_neighbors(coords, ref.center, stationary)
    if plane_neighbors is None:
        return np.asarray(coords, dtype=float).copy(), False

    i, j = plane_neighbors
    center_position = coords[ref.center]
    normal = np.cross(coords[i] - center_position, coords[j] - center_position)
    normal /= np.linalg.norm(normal)
    signed_distance = float(np.dot(coords[incoming_atom] - center_position, normal))
    if abs(signed_distance) < 1e-10:
        return np.asarray(coords, dtype=float).copy(), False

    displacement = -2.0 * signed_distance * normal
    moved = np.asarray(coords, dtype=float).copy()
    atoms = fragment_atoms[atom_to_fragment[incoming_atom]]
    moved[np.asarray(atoms, dtype=int)] += displacement
    return moved, True


def _rotate_smaller_axis_fragment(
    coords: np.ndarray,
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    axis: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    """Rotate the smaller incoming reactant fragment by pi about a stereo axis."""
    atom_to_fragment, fragment_atoms = _fragment_lookup(reactant_mol)
    a, b = axis
    candidate_fragments = set()
    for center in (a, b):
        center_fragment = atom_to_fragment[center]
        for neighbor in product_mol.GetAtomWithIdx(center).GetNeighbors():
            neighbor_fragment = atom_to_fragment[neighbor.GetIdx()]
            if neighbor_fragment != center_fragment:
                candidate_fragments.add(neighbor_fragment)
    if not candidate_fragments:
        return np.asarray(coords, dtype=float).copy(), False
    moving_fragment = min(
        candidate_fragments,
        key=lambda idx: (len(fragment_atoms[idx]), min(fragment_atoms[idx])),
    )
    moved = rotate_atoms_around_axis(
        coords,
        fragment_atoms[moving_fragment],
        axis,
        np.pi,
    )
    return moved, not np.allclose(moved, coords, atol=1e-12, rtol=0.0)


def reposition_intermolecular_stereochemistry(
    reactant_coords: np.ndarray,
    trial_product_coords: np.ndarray,
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    references: IntermolecularStereoReferences,
) -> tuple[np.ndarray, bool]:
    """Move reactant fragments for mismatched new intermolecular stereo.

    Mismatches are diagnosed exclusively with the existing product-coordinate
    checkers.  The returned coordinates differ only by whole-fragment rigid
    translations or rotations.
    """
    moved_coords = np.asarray(reactant_coords, dtype=float).copy()
    changed = False

    for ref in stereo_mismatches(trial_product_coords, references.tetrahedral):
        moved_coords, did_move = _move_tetrahedral_incoming_fragment(
            moved_coords,
            reactant_mol,
            product_mol,
            ref,
        )
        changed = changed or did_move

    for ref in alkene_stereo_mismatches(trial_product_coords, references.alkene):
        moved_coords, did_move = _rotate_smaller_axis_fragment(
            moved_coords,
            reactant_mol,
            product_mol,
            ref.bond,
        )
        changed = changed or did_move

    for ref in atropisomer_stereo_mismatches(
        trial_product_coords, references.atropisomer
    ):
        moved_coords, did_move = _rotate_smaller_axis_fragment(
            moved_coords,
            reactant_mol,
            product_mol,
            ref.axis,
        )
        changed = changed or did_move

    return moved_coords, changed
