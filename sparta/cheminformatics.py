"""Shared cheminformatics utility functions."""

import math
from collections import deque
from copy import copy
from typing import Dict, List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdDistGeom, rdMolTransforms
from rdkit.Geometry import Point3D
from ase import Atoms


def make_mol(smi: str) -> Chem.Mol:
    """
    Initialize an RDKit molecule from a mapped SMILES string, reordering atoms
    so that atom indices match atom map numbers (1-indexed → 0-indexed).
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    og = Chem.MolFromSmiles(smi, ps)
    fake_map = []
    for atom in og.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            fake_map.append(og.GetNumAtoms() - 1)
        else:
            fake_map.append(atom.GetAtomMapNum() - 1)
    indices_order = sorted(range(len(fake_map)), key=lambda x: fake_map[x])
    return Chem.RenumberAtoms(og, indices_order)


def add_coords_to_mol(coords: np.ndarray, mol: Chem.Mol) -> Chem.Mol:
    """Set (or replace) the first conformer of mol using coords."""
    if len(coords) != mol.GetNumAtoms():
        raise ValueError("Coordinate count does not match atom count.")
    if mol.GetNumConformers() == 0:
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
    conf = mol.GetConformer()
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, pos)
    return mol


def add_more_coords_to_mol(coords: np.ndarray, mol: Chem.Mol) -> Chem.Mol:
    """Append a new conformer to mol using coords."""
    if len(coords) != mol.GetNumAtoms():
        raise ValueError("Coordinate count does not match atom count.")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf, assignId=True)
    return mol


def generate_conformer(mol: Chem.Mol, smiles: str, seed: Optional[int]) -> Chem.Mol:
    """
    Generate a single ETKDG conformer. For multi-fragment SMILES (bimolecular),
    each fragment is embedded separately and translated along a different axis
    to avoid initial overlaps.
    """
    if seed is not None:
        result = AllChem.EmbedMolecule(mol, maxAttempts=100000, randomSeed=seed + 1)
    else:
        result = AllChem.EmbedMolecule(mol, maxAttempts=100000)
    if result != 0:
        raise ValueError("Could not embed molecule.")

    if "." in smiles:
        final_coords_symbols = {}
        for i, smi in enumerate(smiles.split(".")):
            translation = [0.0, 0.0, 0.0]
            if i < 3:
                translation[i] = 5.0
            ps = Chem.SmilesParserParams()
            ps.removeHs = False
            temp_mol = Chem.MolFromSmiles(smi, ps)
            if seed is not None:
                result = AllChem.EmbedMolecule(temp_mol, maxAttempts=100000, randomSeed=seed + 1)
            else:
                result = AllChem.EmbedMolecule(temp_mol, maxAttempts=100000)
            if result != 0:
                raise ValueError("Could not embed molecule.")
            AllChem.UFFOptimizeMolecule(temp_mol, maxIters=10000)
            conformer = temp_mol.GetConformer()
            for atom in temp_mol.GetAtoms():
                pos = conformer.GetAtomPosition(atom.GetIdx())
                final_coords_symbols[atom.GetAtomMapNum()] = (
                    pos.x + translation[0],
                    pos.y + translation[1],
                    pos.z + translation[2],
                )
        ordered_coords = [final_coords_symbols[i + 1] for i in range(len(final_coords_symbols))]
        mol = add_coords_to_mol(ordered_coords, mol)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=10000, ignoreInterfragInteractions=False)

    return mol


def transfer_conformer(src_mol: Chem.Mol, dst_mol: Chem.Mol, src_conf_id: int = 0) -> int:
    """Copy a conformer from src_mol to dst_mol. Returns the new conformer id."""
    if src_mol.GetNumAtoms() != dst_mol.GetNumAtoms():
        raise ValueError("Atom count mismatch; cannot transfer conformer.")
    src_conf = src_mol.GetConformer(src_conf_id)
    n = src_mol.GetNumAtoms()
    new_conf = Chem.Conformer(n)
    for i in range(n):
        new_conf.SetAtomPosition(i, src_conf.GetAtomPosition(i))
    return dst_mol.AddConformer(new_conf, assignId=True)


def find_mapped_substructure_match(mol1: Chem.Mol, mol2: Chem.Mol) -> List[List[int]]:
    """
    Find connected components in the intersection of bond adjacency matrices of
    two atom-mapped molecules. Returns all components with more than one atom.
    """
    bond_matrix1 = Chem.GetAdjacencyMatrix(mol1)
    bond_matrix2 = Chem.GetAdjacencyMatrix(mol2)
    common = np.minimum(bond_matrix1, bond_matrix2)

    n = common.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            comp.append(node)
            for nb in np.where(common[node] > 0)[0]:
                if not visited[int(nb)]:
                    stack.append(int(nb))
        components.append(comp)

    return [comp for comp in components if len(comp) > 1]


def get_coords_conformer(mol: Chem.Mol) -> Tuple[Chem.rdchem.Conformer, np.ndarray]:
    """Return the first conformer and its coordinate array."""
    conformer = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.array([
        [conformer.GetAtomPosition(i).x,
         conformer.GetAtomPosition(i).y,
         conformer.GetAtomPosition(i).z]
        for i in range(n)
    ])
    return conformer, coords


def get_all_coords_all_conformers(
    mol: Chem.Mol,
) -> List[Tuple[Chem.rdchem.Conformer, np.ndarray]]:
    """Return a list of (conformer, coords) for every conformer in mol."""
    result = []
    for conformer in mol.GetConformers():
        n = mol.GetNumAtoms()
        coords = np.array([
            [conformer.GetAtomPosition(i).x,
             conformer.GetAtomPosition(i).y,
             conformer.GetAtomPosition(i).z]
            for i in range(n)
        ])
        result.append((conformer, coords))
    return result


def mol_from_xyz(xyz_file: str, charge: int) -> Tuple[Chem.Mol, str]:
    """Generate an RDKit mol from an XYZ file using DetermineBonds."""
    raw_mol = Chem.MolFromXYZFile(xyz_file)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    smiles = Chem.MolToSmiles(mol)
    return mol, smiles


def add_xyz_to_mol(mol: Chem.Mol, xyz_file: str) -> Chem.Mol:
    """Add coordinates from an XYZ file as a new conformer on mol."""
    raw_mol = Chem.MolFromXYZFile(xyz_file)
    conf = raw_mol.GetConformer()
    mol.AddConformer(conf, assignId=True)
    return mol


def get_bonds(mol: Chem.Mol) -> set:
    """Return the set of bond tuples (i, j) with i < j."""
    bonds = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.add((min(i, j), max(i, j)))
    return bonds


def get_dihedral_matches(mol: Chem.Mol, heavy: bool = False) -> List[Tuple[int, int, int, int]]:
    """
    Identify unique dihedral (torsion) atom quartets using RDKit's strict SMARTS
    pattern. Uniquified by the central bond (b, c).
    """
    pattern = (
        r"*~[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)"
        r"&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])"
        r"&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])"
        r"&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)"
        r"&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]~*"
    )
    qmol = Chem.MolFromSmarts(pattern)
    matches = mol.GetSubstructMatches(qmol)

    uniqmatches = []
    seen = set()
    for a, b, c, d in matches:
        if (b, c) not in seen and (c, b) not in seen:
            if heavy:
                if (mol.GetAtomWithIdx(a).GetSymbol() != "H"
                        and mol.GetAtomWithIdx(d).GetSymbol() != "H"):
                    seen.add((b, c))
                    uniqmatches.append((a, b, c, d))
            else:
                if not (mol.GetAtomWithIdx(c).GetSymbol() == "C"
                        and mol.GetAtomWithIdx(d).GetSymbol() == "H"):
                    seen.add((b, c))
                    uniqmatches.append((a, b, c, d))
    return uniqmatches


def get_dihedral_matches_all(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    """
    Return one dihedral per non-ring bond, using the first available neighbor
    on each side of the bond.
    """
    uniqmatches: List[Tuple[int, int, int, int]] = []
    seen_bonds = set()

    for bond in mol.GetBonds():
        b = bond.GetBeginAtomIdx()
        c = bond.GetEndAtomIdx()

        if (b, c) in seen_bonds or (c, b) in seen_bonds:
            continue
        if bond.IsInRing():
            seen_bonds.add((b, c))
            seen_bonds.add((c, b))
            continue

        neigh_b = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(b).GetNeighbors() if nbr.GetIdx() != c]
        neigh_c = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(c).GetNeighbors() if nbr.GetIdx() != b]

        if neigh_b and neigh_c:
            uniqmatches.append((neigh_b[0], b, c, neigh_c[0]))

        seen_bonds.add((b, c))
        seen_bonds.add((c, b))

    return uniqmatches


def generate_bonded_atoms_dict(
    mol: Chem.Mol, dihedrals_list: List[Tuple[int, int, int, int]]
) -> Dict[Tuple[int, int, int, int], List[int]]:
    """
    For each dihedral (a, b, c, d), compute the set of atoms on the c-side
    of the b-c bond (excluding a, b, c themselves). These are the atoms that
    must be moved when the dihedral is rotated.
    """
    if mol is None or not dihedrals_list:
        return {}

    n = mol.GetNumAtoms()
    adj: List[List[int]] = [[] for _ in range(n)]
    for bnd in mol.GetBonds():
        i = bnd.GetBeginAtomIdx()
        j = bnd.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    res: Dict[Tuple[int, int, int, int], List[int]] = {}
    for d in dihedrals_list:
        a, b, c, dd = d
        visited = [False] * n
        stack = [int(c)]
        visited[int(c)] = True
        rotating: List[int] = [int(c)]
        while stack:
            node = stack.pop()
            for nei in adj[node]:
                if int(nei) == int(b) or int(nei) == int(c):
                    continue
                if not visited[int(nei)]:
                    visited[int(nei)] = True
                    stack.append(int(nei))
                    rotating.append(int(nei))
        seen: set = set()
        uniq_rotating: List[int] = []
        for x in rotating:
            if x not in seen:
                seen.add(x)
                uniq_rotating.append(x)
        filt = [x for x in uniq_rotating if x not in (int(a), int(b), int(c))]
        res[tuple(d)] = filt  # type: ignore[index]
    return res


def _min_bond_distance_to_reaction_center(
    dihedral: Tuple[int, int],
    reaction_center_atoms: set,
    mol: Chem.Mol,
) -> int:
    """BFS to find the minimum bond distance from any atom in dihedral to the reaction center."""
    n_atoms = mol.GetNumAtoms()
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    dihedral_atoms = set(dihedral)
    min_dist = float("inf")

    for start_atom in dihedral_atoms:
        if start_atom in reaction_center_atoms:
            return 0
        queue: deque = deque([(start_atom, 0)])
        visited = {start_atom}
        while queue:
            atom, dist = queue.popleft()
            for neighbor in adj[atom]:
                if neighbor in visited:
                    continue
                if neighbor in reaction_center_atoms:
                    min_dist = min(min_dist, dist + 1)
                    break
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return int(min_dist) if min_dist != float("inf") else 999


def filter_dihedrals_k_nearest(
    reactant_dihedrals: List[Tuple[int, int, int, int]],
    product_dihedrals: List[Tuple[int, int, int, int]],
    changing_bonds: List[Tuple[int, int]],
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    k: int = 3,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Filter dihedral lists to only those within k bonds of the reaction center
    (atoms involved in changing bonds).
    """
    if not changing_bonds:
        return reactant_dihedrals, product_dihedrals

    reaction_center_atoms = set()
    for bond in changing_bonds:
        reaction_center_atoms.add(bond[0])
        reaction_center_atoms.add(bond[1])

    filtered_reac = [
        d for d in reactant_dihedrals
        if _min_bond_distance_to_reaction_center(
            (d[1], d[2]), reaction_center_atoms, reactant_mol
        ) <= k
    ]
    filtered_prod = [
        d for d in product_dihedrals
        if _min_bond_distance_to_reaction_center(
            (d[1], d[2]), reaction_center_atoms, product_mol
        ) <= k
    ]
    return filtered_reac, filtered_prod


def rotate_dihedrals(
    conformer: Chem.rdchem.Conformer,
    dihedrals: List[Tuple[int, int, int, int]],
    angles: List[float],
) -> None:
    """Apply rotations to dihedrals in-place on a conformer (angles in degrees)."""
    for dihedral, angle in zip(dihedrals, angles):
        rad = math.pi * angle / 180.0
        rdMolTransforms.SetDihedralRad(conformer, *dihedral, value=rad)


def rotate_dihedrals_coords(
    coords: np.ndarray,
    dihedrals: List[Tuple[int, int, int, int]],
    angles: List[float],
    bonded_atoms_dict: Dict,
) -> np.ndarray:
    """
    Rotate the atoms on the c-side of each dihedral's b-c bond by the given
    angle (degrees). The moving atoms are determined by bonded_atoms_dict.
    """
    coords = np.asarray(coords, dtype=float).copy()

    if len(angles) != len(dihedrals):
        raise ValueError("`angles` must have the same length as `dihedrals`")

    for dihedral, angle in zip(dihedrals, angles):
        key = tuple(dihedral)
        a, b, c, d = key

        rotate_list = bonded_atoms_dict.get(key)
        if rotate_list is None:
            rotate_list = bonded_atoms_dict.get((a, b, c, d))
        rotate_idx = [int(x) for x in rotate_list] if rotate_list is not None else [int(d)]

        p1 = coords[int(b)].astype(float)
        p2 = coords[int(c)].astype(float)
        axis = p2 - p1
        axis_len = math.sqrt(np.dot(axis, axis))
        if axis_len < 1e-12:
            continue
        u = axis / axis_len

        theta = float(angle) * math.pi / 180.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for idx in rotate_idx:
            v = coords[idx] - p1
            cx = u[1] * v[2] - u[2] * v[1]
            cy = u[2] * v[0] - u[0] * v[2]
            cz = u[0] * v[1] - u[1] * v[0]
            udot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
            r = v * cos_t + np.array([cx, cy, cz]) * sin_t + u * (udot * (1 - cos_t))
            coords[idx] = r + p1

    return coords


def rotate_frag(coords: np.ndarray, angles: List[float], frag_indices: List[int]) -> np.ndarray:
    """
    Rotate the fragment defined by frag_indices around its centroid using
    roll, pitch, yaw angles (degrees).
    """
    roll, pitch, yaw = np.radians(angles)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    frag_coords = coords[frag_indices]
    centroid = frag_coords.mean(axis=0)
    frag_rotated = (R @ (frag_coords - centroid).T).T + centroid
    coords[frag_indices] = frag_rotated
    return coords


def rotate_frags(
    coords: np.ndarray,
    angles_list: List[float],
    frag_indices_list: List[List[int]],
) -> np.ndarray:
    """
    Rotate multiple fragments. angles_list is a flat list where every 3 values
    are (roll, pitch, yaw) for one fragment.
    """
    if len(angles_list) != 3 * len(frag_indices_list):
        raise ValueError(
            f"Expected {3 * len(frag_indices_list)} angles for "
            f"{len(frag_indices_list)} fragments, got {len(angles_list)}."
        )
    for i, frag_indices in enumerate(frag_indices_list):
        coords = rotate_frag(coords, angles_list[3 * i: 3 * (i + 1)], list(frag_indices))
    return coords


def translate_frag(
    coords: np.ndarray,
    translation: Tuple[float, float, float],
    frag_indices: List[int],
) -> np.ndarray:
    """Translate the atoms in frag_indices by the given (dx, dy, dz) vector."""
    translated = coords.copy()
    translated[frag_indices] += np.array(translation)
    return translated


def translate_frags(
    coords: np.ndarray,
    translations_list: List[float],
    frag_indices_list: List[List[int]],
) -> np.ndarray:
    """
    Translate multiple fragments. translations_list is a flat list where every
    3 values are (dx, dy, dz) for one fragment.
    """
    if len(translations_list) != 3 * len(frag_indices_list):
        raise ValueError(
            f"Expected {3 * len(frag_indices_list)} translation values for "
            f"{len(frag_indices_list)} fragments, got {len(translations_list)}."
        )
    for i, frag_indices in enumerate(frag_indices_list):
        coords = translate_frag(coords, translations_list[3 * i: 3 * (i + 1)], list(frag_indices))
    return coords


def mol_to_ase_atoms(mol: Chem.Mol) -> Atoms:
    """Convert an RDKit mol (with conformer) to an ASE Atoms object."""
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert mol.GetNumConformers() > 0
    positions = mol.GetConformer().GetPositions()
    return Atoms(symbols=symbols, positions=np.array(positions))


def add_ase_coords_to_mol(atoms: Atoms, mol: Chem.Mol) -> Chem.Mol:
    """Overwrite the first conformer of mol with positions from an ASE Atoms object."""
    positions = atoms.get_positions()
    if len(positions) != mol.GetNumAtoms():
        raise ValueError("Atom count mismatch between ASE Atoms and RDKit Mol.")
    conf = mol.GetConformer()
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos)
    return mol


def kabsch_align(mobile_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """
    Align mobile_coords to target_coords using the Kabsch algorithm.
    Returns the aligned coordinates (same shape as mobile_coords).
    """
    mobile_centered = mobile_coords - np.mean(mobile_coords, axis=0)
    target_centered = target_coords - np.mean(target_coords, axis=0)

    cov = np.dot(mobile_centered.T, target_centered)
    V, S, Wt = np.linalg.svd(cov)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, Wt)
    aligned = np.dot(mobile_centered, R) + np.mean(target_coords, axis=0)
    return aligned
