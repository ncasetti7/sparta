"""
Propagation of internal coordinate changes through a molecular graph.

When a bond length changes or a structural transformation is applied, the
displacement must propagate to all atoms on the moving side of the bond.
These functions implement that propagation using BFS on the molecular graph.

Also contains endpoint gradient computation and adjustment used during
iterative NEB optimisation.
"""

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def _get_connected_fragment(
    bond_matrix: np.ndarray, fixed_atom: int, moving_atom: int
) -> set:
    """
    Return the set of atom indices reachable from moving_atom without crossing
    through fixed_atom (BFS on the adjacency matrix).

    Parameters
    ----------
    bond_matrix : np.ndarray, shape (n_atoms, n_atoms)
        Boolean/integer adjacency matrix.
    fixed_atom : int
        Atom whose side of the bond is held fixed.
    moving_atom : int
        Starting atom for the moving fragment.

    Returns
    -------
    set of int
        Indices of all atoms in the moving fragment.
    """
    n_atoms = bond_matrix.shape[0]
    visited: set = set()
    queue = [moving_atom]
    while queue:
        current = queue.pop(0)
        if current in visited or current == fixed_atom:
            continue
        visited.add(current)
        for neighbor in range(n_atoms):
            if bond_matrix[current, neighbor] and neighbor not in visited and neighbor != fixed_atom:
                queue.append(neighbor)
    return visited


def bond_change(
    coords: np.ndarray,
    bond_matrix: np.ndarray,
    atom_i: int,
    atom_j: int,
    target_length: float,
) -> np.ndarray:
    """
    Adjust the distance between atom_i and atom_j to target_length by translating
    the fragment connected to atom_j (not through atom_i) along the bond axis.

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
    bond_matrix : np.ndarray, shape (n_atoms, n_atoms)
        Adjacency matrix indicating bonded pairs.
    atom_i : int
        Fixed-side atom.
    atom_j : int
        Moving-side atom (its connected fragment will be translated).
    target_length : float
        Desired bond length in Angstroms.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Updated coordinates.
    """
    coords = np.array(coords, dtype=float)
    bond_vec = coords[atom_j] - coords[atom_i]
    current_length = np.linalg.norm(bond_vec)
    if current_length < 1e-10:
        raise ValueError(f"Bond between atoms {atom_i} and {atom_j} has zero length.")
    displacement = (target_length - current_length) * (bond_vec / current_length)
    moving_fragment = _get_connected_fragment(bond_matrix, atom_i, atom_j)
    new_coords = coords.copy()
    for atom_idx in moving_fragment:
        new_coords[atom_idx] += displacement
    return new_coords


def transform_mol(
    reactant_coords: np.ndarray,
    product_mol: Chem.Mol,
    bond_changes: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Transform reactant coordinates to approximate product geometry by adjusting
    the lengths of all changing bonds to their product values.

    For each bond in bond_changes, the fragment connected to the second atom
    (not through the first atom) is translated along the bond axis so that the
    bond length matches the product conformer.

    Parameters
    ----------
    reactant_coords : np.ndarray, shape (n_atoms, 3)
        Starting reactant coordinates.
    product_mol : Chem.Mol
        Product molecule with a 3D conformer (used for target bond lengths).
    bond_changes : list of (int, int)
        Pairs of atom indices whose bond lengths should be adjusted.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Transformed coordinates.
    """
    prod_conf = product_mol.GetConformer()
    bond_matrix = Chem.GetAdjacencyMatrix(product_mol)
    coords = np.array(reactant_coords, dtype=float)
    for i, j in bond_changes:
        target_length = rdMolTransforms.GetBondLength(prod_conf, i, j)
        coords = bond_change(coords, bond_matrix, i, j, target_length)
    return coords


# ---------------------------------------------------------------------------
# Endpoint gradient computation (chain rule through IDPP target distances)
# ---------------------------------------------------------------------------

def compute_endpoint_gradient_total(images: list, d_power: int = 4):
    """
    Gradient of total interior IDPP energy w.r.t. endpoint positions.

    For each interior image with interpolation parameter alpha_k = k/(n-1),
    the target distance for pair (j, l) is linearly interpolated between the
    initial and final endpoint distances. The chain rule gives the contribution
    to the force on each endpoint atom.

    Forces are normalised so the maximum atomic magnitude equals 1.

    Parameters
    ----------
    images : list of ase.Atoms
        Full NEB image list including endpoints.
    d_power : int
        IDPP distance weighting exponent.

    Returns
    -------
    initial_force, final_force : np.ndarray, shape (n_atoms, 3)
    """
    n_images = len(images)
    n_atoms = len(images[0])
    initial_pos = images[0].get_positions()
    final_pos = images[-1].get_positions()

    d_initial = np.zeros((n_atoms, n_atoms))
    d_final = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d_initial[i, j] = d_initial[j, i] = np.linalg.norm(initial_pos[j] - initial_pos[i])
            d_final[i, j] = d_final[j, i] = np.linalg.norm(final_pos[j] - final_pos[i])

    init_force = np.zeros((n_atoms, 3))
    final_force = np.zeros((n_atoms, 3))

    for img_idx in range(1, n_images - 1):
        alpha = img_idx / (n_images - 1)
        cur = images[img_idx].get_positions()
        d_cur = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d_cur[i, j] = d_cur[j, i] = np.linalg.norm(cur[j] - cur[i])
        d_target = (1.0 - alpha) * d_initial + alpha * d_final

        for j in range(n_atoms):
            for k in range(j + 1, n_atoms):
                d_t = d_target[j, k]
                if d_t < 1e-10:
                    continue
                w = 1.0 / (d_t ** d_power)
                err = d_cur[j, k] - d_t

                d_i_jk = d_initial[j, k]
                if d_i_jk > 1e-10:
                    dir_init = (initial_pos[j] - initial_pos[k]) / d_i_jk
                    mag = 2.0 * w * err * (1.0 - alpha)
                    init_force[j] += mag * dir_init
                    init_force[k] -= mag * dir_init

                d_f_jk = d_final[j, k]
                if d_f_jk > 1e-10:
                    dir_fin = (final_pos[j] - final_pos[k]) / d_f_jk
                    mag = 2.0 * w * err * alpha
                    final_force[j] += mag * dir_fin
                    final_force[k] -= mag * dir_fin

    max_f = max(
        np.linalg.norm(init_force, axis=1).max(),
        np.linalg.norm(final_force, axis=1).max(),
    )
    if max_f > 1e-10:
        init_force /= max_f
        final_force /= max_f

    return init_force, final_force


def compute_endpoint_gradient_barrier(images: list, d_power: int = 4):
    """
    Gradient of the MAX interior IDPP energy w.r.t. endpoint positions.

    Same chain-rule derivation as compute_endpoint_gradient_total but only
    the highest-energy interior image contributes.

    Parameters
    ----------
    images : list of ase.Atoms
        Full NEB image list including endpoints.
    d_power : int
        IDPP distance weighting exponent.

    Returns
    -------
    initial_force, final_force : np.ndarray, shape (n_atoms, 3)
    max_img_idx : int
        Index of the highest-energy interior image.
    """
    n_images = len(images)
    n_atoms = len(images[0])
    initial_pos = images[0].get_positions()
    final_pos = images[-1].get_positions()

    d_initial = np.zeros((n_atoms, n_atoms))
    d_final = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d_initial[i, j] = d_initial[j, i] = np.linalg.norm(initial_pos[j] - initial_pos[i])
            d_final[i, j] = d_final[j, i] = np.linalg.norm(final_pos[j] - final_pos[i])

    # Find max-energy interior image
    max_energy = -float("inf")
    max_img_idx = 1
    for img_idx in range(1, n_images - 1):
        alpha = img_idx / (n_images - 1)
        cur = images[img_idx].get_positions()
        d_target = (1.0 - alpha) * d_initial + alpha * d_final
        energy = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d_c = np.linalg.norm(cur[j] - cur[i])
                d_t = d_target[i, j]
                if d_t > 1e-10:
                    energy += (d_c - d_t) ** 2 / (d_t ** d_power)
        if energy > max_energy:
            max_energy = energy
            max_img_idx = img_idx

    # Gradient from max image only
    alpha = max_img_idx / (n_images - 1)
    cur = images[max_img_idx].get_positions()
    d_cur = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d_cur[i, j] = d_cur[j, i] = np.linalg.norm(cur[j] - cur[i])
    d_target = (1.0 - alpha) * d_initial + alpha * d_final

    init_force = np.zeros((n_atoms, 3))
    final_force = np.zeros((n_atoms, 3))

    for j in range(n_atoms):
        for k in range(j + 1, n_atoms):
            d_t = d_target[j, k]
            if d_t < 1e-10:
                continue
            w = 1.0 / (d_t ** d_power)
            err = d_cur[j, k] - d_t

            d_i_jk = d_initial[j, k]
            if d_i_jk > 1e-10:
                dir_init = (initial_pos[j] - initial_pos[k]) / d_i_jk
                mag = 2.0 * w * err * (1.0 - alpha)
                init_force[j] += mag * dir_init
                init_force[k] -= mag * dir_init

            d_f_jk = d_final[j, k]
            if d_f_jk > 1e-10:
                dir_fin = (final_pos[j] - final_pos[k]) / d_f_jk
                mag = 2.0 * w * err * alpha
                final_force[j] += mag * dir_fin
                final_force[k] -= mag * dir_fin

    max_f = max(
        np.linalg.norm(init_force, axis=1).max(),
        np.linalg.norm(final_force, axis=1).max(),
    )
    if max_f > 1e-10:
        init_force /= max_f
        final_force /= max_f

    return init_force, final_force, max_img_idx


def apply_endpoint_adjustment(
    atoms,
    force: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    step_size: float = 2.0,
    max_displacement: float = 2.0,
) -> float:
    """
    Move endpoint atoms along the propagated force, then project back to satisfy
    the bounds matrix. Returns the RMSD of the applied displacement.

    The force array must already be normalised (max atomic magnitude = 1), so
    step_size directly controls the maximum displacement in Ångströms.

    Parameters
    ----------
    atoms : ase.Atoms
        Endpoint to adjust (modified in place).
    force : np.ndarray, shape (n_atoms, 3)
        Normalised propagated force on this endpoint.
    bounds_lower, bounds_upper : np.ndarray, shape (n_atoms, n_atoms)
        Symmetric lower and upper distance bounds.
    step_size : float
        Maximum displacement per atom (Å).
    max_displacement : float
        Hard cap on displacement from the original position (Å).

    Returns
    -------
    float
        RMS displacement applied.
    """
    orig = atoms.get_positions().copy()
    n_atoms = len(orig)

    new_pos = orig + step_size * force

    # Iterative projection onto bounds constraints
    for _ in range(50):
        fixed = 0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = np.linalg.norm(new_pos[j] - new_pos[i])
                lo = bounds_lower[i, j]
                hi = bounds_upper[i, j]
                if d < lo and lo > 0 and d > 1e-10:
                    direction = (new_pos[j] - new_pos[i]) / d
                    correction = (lo - d) / 2.0
                    new_pos[i] -= correction * direction
                    new_pos[j] += correction * direction
                    fixed += 1
                elif d > hi and hi > 0:
                    direction = (new_pos[j] - new_pos[i]) / d
                    correction = (d - hi) / 2.0
                    new_pos[i] += correction * direction
                    new_pos[j] -= correction * direction
                    fixed += 1
        if fixed == 0:
            break

    # Hard cap: total displacement from original ≤ max_displacement
    total_disp = new_pos - orig
    per_atom = np.linalg.norm(total_disp, axis=1)
    if per_atom.max() > max_displacement:
        new_pos = orig + (max_displacement / per_atom.max()) * total_disp

    atoms.set_positions(new_pos)

    final_disp = atoms.get_positions() - orig
    return float(np.sqrt(np.mean(np.sum(final_disp ** 2, axis=1))))
