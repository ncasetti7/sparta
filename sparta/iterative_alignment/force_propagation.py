"""
Propagation of internal coordinate changes through a molecular graph.

When a bond length changes or a structural transformation is applied, the
displacement must propagate to all atoms on the moving side of the bond.
These functions implement that propagation using BFS on the molecular graph.

Also contains endpoint gradient computation and adjustment used during
iterative NEB optimisation.
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

BOUNDS_TOLERANCE = 0.01  # Å
MAX_PROJECTION_ITERATIONS = 500
# bonds further than this (Å) outside their bounds count as broken
BOND_BREAK_TOLERANCE = 0.3


def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
    """Compute symmetric (n, n) pairwise distance matrix from (n, 3) positions."""
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


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

def compute_endpoint_gradient_total(
    images: list,
    d_power: int = 4,
    d_initial: Optional[np.ndarray] = None,
    d_final: Optional[np.ndarray] = None,
):
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
    d_initial : np.ndarray, optional
        Pre-computed (n, n) pairwise distance matrix for images[0]. Computed
        internally when None. Pass a cached value to avoid recomputation.
    d_final : np.ndarray, optional
        Pre-computed (n, n) pairwise distance matrix for images[-1].

    Returns
    -------
    initial_force, final_force : np.ndarray, shape (n_atoms, 3)
    """
    n_images = len(images)
    n_atoms = len(images[0])
    initial_pos = images[0].get_positions()
    final_pos = images[-1].get_positions()

    if d_initial is None:
        d_initial = _pairwise_distances(initial_pos)
    if d_final is None:
        d_final = _pairwise_distances(final_pos)

    i_idx, j_idx = np.triu_indices(n_atoms, k=1)
    d_i_vals = d_initial[i_idx, j_idx]   # (n_pairs,)
    d_f_vals = d_final[i_idx, j_idx]     # (n_pairs,)

    # Precompute endpoint diff vectors — constant across all images
    valid_i = d_i_vals > 1e-10
    valid_f = d_f_vals > 1e-10
    safe_di = np.where(valid_i, d_i_vals, 1.0)
    safe_df = np.where(valid_f, d_f_vals, 1.0)
    diff_initial = initial_pos[i_idx] - initial_pos[j_idx]  # (n_pairs, 3)
    diff_final = final_pos[i_idx] - final_pos[j_idx]        # (n_pairs, 3)

    init_force = np.zeros((n_atoms, 3))
    final_force = np.zeros((n_atoms, 3))

    for img_idx in range(1, n_images - 1):
        alpha = img_idx / (n_images - 1)
        cur_pos = images[img_idx].get_positions()
        d_cur_mat = _pairwise_distances(cur_pos)
        d_target_mat = (1.0 - alpha) * d_initial + alpha * d_final

        d_t_vals = d_target_mat[i_idx, j_idx]
        d_c_vals = d_cur_mat[i_idx, j_idx]

        valid = d_t_vals > 1e-10
        safe_dt = np.where(valid, d_t_vals, 1.0)
        w = np.where(valid, 1.0 / (safe_dt ** d_power), 0.0)
        err = d_c_vals - d_t_vals

        mag_i = np.where(valid & valid_i,
                         2.0 * w * err * (1.0 - alpha) / safe_di, 0.0)
        contrib_i = mag_i[:, None] * diff_initial
        np.add.at(init_force, i_idx, contrib_i)
        np.add.at(init_force, j_idx, -contrib_i)

        mag_f = np.where(valid & valid_f,
                         2.0 * w * err * alpha / safe_df, 0.0)
        contrib_f = mag_f[:, None] * diff_final
        np.add.at(final_force, i_idx, contrib_f)
        np.add.at(final_force, j_idx, -contrib_f)

    max_f = max(
        np.linalg.norm(init_force, axis=1).max(),
        np.linalg.norm(final_force, axis=1).max(),
    )
    if max_f > 1e-10:
        init_force /= max_f
        final_force /= max_f

    return init_force, final_force


def compute_endpoint_gradient_barrier(
    images: list,
    d_power: int = 4,
    d_initial: Optional[np.ndarray] = None,
    d_final: Optional[np.ndarray] = None,
):
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
    d_initial : np.ndarray, optional
        Pre-computed (n, n) pairwise distance matrix for images[0].
    d_final : np.ndarray, optional
        Pre-computed (n, n) pairwise distance matrix for images[-1].

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

    if d_initial is None:
        d_initial = _pairwise_distances(initial_pos)
    if d_final is None:
        d_final = _pairwise_distances(final_pos)

    i_idx, j_idx = np.triu_indices(n_atoms, k=1)

    # Find max-energy interior image (vectorized energy per image)
    max_energy = -float("inf")
    max_img_idx = 1
    for img_idx in range(1, n_images - 1):
        alpha = img_idx / (n_images - 1)
        cur_pos = images[img_idx].get_positions()
        d_cur_mat = _pairwise_distances(cur_pos)
        d_target_mat = (1.0 - alpha) * d_initial + alpha * d_final

        d_t_vals = d_target_mat[i_idx, j_idx]
        d_c_vals = d_cur_mat[i_idx, j_idx]

        valid = d_t_vals > 1e-10
        safe_dt = np.where(valid, d_t_vals, 1.0)
        w = np.where(valid, 1.0 / (safe_dt ** d_power), 0.0)
        energy = float(np.sum(w * (d_c_vals - d_t_vals) ** 2))

        if energy > max_energy:
            max_energy = energy
            max_img_idx = img_idx

    # Gradient from max-energy image only
    alpha = max_img_idx / (n_images - 1)
    cur_pos = images[max_img_idx].get_positions()
    d_cur_mat = _pairwise_distances(cur_pos)
    d_target_mat = (1.0 - alpha) * d_initial + alpha * d_final

    d_t_vals = d_target_mat[i_idx, j_idx]
    d_c_vals = d_cur_mat[i_idx, j_idx]
    d_i_vals = d_initial[i_idx, j_idx]
    d_f_vals = d_final[i_idx, j_idx]

    valid = d_t_vals > 1e-10
    safe_dt = np.where(valid, d_t_vals, 1.0)
    w = np.where(valid, 1.0 / (safe_dt ** d_power), 0.0)
    err = d_c_vals - d_t_vals

    init_force = np.zeros((n_atoms, 3))
    final_force = np.zeros((n_atoms, 3))

    valid_i = valid & (d_i_vals > 1e-10)
    safe_di = np.where(d_i_vals > 1e-10, d_i_vals, 1.0)
    mag_i = np.where(valid_i, 2.0 * w * err * (1.0 - alpha) / safe_di, 0.0)
    diff_initial = initial_pos[i_idx] - initial_pos[j_idx]
    contrib_i = mag_i[:, None] * diff_initial
    np.add.at(init_force, i_idx, contrib_i)
    np.add.at(init_force, j_idx, -contrib_i)

    valid_f = valid & (d_f_vals > 1e-10)
    safe_df = np.where(d_f_vals > 1e-10, d_f_vals, 1.0)
    mag_f = np.where(valid_f, 2.0 * w * err * alpha / safe_df, 0.0)
    diff_final = final_pos[i_idx] - final_pos[j_idx]
    contrib_f = mag_f[:, None] * diff_final
    np.add.at(final_force, i_idx, contrib_f)
    np.add.at(final_force, j_idx, -contrib_f)

    max_f = max(
        np.linalg.norm(init_force, axis=1).max(),
        np.linalg.norm(final_force, axis=1).max(),
    )
    if max_f > 1e-10:
        init_force /= max_f
        final_force /= max_f

    return init_force, final_force, max_img_idx


def bound_violation(
    positions: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    pairs: Optional[np.ndarray] = None,
) -> float:
    """
    Return the largest distance (Å) by which an atom pair lies outside its bounds.

    Pairs whose lower or upper bound is zero are unconstrained on that side.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atomic positions.
    bounds_lower, bounds_upper : np.ndarray, shape (n_atoms, n_atoms)
        Symmetric lower and upper distance bounds.
    pairs : np.ndarray, shape (n_pairs, 2), optional
        Atom pairs to check, such as an endpoint's bonds; all pairs by default.

    Returns
    -------
    float
        Largest bound violation, or 0 when every checked pair is within its bounds.
    """
    if pairs is None:
        i_idx, j_idx = np.triu_indices(len(positions), k=1)
    else:
        i_idx, j_idx = np.asarray(pairs, dtype=int).reshape(-1, 2).T
    d = np.linalg.norm(positions[j_idx] - positions[i_idx], axis=1)
    lo = bounds_lower[i_idx, j_idx]
    hi = bounds_upper[i_idx, j_idx]
    below = np.where(lo > 0, lo - d, 0.0)
    above = np.where(hi > 0, d - hi, 0.0)
    return float(max(below.max(initial=0.0), above.max(initial=0.0), 0.0))


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
        Hard cap on the proposed displacement from the original position (Å).

    Returns
    -------
    float
        RMS displacement applied.
    """
    orig = atoms.get_positions().copy()
    n_atoms = len(orig)

    step = step_size * force
    per_atom = np.linalg.norm(step, axis=1)
    if per_atom.max(initial=0.0) > max_displacement:
        step *= max_displacement / per_atom.max()
    new_pos = orig + step

    # Precompute pair indices and bound values (constant across projection iters)
    i_idx, j_idx = np.triu_indices(n_atoms, k=1)
    lo_pairs = bounds_lower[i_idx, j_idx]
    hi_pairs = bounds_upper[i_idx, j_idx]

    # Jacobi-style iterative projection onto distance bounds, averaged per atom
    best_pos = new_pos.copy()
    best_worst = float("inf")
    for _ in range(MAX_PROJECTION_ITERATIONS):
        vec = new_pos[j_idx] - new_pos[i_idx]          # (n_pairs, 3), points i→j
        d = np.sqrt((vec * vec).sum(axis=-1))            # (n_pairs,)

        lower_viol = (d < lo_pairs) & (lo_pairs > 0) & (d > 1e-10)
        upper_viol = (d > hi_pairs) & (hi_pairs > 0)
        worst = max(
            np.where(lower_viol, lo_pairs - d, 0.0).max(initial=0.0),
            np.where(upper_viol, d - hi_pairs, 0.0).max(initial=0.0),
        )
        if worst < best_worst:
            best_worst = worst
            best_pos = new_pos.copy()
        if worst <= BOUNDS_TOLERANCE:
            break

        safe_d = np.where(d > 1e-10, d, 1.0)
        unit = vec / safe_d[:, None]                     # (n_pairs, 3)

        # Half-step corrections toward constraint satisfaction
        corr_lower = np.where(lower_viol, (lo_pairs - d) / 2.0, 0.0)
        corr_upper = np.where(upper_viol, (d - hi_pairs) / 2.0, 0.0)

        # delta_i[p] moves atom i_idx[p]:
        #   lower viol → away from j  (-corr_lower * unit)
        #   upper viol → toward j     (+corr_upper * unit)
        delta_i = (corr_upper - corr_lower)[:, None] * unit  # (n_pairs, 3)

        active = lower_viol | upper_viol
        counts = np.zeros(n_atoms)
        np.add.at(counts, i_idx[active], 1.0)
        np.add.at(counts, j_idx[active], 1.0)

        displacement = np.zeros_like(new_pos)
        np.add.at(displacement, i_idx, delta_i)
        np.add.at(displacement, j_idx, -delta_i)
        new_pos = new_pos + displacement / np.maximum(counts, 1.0)[:, None]
        if not np.all(np.isfinite(new_pos)):
            break

    atoms.set_positions(best_pos)

    final_disp = atoms.get_positions() - orig
    return float(np.sqrt(np.mean(np.sum(final_disp ** 2, axis=1))))
