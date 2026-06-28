"""
Distance geometry optimization with analytical gradients.

Two variants:
- optimize_to_bounds_matrix: minimize pairwise distance bound violations.
- optimize_to_bounds_matrix_and_rmsd: same, plus a term keeping coordinates
  close to a reference structure.
- optimize_to_distance_matrix: convenience wrapper for exact-distance targets.
"""

import numpy as np
from scipy.optimize import minimize


def _parse_bounds_matrix(bounds_matrix: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Convert an RDKit asymmetric bounds matrix (n, n) or an explicit (n, n, 2)
    matrix into a unified (n, n, 2) array where [..., 0] is the lower bound and
    [..., 1] is the upper bound.
    """
    bm = np.array(bounds_matrix, dtype=float)
    if bm.ndim == 2:
        if bm.shape != (n_atoms, n_atoms):
            raise ValueError(
                f"Bounds matrix shape {bm.shape} does not match {n_atoms} atoms."
            )
        bounds = np.empty((n_atoms, n_atoms, 2), dtype=float)
        bounds[:, :, 0] = np.minimum(bm, bm.T)
        bounds[:, :, 1] = np.maximum(bm, bm.T)
        return bounds
    elif bm.ndim == 3 and bm.shape[2] == 2:
        if bm.shape[:2] != (n_atoms, n_atoms):
            raise ValueError(
                f"Bounds matrix shape {bm.shape[:2]} does not match {n_atoms} atoms."
            )
        return bm
    else:
        raise ValueError(
            f"Bounds matrix must have shape (n, n) or (n, n, 2), got {bm.shape}."
        )


def _prepare_pair_bounds(bounds: np.ndarray):
    """Extract upper-triangle atom pairs and their lower/upper bounds."""
    pair_i, pair_j = np.triu_indices(bounds.shape[0], k=1)
    return pair_i, pair_j, bounds[pair_i, pair_j, 0], bounds[pair_i, pair_j, 1]


def _bounds_error_and_gradient(
    coords: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weight_lower: float,
    weight_upper: float,
):
    """Return the distance-bounds penalty and Cartesian gradient."""
    pair_vectors = coords[pair_i] - coords[pair_j]
    distances = np.sqrt(np.einsum("ij,ij->i", pair_vectors, pair_vectors))

    lower_mask = distances < lower
    # Preserve the scalar implementation's if/elif behavior for malformed bounds.
    upper_mask = (~lower_mask) & (distances > upper)
    lower_error = np.where(lower_mask, lower - distances, 0.0)
    upper_error = np.where(upper_mask, distances - upper, 0.0)
    error = (
        weight_lower * np.dot(lower_error, lower_error)
        + weight_upper * np.dot(upper_error, upper_error)
    )

    nonzero = distances > 1e-10
    safe_distances = np.where(nonzero, distances, 1.0)
    factors = np.zeros_like(distances)
    factors[lower_mask & nonzero] = (
        -2.0
        * weight_lower
        * lower_error[lower_mask & nonzero]
        / safe_distances[lower_mask & nonzero]
    )
    factors[upper_mask & nonzero] = (
        2.0
        * weight_upper
        * upper_error[upper_mask & nonzero]
        / safe_distances[upper_mask & nonzero]
    )

    contributions = factors[:, None] * pair_vectors
    gradient = np.zeros_like(coords)
    np.add.at(gradient, pair_i, contributions)
    np.add.at(gradient, pair_j, -contributions)
    return float(error), gradient


def optimize_to_bounds_matrix(
    coords: np.ndarray,
    bounds_matrix: np.ndarray,
    method: str = "L-BFGS-B",
    max_iter: int = 1000,
    weight_lower: float = 1.0,
    weight_upper: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimize 3D coordinates to satisfy a distance bounds matrix.

    Minimizes squared violations of the form:
        sum_{i<j} [ w_lower * max(0, lower_ij - d_ij)^2
                   + w_upper * max(0, d_ij - upper_ij)^2 ]

    Analytical gradients are used for efficiency.

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
    bounds_matrix : np.ndarray, shape (n_atoms, n_atoms) or (n_atoms, n_atoms, 2)
        RDKit asymmetric format (lower in lower triangle, upper in upper triangle)
        or explicit (lower, upper) format.
    method : str
        scipy.optimize.minimize method (default: 'L-BFGS-B').
    max_iter : int
        Maximum iterations.
    weight_lower, weight_upper : float
        Penalty weights for lower/upper bound violations.
    verbose : bool
        Print progress if True.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Optimized coordinates.
    """
    coords = np.array(coords, dtype=float)
    if coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_atoms, 3).")
    n_atoms = coords.shape[0]
    bounds = _parse_bounds_matrix(bounds_matrix, n_atoms)
    pair_data = _prepare_pair_bounds(bounds)

    def objective_and_grad(x):
        cur = x.reshape(n_atoms, 3)
        total, grad = _bounds_error_and_gradient(
            cur, *pair_data, weight_lower, weight_upper
        )
        if verbose:
            print(f"Objective: {total:.6f}")
        return total, grad.flatten()

    result = minimize(
        objective_and_grad, coords.flatten(), method=method, jac=True,
        options={"maxiter": max_iter},
    )
    if verbose:
        print(f"Optimization {'converged' if result.success else 'did not converge'}")
        print(f"Final objective: {result.fun:.6f}, iterations: {result.nit}")
    return result.x.reshape(n_atoms, 3)


def optimize_to_bounds_matrix_and_rmsd(
    coords: np.ndarray,
    bounds_matrix: np.ndarray,
    reference_coords: np.ndarray,
    method: str = "L-BFGS-B",
    max_iter: int = 1000,
    weight_lower: float = 1.0,
    weight_upper: float = 1.0,
    weight_rmsd: float = 0.1,
    chiral_constraints=None,
    weight_chiral: float = 5.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimize 3D coordinates to satisfy a bounds matrix while staying close to
    reference_coords (weighted sum-of-squared-deviations term).

    Objective:
        bounds_violations + weight_rmsd * sum_i ||r_i - ref_i||^2
                          + weight_chiral * sum_chiral max(0, margin - s*V)^2

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
    bounds_matrix : np.ndarray, shape (n_atoms, n_atoms) or (n_atoms, n_atoms, 2)
    reference_coords : np.ndarray, shape (n_atoms, 3)
    weight_rmsd : float
        Should typically be smaller than weight_lower/weight_upper so bounds
        satisfaction takes priority.
    chiral_constraints : list of (center, n0, n1, n2, target_sign, margin), optional
        Signed chiral-volume targets, with V = (r_n0 - r_c) . ((r_n1 - r_c) x
        (r_n2 - r_c)).  The one-sided penalty drives ``target_sign * V`` up to at
        least ``margin``, so it enforces — and will flip a wrong-handed start into
        — the desired handedness at each stereocenter.  Distance geometry alone is
        chirality-blind, so this is what makes a specified configuration stick.
    weight_chiral : float
        Weight on the chiral-volume penalty (units differ from the distance terms,
        so calibrate it against the volume scale ~ a few cubic ångström).

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
    """
    coords = np.array(coords, dtype=float)
    reference_coords = np.array(reference_coords, dtype=float)
    if coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_atoms, 3).")
    if reference_coords.shape != coords.shape:
        raise ValueError("reference_coords must have the same shape as coords.")
    n_atoms = coords.shape[0]
    bounds = _parse_bounds_matrix(bounds_matrix, n_atoms)
    pair_data = _prepare_pair_bounds(bounds)

    def objective_and_grad(x):
        cur = x.reshape(n_atoms, 3)
        bounds_err, grad = _bounds_error_and_gradient(
            cur, *pair_data, weight_lower, weight_upper
        )
        rmsd_err = weight_rmsd * np.sum((cur - reference_coords) ** 2)
        grad += 2.0 * weight_rmsd * (cur - reference_coords)
        chiral_err = 0.0
        if chiral_constraints:
            for (c, i, j, k, s, m) in chiral_constraints:
                a = cur[i] - cur[c]
                b = cur[j] - cur[c]
                e = cur[k] - cur[c]
                vol = float(a @ np.cross(b, e))
                viol = m - s * vol
                if viol > 0.0:
                    chiral_err += weight_chiral * viol * viol
                    # d(penalty)/dV = -2 w viol s; dV/dr via vector-triple-product
                    g = -2.0 * weight_chiral * viol * s
                    gi = np.cross(b, e)   # dV/dr_i
                    gj = np.cross(e, a)   # dV/dr_j
                    gk = np.cross(a, b)   # dV/dr_k
                    grad[i] += g * gi
                    grad[j] += g * gj
                    grad[k] += g * gk
                    grad[c] -= g * (gi + gj + gk)   # V is translation-invariant
        total = bounds_err + rmsd_err + chiral_err
        if verbose:
            print(f"Objective: {total:.6f} (bounds: {bounds_err:.6f}, "
                  f"rmsd: {rmsd_err:.6f}, chiral: {chiral_err:.6f})")
        return total, grad.flatten()

    result = minimize(
        objective_and_grad, coords.flatten(), method=method, jac=True,
        options={"maxiter": max_iter},
    )
    if verbose:
        print(f"Optimization {'converged' if result.success else 'did not converge'}")
        print(f"Final objective: {result.fun:.6f}, iterations: {result.nit}")
    return result.x.reshape(n_atoms, 3)


def optimize_to_distance_matrix(
    coords: np.ndarray,
    target_distance_matrix: np.ndarray,
    method: str = "L-BFGS-B",
    max_iter: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimize coordinates to match a target pairwise distance matrix exactly
    (lower bound = upper bound = target distance).

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
    target_distance_matrix : np.ndarray, shape (n_atoms, n_atoms)
        Symmetric matrix with zeros on the diagonal.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
    """
    coords = np.array(coords, dtype=float)
    tdm = np.array(target_distance_matrix, dtype=float)
    n_atoms = coords.shape[0]
    if tdm.shape != (n_atoms, n_atoms):
        raise ValueError(
            f"Distance matrix shape {tdm.shape} does not match {n_atoms} atoms."
        )
    bounds_matrix = np.zeros((n_atoms, n_atoms, 2))
    bounds_matrix[:, :, 0] = tdm
    bounds_matrix[:, :, 1] = tdm
    return optimize_to_bounds_matrix(
        coords, bounds_matrix, method=method, max_iter=max_iter, verbose=verbose
    )
