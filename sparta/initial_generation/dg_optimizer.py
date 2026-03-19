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
from scipy.spatial.distance import cdist


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
        bounds = np.zeros((n_atoms, n_atoms, 2))
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    bounds[i, j, 0] = min(bm[i, j], bm[j, i])
                    bounds[i, j, 1] = max(bm[i, j], bm[j, i])
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

    def objective(x):
        cur = x.reshape(n_atoms, 3)
        dists = cdist(cur, cur)
        total = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = dists[i, j]
                lo, hi = bounds[i, j, 0], bounds[i, j, 1]
                if d < lo:
                    total += weight_lower * (lo - d) ** 2
                elif d > hi:
                    total += weight_upper * (d - hi) ** 2
        if verbose:
            print(f"Objective: {total:.6f}")
        return total

    def objective_grad(x):
        cur = x.reshape(n_atoms, 3)
        grad = np.zeros_like(cur)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                vec = cur[j] - cur[i]
                d = np.linalg.norm(vec)
                if d < 1e-10:
                    continue
                lo, hi = bounds[i, j, 0], bounds[i, j, 1]
                if d < lo:
                    factor = -2.0 * weight_lower * (lo - d) / d
                elif d > hi:
                    factor = 2.0 * weight_upper * (d - hi) / d
                else:
                    continue
                contribution = factor * (cur[i] - cur[j])
                grad[i] += contribution
                grad[j] -= contribution
        return grad.flatten()

    result = minimize(
        objective, coords.flatten(), method=method, jac=objective_grad,
        options={"maxiter": max_iter, "disp": verbose},
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
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimize 3D coordinates to satisfy a bounds matrix while staying close to
    reference_coords (weighted sum-of-squared-deviations term).

    Objective:
        bounds_violations + weight_rmsd * sum_i ||r_i - ref_i||^2

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
    bounds_matrix : np.ndarray, shape (n_atoms, n_atoms) or (n_atoms, n_atoms, 2)
    reference_coords : np.ndarray, shape (n_atoms, 3)
    weight_rmsd : float
        Should typically be smaller than weight_lower/weight_upper so bounds
        satisfaction takes priority.

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

    def objective(x):
        cur = x.reshape(n_atoms, 3)
        dists = cdist(cur, cur)
        bounds_err = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = dists[i, j]
                lo, hi = bounds[i, j, 0], bounds[i, j, 1]
                if d < lo:
                    bounds_err += weight_lower * (lo - d) ** 2
                elif d > hi:
                    bounds_err += weight_upper * (d - hi) ** 2
        rmsd_err = weight_rmsd * np.sum((cur - reference_coords) ** 2)
        total = bounds_err + rmsd_err
        if verbose:
            print(f"Objective: {total:.6f} (bounds: {bounds_err:.6f}, rmsd: {rmsd_err:.6f})")
        return total

    def objective_grad(x):
        cur = x.reshape(n_atoms, 3)
        grad = np.zeros_like(cur)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                vec = cur[j] - cur[i]
                d = np.linalg.norm(vec)
                if d < 1e-10:
                    continue
                lo, hi = bounds[i, j, 0], bounds[i, j, 1]
                if d < lo:
                    factor = -2.0 * weight_lower * (lo - d) / d
                elif d > hi:
                    factor = 2.0 * weight_upper * (d - hi) / d
                else:
                    factor = None
                if factor is not None:
                    contribution = factor * (cur[i] - cur[j])
                    grad[i] += contribution
                    grad[j] -= contribution
        grad += 2.0 * weight_rmsd * (cur - reference_coords)
        return grad.flatten()

    result = minimize(
        objective, coords.flatten(), method=method, jac=objective_grad,
        options={"maxiter": max_iter, "disp": verbose},
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
