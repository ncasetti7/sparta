"""
Image-Dependent Pair Potential (IDPP) calculator for ASE.

IDPP improves NEB initial paths by interpolating pairwise distances rather than
Cartesian positions. The target distance for each atom pair at image k is linearly
interpolated between the initial and final image distances.

Reference: Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).
"""

from typing import List

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class IDPP(Calculator):
    """
    IDPP calculator for a single NEB image.

    Energy:
        E = sum_{i<j} w_ij * (d_ij - d_ij^target)^2
    where w_ij = 1 / (d_ij^target)^d_power and d_ij^target is the linearly
    interpolated target distance for this image.

    Parameters
    ----------
    initial_positions : np.ndarray, shape (n_atoms, 3)
        Positions of the initial (reactant) image.
    final_positions : np.ndarray, shape (n_atoms, 3)
        Positions of the final (product) image.
    alpha : float
        Interpolation parameter in [0, 1]. For NEB with n_images total images,
        image k uses alpha = k / (n_images - 1).
    d_power : int
        Distance weighting exponent (default: 4).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        initial_positions: np.ndarray,
        final_positions: np.ndarray,
        alpha: float,
        d_power: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_positions = np.array(initial_positions, dtype=float)
        self.final_positions = np.array(final_positions, dtype=float)
        self.alpha = float(alpha)
        self.d_power = int(d_power)

        n_atoms = self.initial_positions.shape[0]
        if self.final_positions.shape != (n_atoms, 3):
            raise ValueError(
                "initial_positions and final_positions must have the same shape."
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}.")

        self.target_distances = self._compute_target_distances()
        self._pair_i, self._pair_j = np.triu_indices(n_atoms, k=1)
        self._weights = np.zeros_like(self.target_distances)
        valid_targets = self.target_distances > 1e-10
        self._weights[valid_targets] = (
            1.0 / self.target_distances[valid_targets] ** self.d_power
        )
        self._pair_targets = self.target_distances[self._pair_i, self._pair_j]
        self._pair_weights = self._weights[self._pair_i, self._pair_j]

    def _compute_target_distances(self) -> np.ndarray:
        """Linearly interpolate pairwise distances between initial and final images."""
        diff_i = self.initial_positions[:, None, :] - self.initial_positions[None, :, :]
        d_initial = np.sqrt((diff_i * diff_i).sum(axis=-1))
        diff_f = self.final_positions[:, None, :] - self.final_positions[None, :, :]
        d_final = np.sqrt((diff_f * diff_f).sum(axis=-1))
        return (1.0 - self.alpha) * d_initial + self.alpha * d_final

    def check_state(self, atoms, tol=1e-15):
        """Invalidate cached IDPP values only when Cartesian positions change."""
        if self.atoms is None or self.atoms.positions.shape != atoms.positions.shape:
            return ["positions"]
        if np.array_equal(self.atoms.positions, atoms.positions):
            return []
        return ["positions"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = self.atoms.get_positions()
        # Vectorized pairwise distance matrix: diff[i,j] = pos[i] - pos[j]
        diff = positions[:, None, :] - positions[None, :, :]   # (n, n, 3)
        current_distances = np.sqrt((diff * diff).sum(axis=-1))  # (n, n)

        d_t = self.target_distances  # (n, n)

        # Energy over upper triangle (i < j)
        d_ij = current_distances[self._pair_i, self._pair_j]
        self.results["energy"] = float(
            np.sum(self._pair_weights * (d_ij - self._pair_targets) ** 2)
        )

        if "forces" in properties:
            # scale[i,j] = -2 * w[i,j] * (d[i,j] - d_t[i,j]) / d[i,j]
            # forces[i] = sum_j scale[i,j] * diff[i,j]  where diff[i,j] = pos[i] - pos[j]
            valid_both = (d_t > 1e-10) & (current_distances > 1e-10)  # (n, n)
            safe_d = np.where(valid_both, current_distances, 1.0)
            weights = np.where(valid_both, self._weights, 0.0)
            scale = -2.0 * weights * (current_distances - d_t) / safe_d  # (n, n)
            self.results["forces"] = np.einsum("ij,ijk->ik", scale, diff)  # (n, 3)


def create_idpp_calculators(
    initial_atoms,
    final_atoms,
    n_images: int,
    d_power: int = 4,
) -> List[IDPP]:
    """
    Create one IDPP calculator per intermediate NEB image.

    Parameters
    ----------
    initial_atoms : ase.Atoms
        Initial (reactant) image.
    final_atoms : ase.Atoms
        Final (product) image.
    n_images : int
        Total number of images, including initial and final.
    d_power : int
        Distance weighting exponent.

    Returns
    -------
    list of IDPP
        Length is n_images - 2 (one per intermediate image).
    """
    if n_images < 3:
        raise ValueError(f"n_images must be at least 3, got {n_images}.")
    initial_pos = initial_atoms.get_positions()
    final_pos = final_atoms.get_positions()
    return [
        IDPP(initial_pos, final_pos, alpha=i / (n_images - 1), d_power=d_power)
        for i in range(1, n_images - 1)
    ]
