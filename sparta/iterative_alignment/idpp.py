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

    def _compute_target_distances(self) -> np.ndarray:
        """Linearly interpolate pairwise distances between initial and final images."""
        n_atoms = self.initial_positions.shape[0]
        d_initial = np.zeros((n_atoms, n_atoms))
        d_final = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d_initial[i, j] = d_initial[j, i] = np.linalg.norm(
                    self.initial_positions[j] - self.initial_positions[i]
                )
                d_final[i, j] = d_final[j, i] = np.linalg.norm(
                    self.final_positions[j] - self.final_positions[i]
                )
        return (1.0 - self.alpha) * d_initial + self.alpha * d_final

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = self.atoms.get_positions()
        n_atoms = len(positions)

        current_distances = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = np.linalg.norm(positions[j] - positions[i])
                current_distances[i, j] = current_distances[j, i] = d

        energy = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d_ij = current_distances[i, j]
                d_t = self.target_distances[i, j]
                if d_t < 1e-10:
                    continue
                w = 1.0 / (d_t ** self.d_power)
                energy += w * (d_ij - d_t) ** 2
        self.results["energy"] = energy

        if "forces" in properties:
            forces = np.zeros((n_atoms, 3))
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i == j:
                        continue
                    d_ij = current_distances[i, j]
                    d_t = self.target_distances[i, j]
                    if d_t < 1e-10 or d_ij < 1e-10:
                        continue
                    w = 1.0 / (d_t ** self.d_power)
                    diff = positions[i] - positions[j]
                    forces[i] += -2.0 * w * (d_ij - d_t) / d_ij * diff
            self.results["forces"] = forces


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
