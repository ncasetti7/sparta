"""Tests for sparta.initial_generation.dg_optimizer."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from sparta.initial_generation.dg_optimizer import (
    _parse_bounds_matrix,
    optimize_to_bounds_matrix,
    optimize_to_bounds_matrix_and_rmsd,
    optimize_to_distance_matrix,
)


# ---------------------------------------------------------------------------
# _parse_bounds_matrix
# ---------------------------------------------------------------------------

class TestParseBoundsMatrix:
    def test_2d_rdkit_format_parsed_correctly(self):
        # RDKit format: lower in lower triangle, upper in upper triangle
        bm = np.array([[0.0, 2.0],
                       [1.0, 0.0]])  # lower=1.0, upper=2.0 for pair (0,1)
        bounds = _parse_bounds_matrix(bm, 2)
        assert bounds.shape == (2, 2, 2)
        assert bounds[0, 1, 0] == 1.0  # lower bound
        assert bounds[0, 1, 1] == 2.0  # upper bound

    def test_3d_passthrough(self):
        bm = np.zeros((3, 3, 2))
        bm[0, 1] = bm[1, 0] = [1.0, 3.0]
        bm[1, 2] = bm[2, 1] = [1.2, 2.5]
        result = _parse_bounds_matrix(bm, 3)
        np.testing.assert_array_equal(result, bm)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            _parse_bounds_matrix(np.zeros((3, 4)), 3)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            _parse_bounds_matrix(np.zeros((3,)), 3)


# ---------------------------------------------------------------------------
# optimize_to_bounds_matrix
# ---------------------------------------------------------------------------

class TestOptimizeToBoundsMatrix:
    def _make_rdkit_bm(self, n, lo, hi):
        """Simple uniform bounds matrix in RDKit format."""
        bm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i > j:
                    bm[i, j] = lo
                elif i < j:
                    bm[i, j] = hi
        return bm

    def test_already_satisfied_coords_unchanged(self):
        # 2 atoms 1.5 Å apart, bounds 1.0–2.0 Å → already satisfied
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        bm = self._make_rdkit_bm(2, 1.0, 2.0)
        result = optimize_to_bounds_matrix(coords, bm, max_iter=50)
        d = np.linalg.norm(result[0] - result[1])
        assert 1.0 <= d <= 2.0

    def test_violation_reduced(self):
        # 2 atoms too close (0.5 Å), lower bound 1.0 Å → optimizer should push apart
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        bm = self._make_rdkit_bm(2, 1.0, 3.0)
        result = optimize_to_bounds_matrix(coords, bm, max_iter=500)
        d = np.linalg.norm(result[0] - result[1])
        assert d >= 0.9  # significant improvement toward bound

    def test_output_shape(self):
        coords = np.random.default_rng(0).standard_normal((4, 3))
        bm = self._make_rdkit_bm(4, 1.0, 5.0)
        result = optimize_to_bounds_matrix(coords, bm)
        assert result.shape == (4, 3)

    def test_wrong_coord_shape_raises(self):
        with pytest.raises(ValueError):
            optimize_to_bounds_matrix(np.zeros((3, 2)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# optimize_to_bounds_matrix_and_rmsd
# ---------------------------------------------------------------------------

class TestOptimizeToBoundsMatrixAndRmsd:
    def test_stays_close_to_reference(self):
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((3, 3)) * 2.0
        coords = ref + rng.standard_normal((3, 3)) * 0.1
        # Very loose bounds so RMSD term dominates
        bm = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i > j:
                    bm[i, j] = 0.0
                elif i < j:
                    bm[i, j] = 100.0
        result = optimize_to_bounds_matrix_and_rmsd(
            coords, bm, ref, weight_rmsd=1.0, max_iter=200
        )
        rmsd = np.sqrt(np.mean(np.sum((result - ref) ** 2, axis=1)))
        assert rmsd < 0.5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            optimize_to_bounds_matrix_and_rmsd(
                np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((4, 3))
            )

    def test_wrong_coord_shape_raises(self):
        with pytest.raises(ValueError):
            optimize_to_bounds_matrix_and_rmsd(
                np.zeros((3, 2)), np.zeros((3, 3)), np.zeros((3, 2))
            )


# ---------------------------------------------------------------------------
# optimize_to_distance_matrix
# ---------------------------------------------------------------------------

class TestOptimizeToDistanceMatrix:
    def test_converges_to_target_distances(self):
        # Triangle with sides 1, 1, sqrt(2)
        target = np.array([
            [0.0, 1.0, np.sqrt(2)],
            [1.0, 0.0, 1.0],
            [np.sqrt(2), 1.0, 0.0],
        ])
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        result = optimize_to_distance_matrix(coords, target, max_iter=2000)
        achieved = cdist(result, result)
        np.testing.assert_allclose(achieved, target, atol=0.15)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            optimize_to_distance_matrix(np.zeros((3, 3)), np.zeros((4, 4)))
