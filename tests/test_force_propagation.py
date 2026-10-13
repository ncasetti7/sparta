"""Tests for sparta.iterative_alignment.force_propagation."""

import numpy as np
import pytest
import ase

from sparta.iterative_alignment.force_propagation import (
    BOUNDS_TOLERANCE,
    bound_violation,
    _get_connected_fragment,
    _pairwise_distances,
    apply_endpoint_adjustment,
    bond_change,
    compute_endpoint_gradient_barrier,
    compute_endpoint_gradient_total,
    transform_mol,
)
from sparta.cheminformatics import make_mol, add_coords_to_mol


# ---------------------------------------------------------------------------
# _pairwise_distances
# ---------------------------------------------------------------------------

class TestPairwiseDistances:
    def test_diagonal_is_zero(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        d = _pairwise_distances(pos)
        np.testing.assert_array_equal(np.diag(d), np.zeros(3))

    def test_known_distance(self):
        pos = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # 3-4-5 triangle
        d = _pairwise_distances(pos)
        assert abs(d[0, 1] - 5.0) < 1e-10
        assert abs(d[1, 0] - 5.0) < 1e-10

    def test_symmetric(self):
        np.random.seed(1)
        pos = np.random.randn(6, 3)
        d = _pairwise_distances(pos)
        np.testing.assert_array_equal(d, d.T)

    def test_shape(self):
        pos = np.random.randn(7, 3)
        d = _pairwise_distances(pos)
        assert d.shape == (7, 7)


# ---------------------------------------------------------------------------
# _get_connected_fragment
# ---------------------------------------------------------------------------

class TestGetConnectedFragment:
    def _chain_adj(self):
        """Adjacency matrix for chain 0-1-2-3."""
        a = np.zeros((4, 4), dtype=int)
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            a[i, j] = a[j, i] = 1
        return a

    def test_moving_side_contains_moving_atom(self):
        adj = self._chain_adj()
        # Fixed=0, moving=1 → fragment on moving side of bond 0-1 = {1, 2, 3}
        frag = _get_connected_fragment(adj, fixed_atom=0, moving_atom=1)
        assert frag == {1, 2, 3}

    def test_fixed_atom_not_in_fragment(self):
        adj = self._chain_adj()
        frag = _get_connected_fragment(adj, fixed_atom=0, moving_atom=1)
        assert 0 not in frag

    def test_mid_bond_splits_chain(self):
        adj = self._chain_adj()
        # Fixed=1, moving=2 → {2, 3}
        frag = _get_connected_fragment(adj, fixed_atom=1, moving_atom=2)
        assert frag == {2, 3}

    def test_end_atom_only(self):
        adj = self._chain_adj()
        # Fixed=2, moving=3 → {3}
        frag = _get_connected_fragment(adj, fixed_atom=2, moving_atom=3)
        assert frag == {3}


# ---------------------------------------------------------------------------
# bond_change
# ---------------------------------------------------------------------------

class TestBondChange:
    def _chain4_adj(self):
        a = np.zeros((4, 4), dtype=int)
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            a[i, j] = a[j, i] = 1
        return a

    def test_bond_reaches_target_length(self):
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ])
        adj = self._chain4_adj()
        new_coords = bond_change(coords, adj, atom_i=0, atom_j=1, target_length=2.0)
        d = np.linalg.norm(new_coords[1] - new_coords[0])
        assert abs(d - 2.0) < 1e-8

    def test_downstream_atoms_also_move(self):
        """Atoms 2 and 3 (on the moving side of bond 0-1) should also translate."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        adj = self._chain4_adj()
        original_spacing_23 = np.linalg.norm(coords[3] - coords[2])
        new_coords = bond_change(coords, adj, atom_i=0, atom_j=1, target_length=1.5)
        new_spacing_23 = np.linalg.norm(new_coords[3] - new_coords[2])
        # Internal spacing within moving fragment should be preserved
        assert abs(new_spacing_23 - original_spacing_23) < 1e-8

    def test_zero_length_bond_raises(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        adj = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError):
            bond_change(coords, adj, 0, 1, 1.5)


# ---------------------------------------------------------------------------
# transform_mol
# ---------------------------------------------------------------------------

class TestTransformMol:
    def test_transforms_single_bond(self):
        # FCl molecule: apply bond change so F-Cl reaches product length
        mol = make_mol("[F:1][Cl:2]")
        coords_reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Set product mol with a longer bond
        prod_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mol = add_coords_to_mol(prod_coords, mol)
        result = transform_mol(coords_reactant, mol, [(0, 1)])
        d = np.linalg.norm(result[1] - result[0])
        assert abs(d - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_endpoint_gradient_total
# ---------------------------------------------------------------------------

class TestEndpointGradientTotal:
    def test_output_shape(self, two_atom_ase_images):
        init_f, final_f = compute_endpoint_gradient_total(two_atom_ase_images)
        assert init_f.shape == (2, 3)
        assert final_f.shape == (2, 3)

    def _perturbed_images(self):
        """Images where the middle frame deviates from the linear interpolation."""
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        images = []
        for k in range(5):
            alpha = k / 4
            pos = (1 - alpha) * initial + alpha * final
            if k == 2:
                pos = pos + np.array([[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]])
            images.append(ase.Atoms(symbols=["F", "F"], positions=pos))
        return images

    def test_nonzero_gradient_when_perturbed(self):
        """Force should be nonzero when images deviate from target distances."""
        images = self._perturbed_images()
        init_f, final_f = compute_endpoint_gradient_total(images)
        total_norm = np.linalg.norm(init_f) + np.linalg.norm(final_f)
        assert total_norm > 0.0

    def test_normalised_max_one_when_perturbed(self):
        """When forces are nonzero they should be normalised to max-norm = 1."""
        images = self._perturbed_images()
        init_f, final_f = compute_endpoint_gradient_total(images)
        max_norm = max(
            np.linalg.norm(init_f, axis=1).max(),
            np.linalg.norm(final_f, axis=1).max(),
        )
        assert abs(max_norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_endpoint_gradient_barrier
# ---------------------------------------------------------------------------

class TestEndpointGradientBarrier:
    def test_output_shape_and_index(self, two_atom_ase_images):
        init_f, final_f, idx = compute_endpoint_gradient_barrier(two_atom_ase_images)
        assert init_f.shape == (2, 3)
        assert final_f.shape == (2, 3)
        assert 1 <= idx <= len(two_atom_ase_images) - 2

    def test_normalised_max_one_when_perturbed(self):
        """Normalisation should give max-norm = 1 when forces are nonzero."""
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        images = []
        for k in range(5):
            alpha = k / 4
            pos = (1 - alpha) * initial + alpha * final
            if k == 2:
                pos = pos + np.array([[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]])
            images.append(ase.Atoms(symbols=["F", "F"], positions=pos))
        init_f, final_f, _ = compute_endpoint_gradient_barrier(images)
        max_norm = max(
            np.linalg.norm(init_f, axis=1).max(),
            np.linalg.norm(final_f, axis=1).max(),
        )
        assert abs(max_norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Gradient caching (d_initial / d_final optional args)
# ---------------------------------------------------------------------------

def _linear_images_with_perturbation():
    """5-image path where image 2 deviates from linear interpolation."""
    initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
    images = []
    for k in range(5):
        alpha = k / 4
        pos = (1 - alpha) * initial + alpha * final
        if k == 2:
            pos = pos + np.array([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]])
        images.append(ase.Atoms(symbols=["F", "F"], positions=pos.copy()))
    return images, initial, final


class TestEndpointGradientCaching:
    def test_total_cached_equals_uncached(self):
        images, initial, final = _linear_images_with_perturbation()
        fi, ff = compute_endpoint_gradient_total(images)
        d_i = _pairwise_distances(initial)
        d_f = _pairwise_distances(final)
        fi_c, ff_c = compute_endpoint_gradient_total(images, d_initial=d_i, d_final=d_f)
        np.testing.assert_allclose(fi, fi_c, atol=1e-12)
        np.testing.assert_allclose(ff, ff_c, atol=1e-12)

    def test_barrier_cached_equals_uncached(self):
        images, initial, final = _linear_images_with_perturbation()
        fi, ff, idx = compute_endpoint_gradient_barrier(images)
        d_i = _pairwise_distances(initial)
        d_f = _pairwise_distances(final)
        fi_c, ff_c, idx_c = compute_endpoint_gradient_barrier(
            images, d_initial=d_i, d_final=d_f
        )
        np.testing.assert_allclose(fi, fi_c, atol=1e-12)
        np.testing.assert_allclose(ff, ff_c, atol=1e-12)
        assert idx == idx_c


# ---------------------------------------------------------------------------
# compute_endpoint_gradient_barrier — max-index correctness
# ---------------------------------------------------------------------------

class TestEndpointGradientBarrierMaxIdx:
    def test_max_index_identifies_highest_energy_image(self):
        """max_img_idx must point to the image with the largest IDPP energy."""
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        images = []
        for k in range(5):
            alpha = k / 4
            pos = (1 - alpha) * initial + alpha * final
            # Image 3 has a large deviation — should be the barrier
            if k == 3:
                pos = pos + np.array([[1.0, 0.5, 0.0], [-1.0, -0.5, 0.0]])
            images.append(ase.Atoms(symbols=["F", "F"], positions=pos.copy()))
        _, _, idx = compute_endpoint_gradient_barrier(images)
        assert idx == 3


# ---------------------------------------------------------------------------
# apply_endpoint_adjustment
# ---------------------------------------------------------------------------

class TestApplyEndpointAdjustment:
    def test_atoms_move_along_force(self):
        pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        atoms = ase.Atoms(symbols=["F", "F"], positions=pos.copy())
        # Force pushes atoms apart
        force = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        force = force / max(np.linalg.norm(force, axis=1).max(), 1e-10)
        n = 2
        lo = np.zeros((n, n))
        hi = np.ones((n, n)) * 10.0
        rmsd = apply_endpoint_adjustment(atoms, force, lo, hi, step_size=0.5)
        new_pos = atoms.get_positions()
        # atom 1 should have moved in +x direction
        assert new_pos[1, 0] > pos[1, 0]

    def test_returns_float_rmsd(self):
        atoms = ase.Atoms(symbols=["F"], positions=[[0.0, 0.0, 0.0]])
        force = np.array([[1.0, 0.0, 0.0]])
        lo = np.zeros((1, 1))
        hi = np.ones((1, 1)) * 10.0
        result = apply_endpoint_adjustment(atoms, force, lo, hi, step_size=0.1)
        assert isinstance(result, float)

    def test_upper_bound_respected(self):
        """Atoms should not violate a tight upper bound after adjustment."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atoms = ase.Atoms(symbols=["F", "F"], positions=pos.copy())
        force = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        n = 2
        lo = np.zeros((n, n))
        hi = np.array([[0.0, 1.2], [1.2, 0.0]])  # max distance = 1.2
        apply_endpoint_adjustment(atoms, force, lo, hi, step_size=2.0)
        new_d = np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0])
        assert new_d <= 1.21  # allow small numerical tolerance

    def test_lower_bound_respected(self):
        """Atoms that start too close should be pushed apart to satisfy lower bound."""
        pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])  # distance = 0.4
        atoms = ase.Atoms(symbols=["F", "F"], positions=pos.copy())
        force = np.zeros((2, 3))  # no driving force — bounds projection only
        n = 2
        lo = np.array([[0.0, 1.5], [1.5, 0.0]])  # require distance >= 1.5
        hi = np.full((n, n), 10.0)
        apply_endpoint_adjustment(atoms, force, lo, hi, step_size=0.0, max_displacement=5.0)
        new_d = np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0])
        assert new_d >= 1.5 - 1e-6

    def test_max_displacement_cap(self):
        """No atom should move further than max_displacement from its start."""
        pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        atoms = ase.Atoms(symbols=["F", "F"], positions=pos.copy())
        force = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # push apart
        lo = np.zeros((2, 2))
        hi = np.full((2, 2), 100.0)  # loose bounds
        max_d = 0.3
        apply_endpoint_adjustment(atoms, force, lo, hi, step_size=5.0, max_displacement=max_d)
        per_atom = np.linalg.norm(atoms.get_positions() - pos, axis=1)
        assert per_atom.max() <= max_d + 1e-8


class TestBoundViolation:
    def test_reports_worst_pair(self):
        """The largest excursion outside any pair's bounds is returned."""
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.8, 0.0]])
        lo = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        hi = np.array([[0.0, 1.2, 2.0], [1.2, 0.0, 2.0], [2.0, 2.0, 0.0]])
        # pair 0-1 is 0.3 above its upper bound; pair 0-2 is 0.2 below its lower bound
        assert bound_violation(pos, lo, hi) == pytest.approx(0.3)

    def test_zero_when_within_bounds(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        lo = np.array([[0.0, 1.0], [1.0, 0.0]])
        hi = np.array([[0.0, 1.2], [1.2, 0.0]])
        assert bound_violation(pos, lo, hi) == 0.0


class TestProjectionTolerance:
    def test_large_step_ends_within_bounds(self):
        """A step that would break a tightly bounded bond is projected back within tolerance.

        The cap used to be applied after projection and could push pairs back out of bounds.
        """
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0]])
        atoms = ase.Atoms(symbols=["Pd", "C", "O"], positions=pos.copy())
        force = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        lo = np.array([[0.0, 1.95, 1.9], [1.95, 0.0, 1.0], [1.9, 1.0, 0.0]])
        hi = np.array([[0.0, 2.07, 2.1], [2.07, 0.0, 5.0], [2.1, 5.0, 0.0]])
        apply_endpoint_adjustment(atoms, force, lo, hi, step_size=1.0, max_displacement=1.0)
        assert bound_violation(atoms.get_positions(), lo, hi) <= BOUNDS_TOLERANCE
