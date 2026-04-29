"""Tests for sparta.iterative_alignment.optimization."""

import numpy as np
import pytest

import ase

from sparta import Reaction
from sparta.initial_generation.bounds_matrix import generate_initial_conformers
from sparta.iterative_alignment.optimization import (
    _build_neb_images,
    _build_neb_images_warm,
    _extract_symmetric_bounds,
    _relax_bounds_to_structure,
    _run_alignment_single,
    find_optimal_stretch_factor,
    run_iterative_alignment,
    score_conformer_pair,
)

REACTANT_SMILES = "[F:1][F:2].[Cl:3][Cl:4]"
PRODUCT_SMILES = "[F:1][Cl:3].[F:2][Cl:4]"

FAST_KWARGS = dict(
    neb_fmax=0.1,
    neb_steps=5,
    propagation_threshold=0.01,
    propagation_step_size=0.5,
    max_iterations=2,
)


@pytest.fixture(scope="module")
def rxn_with_conformers():
    r = Reaction(
        reactant_smiles=REACTANT_SMILES,
        product_smiles=PRODUCT_SMILES,
        n_conformers=1,
        n_seeds=1,
        stretch_factors=1.5,
        top_k=1,
        n_images=3,
    )
    return generate_initial_conformers(r)


# ---------------------------------------------------------------------------
# score_conformer_pair
# ---------------------------------------------------------------------------

class TestScoreConformerPair:
    def test_returns_finite_float(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        score = score_conformer_pair(
            rxn_with_conformers,
            c["reactant_coords"],
            c["product_coords"],
            steps=5,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_score_is_positive(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        score = score_conformer_pair(
            rxn_with_conformers,
            c["reactant_coords"],
            c["product_coords"],
            steps=5,
        )
        assert score > 0.0

    def test_identical_endpoints_has_zero_score(self, rxn_with_conformers):
        """Identical reactant and product → no path cost."""
        c = rxn_with_conformers.top_k_conformers[0]
        score = score_conformer_pair(
            rxn_with_conformers,
            c["reactant_coords"],
            c["reactant_coords"],  # same as reactant
            steps=5,
        )
        assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _run_alignment_single
# ---------------------------------------------------------------------------

class TestRunAlignmentSingle:
    def test_returns_dict_with_required_keys(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(rxn_with_conformers, c, **FAST_KWARGS)
        required = {"reactant_coords", "product_coords", "images", "energies", "score"}
        assert required.issubset(result.keys())

    def test_score_is_finite_float(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(rxn_with_conformers, c, **FAST_KWARGS)
        assert isinstance(result["score"], float)
        assert np.isfinite(result["score"])

    def test_coords_shape(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(rxn_with_conformers, c, **FAST_KWARGS)
        n = rxn_with_conformers.reactant_mol.GetNumAtoms()
        assert result["reactant_coords"].shape == (n, 3)
        assert result["product_coords"].shape == (n, 3)

    def test_images_length(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(rxn_with_conformers, c, **FAST_KWARGS)
        assert len(result["images"]) == rxn_with_conformers.n_images

    def test_energies_length_matches_images(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(rxn_with_conformers, c, **FAST_KWARGS)
        assert len(result["energies"]) == len(result["images"])

    def test_total_objective(self, rxn_with_conformers):
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(rxn_with_conformers, c, objective="total", **FAST_KWARGS)
        assert np.isfinite(result["score"])

    def test_custom_backtrack_params_accepted(self, rxn_with_conformers):
        """backtrack_fmax and backtrack_steps should be forwarded without error."""
        c = rxn_with_conformers.top_k_conformers[0]
        result = _run_alignment_single(
            rxn_with_conformers, c,
            backtrack_fmax=0.05, backtrack_steps=3,
            **FAST_KWARGS,
        )
        assert np.isfinite(result["score"])


# ---------------------------------------------------------------------------
# run_iterative_alignment (high-level)
# ---------------------------------------------------------------------------

class TestRunIterativeAlignment:
    def test_populates_aligned_results(self, rxn_with_conformers):
        rxn = run_iterative_alignment(rxn_with_conformers, **FAST_KWARGS)
        assert len(rxn.aligned_results) > 0

    def test_returns_reaction(self, rxn_with_conformers):
        rxn = run_iterative_alignment(rxn_with_conformers, **FAST_KWARGS)
        assert isinstance(rxn, Reaction)

    def test_returns_same_object(self, rxn_with_conformers):
        rxn = run_iterative_alignment(rxn_with_conformers, **FAST_KWARGS)
        assert rxn is rxn_with_conformers

    def test_aligned_results_sorted_by_score(self, rxn_with_conformers):
        run_iterative_alignment(rxn_with_conformers, **FAST_KWARGS)
        scores = [r["score"] for r in rxn_with_conformers.aligned_results]
        assert scores == sorted(scores)

    def test_result_has_provenance_keys(self, rxn_with_conformers):
        run_iterative_alignment(rxn_with_conformers, **FAST_KWARGS)
        for result in rxn_with_conformers.aligned_results:
            assert "conformer_idx" in result
            assert "stretch_factor" in result
            assert "seed" in result

    def test_custom_backtrack_params_accepted(self, rxn_with_conformers):
        """backtrack_fmax and backtrack_steps should be forwarded without error."""
        rxn = run_iterative_alignment(
            rxn_with_conformers,
            backtrack_fmax=0.05, backtrack_steps=3,
            **FAST_KWARGS,
        )
        assert len(rxn.aligned_results) > 0

    def test_raises_on_empty_top_k(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )
        # top_k_conformers is empty — run_iterative_alignment should raise
        with pytest.raises(ValueError, match="generate_initial_conformers"):
            run_iterative_alignment(r)


# ---------------------------------------------------------------------------
# _extract_symmetric_bounds
# ---------------------------------------------------------------------------

class TestExtractSymmetricBounds:
    def test_lower_triangle_populates_lower(self):
        bm = np.zeros((3, 3))
        bm[1, 0] = 1.5  # lower[0,1]
        lower, upper = _extract_symmetric_bounds(bm)
        assert lower[0, 1] == pytest.approx(1.5)
        assert lower[1, 0] == pytest.approx(1.5)  # symmetrised

    def test_upper_triangle_populates_upper(self):
        bm = np.zeros((3, 3))
        bm[0, 1] = 3.0  # upper[0,1]
        lower, upper = _extract_symmetric_bounds(bm)
        assert upper[0, 1] == pytest.approx(3.0)
        assert upper[1, 0] == pytest.approx(3.0)  # symmetrised

    def test_diagonal_is_zero(self):
        bm = np.eye(3) * 5.0
        lower, upper = _extract_symmetric_bounds(bm)
        np.testing.assert_array_equal(np.diag(lower), np.zeros(3))
        np.testing.assert_array_equal(np.diag(upper), np.zeros(3))

    def test_output_shape_matches_input(self):
        bm = np.zeros((4, 4))
        lower, upper = _extract_symmetric_bounds(bm)
        assert lower.shape == (4, 4)
        assert upper.shape == (4, 4)


# ---------------------------------------------------------------------------
# _relax_bounds_to_structure
# ---------------------------------------------------------------------------

class TestRelaxBoundsToStructure:
    def _make_bounds(self, n=2, lower_val=2.0, upper_val=10.0):
        lower = np.zeros((n, n))
        upper = np.full((n, n), upper_val)
        lower[0, 1] = lower[1, 0] = lower_val
        return lower, upper

    def test_lower_bound_relaxed_when_violated(self):
        lower, upper = self._make_bounds(lower_val=2.0)
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # dist=1.0 < 2.0
        lower_out, _ = _relax_bounds_to_structure(lower, upper, positions)
        assert lower_out[0, 1] == pytest.approx(1.0)
        assert lower_out[1, 0] == pytest.approx(1.0)

    def test_lower_bound_unchanged_when_satisfied(self):
        lower, upper = self._make_bounds(lower_val=0.5)
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # dist=1.0 > 0.5
        lower_out, _ = _relax_bounds_to_structure(lower, upper, positions)
        assert lower_out[0, 1] == pytest.approx(0.5)

    def test_upper_bound_not_modified(self):
        lower, upper = self._make_bounds(lower_val=2.0, upper_val=5.0)
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        _, upper_out = _relax_bounds_to_structure(lower, upper, positions)
        np.testing.assert_array_equal(upper_out, upper)

    def test_input_not_mutated(self):
        lower, upper = self._make_bounds(lower_val=2.0)
        original_lower = lower.copy()
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        _relax_bounds_to_structure(lower, upper, positions)
        np.testing.assert_array_equal(lower, original_lower)


# ---------------------------------------------------------------------------
# _build_neb_images
# ---------------------------------------------------------------------------

class TestBuildNebImages:
    def _make_endpoints(self):
        pos1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        pos2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.8]])
        symbols = ["F", "F"]
        return pos1, pos2, symbols

    def test_image_count(self):
        pos1, pos2, symbols = self._make_endpoints()
        images = _build_neb_images(pos1, pos2, symbols, n_images=7)
        assert len(images) == 7

    def test_endpoint_positions_preserved(self):
        pos1, pos2, symbols = self._make_endpoints()
        images = _build_neb_images(pos1, pos2, symbols, n_images=5)
        np.testing.assert_allclose(images[0].get_positions(), pos1, atol=1e-10)
        np.testing.assert_allclose(images[-1].get_positions(), pos2, atol=1e-10)

    def test_midpoint_linearly_interpolated(self):
        pos1, pos2, symbols = self._make_endpoints()
        images = _build_neb_images(pos1, pos2, symbols, n_images=3)
        expected = (pos1 + pos2) / 2.0
        np.testing.assert_allclose(images[1].get_positions(), expected, atol=1e-10)

    def test_symbols_assigned(self):
        pos1, pos2, symbols = self._make_endpoints()
        images = _build_neb_images(pos1, pos2, symbols, n_images=3)
        for img in images:
            assert img.get_chemical_symbols() == symbols

    def test_returns_ase_atoms_objects(self):
        pos1, pos2, symbols = self._make_endpoints()
        images = _build_neb_images(pos1, pos2, symbols, n_images=3)
        for img in images:
            assert isinstance(img, ase.Atoms)


# ---------------------------------------------------------------------------
# _build_neb_images_warm
# ---------------------------------------------------------------------------

class TestBuildNebImagesWarm:
    def _make_current_images(self, n_images=5):
        symbols = ["F", "F"]
        images = [
            ase.Atoms(symbols=symbols, positions=np.array([[float(k), 0.0, 0.0],
                                                            [float(k) + 1.0, 0.0, 0.0]]))
            for k in range(n_images)
        ]
        return images, symbols

    def test_image_count_preserved(self):
        images, symbols = self._make_current_images(7)
        new_images = _build_neb_images_warm(np.zeros((2, 3)), np.zeros((2, 3)), images, symbols)
        assert len(new_images) == 7

    def test_endpoint_positions_updated(self):
        images, symbols = self._make_current_images()
        new_reac = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
        new_prod = np.array([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]])
        new_images = _build_neb_images_warm(new_reac, new_prod, images, symbols)
        np.testing.assert_allclose(new_images[0].get_positions(), new_reac)
        np.testing.assert_allclose(new_images[-1].get_positions(), new_prod)

    def test_interior_positions_preserved(self):
        images, symbols = self._make_current_images()
        new_reac = images[0].get_positions().copy()
        new_prod = images[-1].get_positions().copy()
        new_images = _build_neb_images_warm(new_reac, new_prod, images, symbols)
        for k in range(1, len(images) - 1):
            np.testing.assert_allclose(
                new_images[k].get_positions(), images[k].get_positions()
            )

    def test_interior_differs_from_linear_interpolation(self):
        """Interior positions must come from current_images, not be re-interpolated."""
        symbols = ["F", "F"]
        reac = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        prod = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        off_path = np.array([[9.0, 9.0, 0.0], [10.0, 9.0, 0.0]])
        current = [
            ase.Atoms(symbols=symbols, positions=reac),
            ase.Atoms(symbols=symbols, positions=off_path),
            ase.Atoms(symbols=symbols, positions=prod),
        ]
        warm = _build_neb_images_warm(reac, prod, current, symbols)
        linear = _build_neb_images(reac, prod, symbols, 3)
        assert not np.allclose(warm[1].get_positions(), linear[1].get_positions())
        np.testing.assert_allclose(warm[1].get_positions(), off_path)


# ---------------------------------------------------------------------------
# find_optimal_stretch_factor
# ---------------------------------------------------------------------------

FAST_STRETCH_KWARGS = dict(
    neb_fmax=0.1,
    neb_steps=5,
    propagation_threshold=0.05,
    propagation_step_size=0.5,
    max_iterations=1,
)


@pytest.fixture(scope="module")
def rxn_for_stretch():
    return Reaction(
        reactant_smiles=REACTANT_SMILES,
        product_smiles=PRODUCT_SMILES,
        n_conformers=1,
        n_seeds=1,
        stretch_factors=1.5,
        top_k=1,
        n_images=3,
    )


class TestFindOptimalStretchFactor:
    def test_returns_reaction(self, rxn_for_stretch):
        result = find_optimal_stretch_factor(
            rxn_for_stretch,
            start=1.5, step=0.1, max_stretch=1.5,
            **FAST_STRETCH_KWARGS,
        )
        assert isinstance(result, Reaction)

    def test_has_aligned_results(self, rxn_for_stretch):
        result = find_optimal_stretch_factor(
            rxn_for_stretch,
            start=1.5, step=0.1, max_stretch=1.5,
            **FAST_STRETCH_KWARGS,
        )
        assert len(result.aligned_results) > 0

    def test_best_score_is_finite(self, rxn_for_stretch):
        result = find_optimal_stretch_factor(
            rxn_for_stretch,
            start=1.5, step=0.1, max_stretch=1.5,
            **FAST_STRETCH_KWARGS,
        )
        assert np.isfinite(result.aligned_results[0]["score"])

    def test_raises_when_start_exceeds_max_stretch(self, rxn_for_stretch):
        with pytest.raises(ValueError):
            find_optimal_stretch_factor(
                rxn_for_stretch,
                start=2.0, step=0.1, max_stretch=1.9,
                **FAST_STRETCH_KWARGS,
            )

    def test_returns_new_reaction_not_input(self, rxn_for_stretch):
        result = find_optimal_stretch_factor(
            rxn_for_stretch,
            start=1.5, step=0.1, max_stretch=1.5,
            **FAST_STRETCH_KWARGS,
        )
        assert result is not rxn_for_stretch
