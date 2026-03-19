"""Tests for sparta.iterative_alignment.optimization."""

import numpy as np
import pytest

from sparta import Reaction
from sparta.initial_generation.bounds_matrix import generate_initial_conformers
from sparta.iterative_alignment.optimization import (
    _run_alignment_single,
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
