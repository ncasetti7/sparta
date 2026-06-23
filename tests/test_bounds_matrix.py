"""Tests for sparta.initial_generation.bounds_matrix."""

import numpy as np
import pytest
from rdkit import Chem

from sparta import Reaction
from sparta.initial_generation.bounds_matrix import (
    _generate_reaction_conformers_single,
    generate_initial_conformers,
)

REACTANT_SMILES = "[F:1][F:2].[Cl:3][Cl:4]"
PRODUCT_SMILES = "[F:1][Cl:3].[F:2][Cl:4]"


@pytest.fixture(scope="module")
def rxn():
    return Reaction(
        reactant_smiles=REACTANT_SMILES,
        product_smiles=PRODUCT_SMILES,
        n_conformers=2,
        n_seeds=1,
        stretch_factors=1.5,
        top_k=2,
        n_images=3,
    )


# ---------------------------------------------------------------------------
# _generate_reaction_conformers_single
# ---------------------------------------------------------------------------

class TestGenerateSingle:
    def test_returns_four_values(self, rxn):
        result = _generate_reaction_conformers_single(rxn, stretch_factor=1.5, seed=0)
        assert len(result) == 4

    def test_mols_are_rdkit_mol(self, rxn):
        reac_mol, prod_mol, _, _ = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        assert isinstance(reac_mol, Chem.Mol)
        assert isinstance(prod_mol, Chem.Mol)

    def test_reactant_has_conformers(self, rxn):
        reac_mol, _, _, _ = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        assert reac_mol.GetNumConformers() >= 1

    def test_product_has_conformers(self, rxn):
        _, prod_mol, _, _ = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        assert prod_mol.GetNumConformers() >= 1

    def test_conformer_counts_match(self, rxn):
        reac_mol, prod_mol, _, _ = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        assert reac_mol.GetNumConformers() == prod_mol.GetNumConformers()

    def test_bounds_matrices_correct_shape(self, rxn):
        _, _, reac_bm, prod_bm = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        n = rxn.reactant_mol.GetNumAtoms()
        assert reac_bm.shape == (n, n)
        assert prod_bm.shape == (n, n)

    def test_bounds_matrix_not_all_zeros(self, rxn):
        _, _, reac_bm, _ = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        assert np.any(reac_bm != 0)

    def test_uses_cached_base_bm(self, rxn):
        """Two calls should use the same base BM (reactant_base_bm is unchanged)."""
        _, _, bm1, _ = _generate_reaction_conformers_single(rxn, stretch_factor=1.5, seed=0)
        _, _, bm2, _ = _generate_reaction_conformers_single(rxn, stretch_factor=1.5, seed=0)
        # The returned BMs should be identical across calls with same parameters
        np.testing.assert_array_equal(bm1, bm2)

    def test_different_stretch_factors_give_different_bm(self, rxn):
        _, _, bm_1_5, _ = _generate_reaction_conformers_single(rxn, stretch_factor=1.5, seed=0)
        _, _, bm_2_0, _ = _generate_reaction_conformers_single(rxn, stretch_factor=2.0, seed=0)
        # Bounds matrices should differ because target lengths differ
        assert not np.allclose(bm_1_5, bm_2_0)

    def test_coords_are_finite(self, rxn):
        reac_mol, prod_mol, _, _ = _generate_reaction_conformers_single(
            rxn, stretch_factor=1.5, seed=0
        )
        for conf in reac_mol.GetConformers():
            pos = conf.GetPositions()
            assert np.all(np.isfinite(pos))
        for conf in prod_mol.GetConformers():
            pos = conf.GetPositions()
            assert np.all(np.isfinite(pos))


# ---------------------------------------------------------------------------
# generate_initial_conformers
# ---------------------------------------------------------------------------

class TestGenerateInitialConformers:
    @pytest.fixture(scope="class")
    def rxn_with_conformers(self):
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

    def test_returns_reaction(self, rxn_with_conformers):
        assert isinstance(rxn_with_conformers, Reaction)

    def test_all_conformers_populated(self, rxn_with_conformers):
        assert len(rxn_with_conformers.all_conformers) > 0

    def test_top_k_conformers_populated(self, rxn_with_conformers):
        assert len(rxn_with_conformers.top_k_conformers) > 0

    def test_top_k_leq_total(self, rxn_with_conformers):
        assert len(rxn_with_conformers.top_k_conformers) <= len(
            rxn_with_conformers.all_conformers
        )

    def test_top_k_leq_k(self, rxn_with_conformers):
        assert len(rxn_with_conformers.top_k_conformers) <= rxn_with_conformers.top_k

    def test_conformer_dicts_have_required_keys(self, rxn_with_conformers):
        required = {
            "reactant_coords", "product_coords",
            "reactant_bounds_matrix", "product_bounds_matrix",
            "stretch_factor", "seed", "score",
        }
        for c in rxn_with_conformers.all_conformers:
            assert required.issubset(c.keys())

    def test_conformer_scores_are_finite(self, rxn_with_conformers):
        for c in rxn_with_conformers.all_conformers:
            assert np.isfinite(c["score"])

    def test_conformers_sorted_by_score(self, rxn_with_conformers):
        scores = [c["score"] for c in rxn_with_conformers.all_conformers]
        assert scores == sorted(scores)

    def test_bounds_matrices_stored_in_conformer(self, rxn_with_conformers):
        c = rxn_with_conformers.all_conformers[0]
        n = rxn_with_conformers.reactant_mol.GetNumAtoms()
        assert c["reactant_bounds_matrix"].shape == (n, n)
        assert c["product_bounds_matrix"].shape == (n, n)

    def test_parallel_generation_populates_conformers(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=2,
            stretch_factors=1.5,
            top_k=2,
            n_images=3,
        )

        generate_initial_conformers(r, n_workers=2)

        assert len(r.all_conformers) > 0
        assert len(r.top_k_conformers) > 0
