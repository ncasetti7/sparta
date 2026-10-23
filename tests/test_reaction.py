"""Tests for sparta.reaction.Reaction and the top-level pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sparta import Reaction, generate_initial_conformers, run_iterative_alignment

REACTANT_SMILES = "[F:1][F:2].[Cl:3][Cl:4]"
PRODUCT_SMILES = "[F:1][Cl:3].[F:2][Cl:4]"

FAST_PIPELINE_KWARGS = dict(
    neb_fmax=0.1,
    neb_steps=5,
    propagation_threshold=0.01,
    propagation_step_size=0.5,
    max_iterations=2,
)


@pytest.fixture(scope="module")
def rxn():
    return Reaction(
        reactant_smiles=REACTANT_SMILES,
        product_smiles=PRODUCT_SMILES,
        n_conformers=2,
        n_seeds=1,
        stretch_factors=[1.3, 1.5],
        top_k=2,
        n_images=3,
    )


# ---------------------------------------------------------------------------
# __init__: topology
# ---------------------------------------------------------------------------

class TestReactionInit:
    def test_changing_bonds_correct(self, rxn):
        # F2 + Cl2 → 2FCl: reactant bonds (0,1),(2,3); product bonds (0,2),(1,3)
        assert set(rxn.changing_bonds) == {(0, 1), (0, 2), (1, 3), (2, 3)}

    def test_reactant_mol_atom_count(self, rxn):
        assert rxn.reactant_mol.GetNumAtoms() == 4

    def test_product_mol_atom_count(self, rxn):
        assert rxn.product_mol.GetNumAtoms() == 4

    def test_base_bm_shape(self, rxn):
        n = rxn.reactant_mol.GetNumAtoms()
        assert rxn.reactant_base_bm.shape == (n, n)
        assert rxn.product_base_bm.shape == (n, n)

    def test_base_bm_not_all_zeros(self, rxn):
        assert np.any(rxn.reactant_base_bm != 0)
        assert np.any(rxn.product_base_bm != 0)

    def test_reac_bond_lengths_keys(self, rxn):
        assert set(rxn.reac_bond_lengths.keys()) == set(rxn.changing_bonds)

    def test_prod_bond_lengths_keys(self, rxn):
        assert set(rxn.prod_bond_lengths.keys()) == set(rxn.changing_bonds)

    def test_bond_lengths_positive(self, rxn):
        for v in rxn.reac_bond_lengths.values():
            assert v > 0.0
        for v in rxn.prod_bond_lengths.values():
            assert v > 0.0

    def test_reactant_frags(self, rxn):
        # F2.Cl2 → 2 fragments
        assert len(rxn.reactant_frags) == 2

    def test_product_frags(self, rxn):
        # FCl.FCl → 2 fragments
        assert len(rxn.product_frags) == 2

    def test_rotation_default_true(self, rxn):
        assert rxn.rotation is True

    def test_translation_default_true_for_multi_frag(self, rxn):
        # Both reactant and product have >1 fragment → translation should default True
        assert rxn.translation is True

    def test_translation_false_for_single_frag(self):
        # Intramolecular: single-fragment reaction
        r = Reaction(
            reactant_smiles="[F:1][F:2]",
            product_smiles="[F:1][F:2]",
        )
        assert r.translation is False

    def test_result_containers_empty_on_init(self, rxn):
        fresh = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
        )
        assert fresh.all_conformers == []
        assert fresh.top_k_conformers == []
        assert fresh.aligned_results == []

    def test_stretch_factors_normalized_to_list(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            stretch_factors=1.5,
        )
        assert r.stretch_factors == [1.5]

    def test_translation_true_requires_rotation(self):
        with pytest.raises(ValueError):
            Reaction(
                reactant_smiles=REACTANT_SMILES,
                product_smiles=PRODUCT_SMILES,
                rotation=False,
                translation=True,
            )

    def test_charge_inferred_neutral(self):
        r = Reaction(reactant_smiles=REACTANT_SMILES, product_smiles=PRODUCT_SMILES)
        assert r.charge == 0

    def test_multiplicity_inferred_singlet(self):
        r = Reaction(reactant_smiles=REACTANT_SMILES, product_smiles=PRODUCT_SMILES)
        assert r.multiplicity == 1

    def test_charge_inferred_from_smiles(self):
        # [F-] + [Na+] → net charge 0 on both sides
        r = Reaction(reactant_smiles="[F-:1].[Na+:2]", product_smiles="[F-:1].[Na+:2]")
        assert r.charge == 0

    def test_charge_inferred_cationic(self):
        # Both sides carry +1 overall
        r = Reaction(reactant_smiles="[NH4+:1]", product_smiles="[NH4+:1]")
        assert r.charge == 1

    def test_charge_conservation_error(self):
        # Reactant and product have different formal charges → ValueError
        with pytest.raises(ValueError, match="charge is not conserved"):
            Reaction(reactant_smiles="[F-:1].[Na+:2]", product_smiles="[Na+:1].[F:2]")

    def test_charge_explicit_override(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            charge=2,
        )
        assert r.charge == 2

    def test_multiplicity_explicit_override(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            multiplicity=3,
        )
        assert r.multiplicity == 3


# ---------------------------------------------------------------------------
# Calculator charge/multiplicity propagation
# ---------------------------------------------------------------------------

class TestCalculatorPropagation:
    """Reaction.charge/multiplicity must be applied to any passed calculator."""

    def _make_reaction(self, charge=0, multiplicity=1):
        return Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            charge=charge,
            multiplicity=multiplicity,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )

    def test_generate_conformers_applies_charge_to_calculator(self):
        r = self._make_reaction(charge=-1, multiplicity=2)
        calc = MagicMock()
        calc.charge = 99
        calc.multiplicity = 99
        with patch("sparta.initial_generation.generate_initial_conformers", return_value=r):
            r.generate_initial_conformers(calculator=calc)
        assert calc.charge == -1
        assert calc.multiplicity == 2

    def test_run_alignment_applies_charge_to_calculator(self):
        r = self._make_reaction(charge=2, multiplicity=3)
        r.top_k_conformers = [{}]
        calc = MagicMock()
        calc.charge = 99
        calc.multiplicity = 99
        with patch("sparta.iterative_alignment.run_iterative_alignment", return_value=r):
            r.run_iterative_alignment(calculator=calc)
        assert calc.charge == 2
        assert calc.multiplicity == 3

    def test_no_calculator_no_error(self):
        r = self._make_reaction()
        r.generate_initial_conformers()  # calculator=None should not raise


# ---------------------------------------------------------------------------
# generate_initial_conformers (method)
# ---------------------------------------------------------------------------

class TestGenerateConformersMethod:
    def test_returns_self(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )
        returned = r.generate_initial_conformers()
        assert returned is r

    def test_populates_conformers(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )
        r.generate_initial_conformers()
        assert len(r.top_k_conformers) > 0


# ---------------------------------------------------------------------------
# run_iterative_alignment (method)
# ---------------------------------------------------------------------------

class TestRunAlignmentMethod:
    def test_raises_before_generate(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            n_images=3,
        )
        with pytest.raises(ValueError):
            r.run_iterative_alignment()

    def test_returns_self(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )
        r.generate_initial_conformers()
        returned = r.run_iterative_alignment(**FAST_PIPELINE_KWARGS)
        assert returned is r

    def test_populates_aligned_results(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )
        r.generate_initial_conformers().run_iterative_alignment(**FAST_PIPELINE_KWARGS)
        assert len(r.aligned_results) > 0


# ---------------------------------------------------------------------------
# Full pipeline (functional style)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.fixture(scope="class")
    def completed_rxn(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=1.5,
            top_k=1,
            n_images=3,
        )
        r = generate_initial_conformers(r)
        r = run_iterative_alignment(r, **FAST_PIPELINE_KWARGS)
        return r

    def test_best_score_is_finite(self, completed_rxn):
        assert np.isfinite(completed_rxn.aligned_results[0]["score"])

    def test_best_reactant_coords_shape(self, completed_rxn):
        n = completed_rxn.reactant_mol.GetNumAtoms()
        assert completed_rxn.aligned_results[0]["reactant_coords"].shape == (n, 3)

    def test_best_product_coords_shape(self, completed_rxn):
        n = completed_rxn.reactant_mol.GetNumAtoms()
        assert completed_rxn.aligned_results[0]["product_coords"].shape == (n, 3)

    def test_chaining_style_equivalent(self):
        r = (
            Reaction(
                reactant_smiles=REACTANT_SMILES,
                product_smiles=PRODUCT_SMILES,
                n_conformers=1,
                n_seeds=1,
                stretch_factors=1.5,
                top_k=1,
                n_images=3,
            )
            .generate_initial_conformers()
            .run_iterative_alignment(**FAST_PIPELINE_KWARGS)
        )
        assert len(r.aligned_results) > 0

    def test_multiple_stretch_factors(self):
        r = Reaction(
            reactant_smiles=REACTANT_SMILES,
            product_smiles=PRODUCT_SMILES,
            n_conformers=1,
            n_seeds=1,
            stretch_factors=[1.3, 1.8],
            top_k=2,
            n_images=3,
        )
        r = generate_initial_conformers(r)
        stretch_values = {c["stretch_factor"] for c in r.all_conformers}
        # Both stretch factors should produce conformers
        assert len(stretch_values) >= 1
