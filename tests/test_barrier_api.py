"""Tests for the barrier-reporting API added in 0.2.0.

These cover the parts a reader of the paper actually touches: that a barrier
comes out in kcal/mol, that it is measured against a separated-fragment
reference, that the stretch-factor scan honors `patience` and ranks on the
right quantity, and that nothing electronic appears when no calculator is
given. A stub calculator stands in for xtb so the suite stays fast and has no
external dependency.
"""
import numpy as np
import pytest

from sparta import (
    Reaction,
    generate_initial_conformers,
    run_iterative_alignment,
    run_sparta,
    sparta_barrier,
)
from sparta.iterative_alignment.optimization import HARTREE_TO_KCAL

R_SMI = "[C:1]([H:2])([H:3])([H:4])[H:5].[F:6][F:7]"
P_SMI = "[C:1]([H:2])([H:3])([H:4])[F:6].[H:5][F:7]"


class StubCalculator:
    """Energy is a deterministic function of geometry, so barriers are stable.

    Returns a value that grows with how compressed the structure is, which
    makes an interior image of a reaction path reliably higher in energy than
    a relaxed fragment, without invoking a quantum-chemistry program.
    """

    def __init__(self):
        self.charge = 0
        self.multiplicity = 1
        self.n_energy_calls = 0

    def energy(self, atoms):
        self.n_energy_calls += 1
        pos = atoms.get_positions()
        if len(pos) < 2:
            return 0.0
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        iu = np.triu_indices(len(pos), k=1)
        return float(np.sum(1.0 / np.clip(d[iu], 0.3, None)))

    def optimize(self, atoms):
        return atoms, self.energy(atoms)


@pytest.fixture
def aligned():
    rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                   n_conformers=2, n_seeds=1, stretch_factors=[1.5], top_k=1)
    rxn = generate_initial_conformers(rxn)
    return run_iterative_alignment(rxn)


class TestNoCalculator:
    def test_no_electronic_keys_without_calculator(self, aligned):
        best = aligned.aligned_results[0]
        assert "barrier_kcal" not in best
        assert "electronic_energies" not in best

    def test_idpp_score_still_present(self, aligned):
        assert isinstance(aligned.aligned_results[0]["score"], float)


class TestWithCalculator:
    def test_barrier_is_reported_in_kcal(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, stretch_factors=[1.5], top_k=1)
        rxn = generate_initial_conformers(rxn)
        rxn = run_iterative_alignment(rxn, calculator=StubCalculator())

        best = rxn.aligned_results[0]
        assert "barrier_kcal" in best
        assert np.isfinite(best["barrier_kcal"])
        assert rxn.barrier_kcal == best["barrier_kcal"]

    def test_barrier_matches_peak_minus_reference(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, stretch_factors=[1.5], top_k=1)
        rxn = generate_initial_conformers(rxn)
        rxn = run_iterative_alignment(rxn, calculator=StubCalculator())

        best = rxn.aligned_results[0]
        peak = max(best["electronic_energies"])
        expected = (peak - rxn.reference_energy) * HARTREE_TO_KCAL
        assert best["barrier_kcal"] == pytest.approx(expected)

    def test_all_interior_images_evaluated(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, stretch_factors=[1.5],
                       top_k=1, n_images=11)
        rxn = generate_initial_conformers(rxn)
        rxn = run_iterative_alignment(rxn, calculator=StubCalculator())
        # 11 images means 9 interior ones
        assert len(rxn.aligned_results[0]["electronic_energies"]) == 9

    def test_peak_only_evaluates_one_image(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, stretch_factors=[1.5], top_k=1)
        rxn = generate_initial_conformers(rxn)
        rxn = run_iterative_alignment(
            rxn, calculator=StubCalculator(), sp_images="peak"
        )
        assert len(rxn.aligned_results[0]["electronic_energies"]) == 1

    def test_reference_uses_separated_fragments(self):
        """The two reactant fragments are relaxed apart, so the reference does
        not carry an inter-fragment interaction term."""
        calc = StubCalculator()
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, stretch_factors=[1.5], top_k=1)
        rxn = generate_initial_conformers(rxn)
        rxn = run_iterative_alignment(rxn, calculator=calc)

        import ase
        symbols = [rxn.reactant_mol.GetAtomWithIdx(i).GetSymbol()
                   for i in range(rxn.reactant_mol.GetNumAtoms())]
        together = calc.energy(
            ase.Atoms(symbols=symbols, positions=rxn.reac_ref_coords)
        )
        # a co-optimized reference would include the cross term and so differ
        assert rxn.reference_energy != pytest.approx(together)


class TestSpartaBarrier:
    def test_raises_without_alignment(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI)
        with pytest.raises(ValueError, match="no aligned results"):
            sparta_barrier(rxn, StubCalculator())

    def test_reuses_stored_energies(self):
        calc = StubCalculator()
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, stretch_factors=[1.5], top_k=1)
        rxn = generate_initial_conformers(rxn)
        rxn = run_iterative_alignment(rxn, calculator=calc)

        before = calc.n_energy_calls
        value = sparta_barrier(rxn, calc)
        assert calc.n_energy_calls == before      # nothing recomputed
        assert value == rxn.barrier_kcal

    def test_computes_when_not_already_present(self, aligned):
        calc = StubCalculator()
        value = sparta_barrier(aligned, calc)
        assert np.isfinite(value)
        assert calc.n_energy_calls > 0


class TestRunSparta:
    def test_forces_one_endpoint_pair(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1, top_k=5)
        rxn = run_sparta(rxn, StubCalculator(), stretch_factors=[1.5])
        assert rxn.top_k == 1

    def test_fixed_factors_skip_the_scan(self):
        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI,
                       n_conformers=2, n_seeds=1)
        rxn = run_sparta(rxn, StubCalculator(), stretch_factors=[1.4, 1.6])
        assert np.isfinite(rxn.barrier_kcal)
        assert rxn.aligned_results[0]["stretch_factor"] in (1.4, 1.6)


class TestScanPatience:
    def test_patience_one_stops_at_first_non_improvement(self, monkeypatch):
        """Patience is what distinguishes a greedy stop from a real scan."""
        from sparta.iterative_alignment import optimization as opt

        seen = []
        scores = iter([5.0, 6.0, 7.0, 1.0, 8.0, 9.0])

        def fake_align(reaction, **kwargs):
            s = next(scores)
            seen.append(reaction.stretch_factors[0])
            reaction.aligned_results = [
                {"score": s, "barrier_kcal": s, "stretch_factor":
                 reaction.stretch_factors[0], "images": [], "energies": []}
            ]
            return reaction

        monkeypatch.setattr(opt, "run_iterative_alignment", fake_align)
        monkeypatch.setattr(
            opt, "generate_initial_conformers",
            lambda r, **k: setattr(r, "top_k_conformers", [object()]) or r,
            raising=False,
        )

        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI)
        opt.find_optimal_stretch_factor(rxn, start=1.0, step=0.1,
                                        max_stretch=1.6, patience=1)
        # 5.0 then 6.0 is one miss, and patience 1 stops there
        assert len(seen) == 2

    def test_patience_three_looks_past_a_dip(self, monkeypatch):
        from sparta.iterative_alignment import optimization as opt

        seen = []
        scores = iter([5.0, 6.0, 7.0, 1.0, 8.0, 9.0])

        def fake_align(reaction, **kwargs):
            s = next(scores)
            seen.append(reaction.stretch_factors[0])
            reaction.aligned_results = [
                {"score": s, "barrier_kcal": s, "stretch_factor":
                 reaction.stretch_factors[0], "images": [], "energies": []}
            ]
            return reaction

        monkeypatch.setattr(opt, "run_iterative_alignment", fake_align)
        monkeypatch.setattr(
            opt, "generate_initial_conformers",
            lambda r, **k: setattr(r, "top_k_conformers", [object()]) or r,
            raising=False,
        )

        rxn = Reaction(reactant_smiles=R_SMI, product_smiles=P_SMI)
        best = opt.find_optimal_stretch_factor(rxn, start=1.0, step=0.1,
                                               max_stretch=1.6, patience=3)
        # patience 3 survives the 6.0/7.0 misses and reaches the 1.0 at the
        # fourth factor, which patience 1 never sees
        assert len(seen) >= 4
        assert best.aligned_results[0]["score"] == 1.0
