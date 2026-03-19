"""Tests for sparta.iterative_alignment.idpp."""

import numpy as np
import pytest
import ase

from sparta.iterative_alignment.idpp import IDPP, create_idpp_calculators


def _make_atoms(positions, symbols=None):
    if symbols is None:
        symbols = ["F"] * len(positions)
    return ase.Atoms(symbols=symbols, positions=np.array(positions))


# ---------------------------------------------------------------------------
# IDPP calculator
# ---------------------------------------------------------------------------

class TestIDPP:
    def test_zero_energy_at_target_geometry(self):
        """When current positions reproduce the linearly interpolated target
        distances exactly, IDPP energy should be zero."""
        initial = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        alpha = 0.5
        # Target = interpolated geometry
        target_pos = (1 - alpha) * initial + alpha * final
        calc = IDPP(initial, final, alpha=alpha)
        atoms = _make_atoms(target_pos, ["F", "F", "F"])
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert abs(energy) < 1e-8

    def test_positive_energy_when_perturbed(self):
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        calc = IDPP(initial, final, alpha=0.5)
        # Perturb significantly from the interpolated geometry
        atoms = _make_atoms([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert energy > 0.0

    def test_forces_shape(self):
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        calc = IDPP(initial, final, alpha=0.5)
        atoms = _make_atoms([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
        atoms.calc = calc
        forces = atoms.get_forces()
        assert forces.shape == (2, 3)

    def test_forces_zero_at_target(self):
        """Forces must vanish when current geometry = target geometry."""
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        alpha = 0.5
        target_pos = (1 - alpha) * initial + alpha * final
        calc = IDPP(initial, final, alpha=alpha)
        atoms = _make_atoms(target_pos)
        atoms.calc = calc
        forces = atoms.get_forces()
        np.testing.assert_allclose(forces, np.zeros((2, 3)), atol=1e-8)

    def test_alpha_out_of_range_raises(self):
        initial = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            IDPP(initial, final, alpha=1.5)

    def test_shape_mismatch_raises(self):
        initial = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            IDPP(initial, final, alpha=0.5)

    def test_alpha_zero_targets_initial_geometry(self):
        """At alpha=0, target distances equal the initial geometry distances."""
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        calc = IDPP(initial, final, alpha=0.0)
        atoms = _make_atoms(initial.copy())
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert abs(energy) < 1e-8

    def test_d_power_affects_weighting(self):
        """Higher d_power should give different (generally lower) energy for same deviation."""
        initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        perturbed = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        calc4 = IDPP(initial, final, alpha=0.5, d_power=4)
        calc2 = IDPP(initial, final, alpha=0.5, d_power=2)
        atoms4 = _make_atoms(perturbed)
        atoms2 = _make_atoms(perturbed)
        atoms4.calc = calc4
        atoms2.calc = calc2
        e4 = atoms4.get_potential_energy()
        e2 = atoms2.get_potential_energy()
        assert e4 != e2


# ---------------------------------------------------------------------------
# create_idpp_calculators
# ---------------------------------------------------------------------------

class TestCreateIdppCalculators:
    def test_returns_n_minus_2_calculators(self):
        initial = _make_atoms([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = _make_atoms([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        calcs = create_idpp_calculators(initial, final, n_images=7)
        assert len(calcs) == 5  # 7 - 2 intermediate images

    def test_alpha_values_are_correct(self):
        initial = _make_atoms([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        final = _make_atoms([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        n = 5
        calcs = create_idpp_calculators(initial, final, n_images=n)
        expected_alphas = [k / (n - 1) for k in range(1, n - 1)]
        for calc, expected in zip(calcs, expected_alphas):
            assert abs(calc.alpha - expected) < 1e-10

    def test_minimum_images_is_3(self):
        initial = _make_atoms([[0.0, 0.0, 0.0]])
        final = _make_atoms([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            create_idpp_calculators(initial, final, n_images=2)

    def test_calculators_are_idpp_instances(self):
        initial = _make_atoms([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        final = _make_atoms([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        calcs = create_idpp_calculators(initial, final, n_images=4)
        assert all(isinstance(c, IDPP) for c in calcs)
