"""Tests for sparta.calculation (base, ASECalculation, XTBCalculation)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ase
from ase import Atoms

from sparta.calculation.base import Calculation
from sparta.calculation.ase_calc import ASECalculation
from sparta.calculation.xtb_calc import XTBCalculation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

XTB_OUTPUT = """\
      -----------------------------------------------------------
     |                   =====================                   |
     |                           x T B                           |
     |                   =====================                   |
          | TOTAL ENERGY              -5.768755329182 Eh   |
          | GRADIENT NORM               0.000000000000 Eh/a0 |
"""

XTB_OUTPUT_ALTERNATE = """\
  TOTAL ENERGY            -5.768755329182 Eh
"""


@pytest.fixture
def lj_calculator():
    """LennardJones calculator — pure ASE, no external binary required."""
    from ase.calculators.lj import LennardJones
    return LennardJones(rc=10.0)


@pytest.fixture
def h2_near_eq():
    """H2 slightly inside LJ equilibrium (~1.12 Å); non-zero force present."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])


@pytest.fixture
def h2_at_rest():
    """H2 at a geometry where forces are negligible (well outside LJ well)."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 5.0]])


def _mock_xtb_sp(returncode=0, stdout=XTB_OUTPUT, stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _mock_xtb_opt_side_effect(cwd, opt_atoms):
    """Return a side-effect function that writes xtbopt.xyz before returning."""
    import ase.io

    def side_effect(cmd, **kwargs):
        ase.io.write(os.path.join(kwargs.get("cwd", cwd), "xtbopt.xyz"), opt_atoms, format="xyz")
        return MagicMock(returncode=0, stdout=XTB_OUTPUT, stderr="")

    return side_effect


# ---------------------------------------------------------------------------
# TestCalculationBase
# ---------------------------------------------------------------------------

class TestCalculationBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Calculation()

    def test_subclass_missing_optimize_raises(self):
        class OnlyEnergy(Calculation):
            def energy(self, atoms):
                return 0.0

        with pytest.raises(TypeError):
            OnlyEnergy()

    def test_subclass_missing_energy_raises(self):
        class OnlyOptimize(Calculation):
            def optimize(self, atoms):
                return atoms, 0.0

        with pytest.raises(TypeError):
            OnlyOptimize()

    def test_default_charge_and_multiplicity(self):
        class Concrete(Calculation):
            def energy(self, atoms): return 0.0
            def optimize(self, atoms): return atoms, 0.0

        calc = Concrete()
        assert calc.charge == 0
        assert calc.multiplicity == 1

    def test_custom_charge_and_multiplicity_stored(self):
        class Concrete(Calculation):
            def energy(self, atoms): return 0.0
            def optimize(self, atoms): return atoms, 0.0

        calc = Concrete(charge=-2, multiplicity=3)
        assert calc.charge == -2
        assert calc.multiplicity == 3


# ---------------------------------------------------------------------------
# TestASECalculation
# ---------------------------------------------------------------------------

class TestASECalculation:
    def test_is_calculation_subclass(self, lj_calculator):
        assert isinstance(ASECalculation(lj_calculator), Calculation)

    def test_default_charge_multiplicity(self, lj_calculator):
        calc = ASECalculation(lj_calculator)
        assert calc.charge == 0
        assert calc.multiplicity == 1

    def test_custom_charge_multiplicity_stored(self, lj_calculator):
        calc = ASECalculation(lj_calculator, charge=1, multiplicity=2)
        assert calc.charge == 1
        assert calc.multiplicity == 2

    def test_default_fmax_max_steps(self, lj_calculator):
        calc = ASECalculation(lj_calculator)
        assert calc.fmax == 0.05
        assert calc.max_steps == 200

    def test_custom_fmax_max_steps_stored(self, lj_calculator):
        calc = ASECalculation(lj_calculator, fmax=0.01, max_steps=500)
        assert calc.fmax == 0.01
        assert calc.max_steps == 500

    def test_energy_returns_float(self, lj_calculator, h2_near_eq):
        energy = ASECalculation(lj_calculator).energy(h2_near_eq)
        assert isinstance(energy, float)

    def test_energy_is_finite(self, lj_calculator, h2_near_eq):
        energy = ASECalculation(lj_calculator).energy(h2_near_eq)
        assert np.isfinite(energy)

    def test_energy_does_not_mutate_atoms(self, lj_calculator, h2_near_eq):
        original_pos = h2_near_eq.get_positions().copy()
        ASECalculation(lj_calculator).energy(h2_near_eq)
        np.testing.assert_array_equal(h2_near_eq.get_positions(), original_pos)

    def test_optimize_returns_tuple_of_two(self, lj_calculator, h2_near_eq):
        result = ASECalculation(lj_calculator).optimize(h2_near_eq)
        assert isinstance(result, tuple) and len(result) == 2

    def test_optimize_returns_atoms_and_float(self, lj_calculator, h2_near_eq):
        opt_atoms, energy = ASECalculation(lj_calculator).optimize(h2_near_eq)
        assert isinstance(opt_atoms, Atoms)
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_optimize_does_not_mutate_input(self, lj_calculator, h2_near_eq):
        original_pos = h2_near_eq.get_positions().copy()
        ASECalculation(lj_calculator).optimize(h2_near_eq)
        np.testing.assert_array_equal(h2_near_eq.get_positions(), original_pos)

    def test_optimize_moves_atoms_from_non_equilibrium(self, lj_calculator, h2_near_eq):
        original_pos = h2_near_eq.get_positions().copy()
        opt_atoms, _ = ASECalculation(lj_calculator).optimize(h2_near_eq)
        assert not np.allclose(opt_atoms.get_positions(), original_pos, atol=1e-3)

    def test_optimize_preserves_atom_count(self, lj_calculator, h2_near_eq):
        opt_atoms, _ = ASECalculation(lj_calculator).optimize(h2_near_eq)
        assert len(opt_atoms) == len(h2_near_eq)

    def test_optimize_preserves_chemical_symbols(self, lj_calculator, h2_near_eq):
        opt_atoms, _ = ASECalculation(lj_calculator).optimize(h2_near_eq)
        assert opt_atoms.get_chemical_symbols() == h2_near_eq.get_chemical_symbols()


# ---------------------------------------------------------------------------
# TestXTBCalculation
# ---------------------------------------------------------------------------

class TestXTBCalculation:
    def test_is_calculation_subclass(self):
        assert isinstance(XTBCalculation(), Calculation)

    def test_default_charge_multiplicity(self):
        calc = XTBCalculation()
        assert calc.charge == 0
        assert calc.multiplicity == 1

    def test_custom_charge_multiplicity_stored(self):
        calc = XTBCalculation(charge=-1, multiplicity=2)
        assert calc.charge == -1
        assert calc.multiplicity == 2

    def test_default_xtb_cmd(self):
        assert XTBCalculation()._xtb_cmd == "xtb"

    def test_custom_xtb_cmd_stored(self):
        calc = XTBCalculation(xtb_cmd="/usr/local/bin/xtb")
        assert calc._xtb_cmd == "/usr/local/bin/xtb"

    # -- Energy parsing --

    def test_parse_energy_pipe_format(self):
        calc = XTBCalculation()
        assert calc._parse_energy(XTB_OUTPUT) == pytest.approx(-5.768755329182)

    def test_parse_energy_plain_format(self):
        calc = XTBCalculation()
        assert calc._parse_energy(XTB_OUTPUT_ALTERNATE) == pytest.approx(-5.768755329182)

    def test_parse_energy_negative_value(self):
        calc = XTBCalculation()
        out = "  | TOTAL ENERGY  -14.957914568831 Eh  |\n"
        assert calc._parse_energy(out) == pytest.approx(-14.957914568831)

    def test_parse_energy_raises_if_missing(self):
        calc = XTBCalculation()
        with pytest.raises(ValueError, match="TOTAL ENERGY"):
            calc._parse_energy("no useful output here\n")

    # -- energy() method --

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_returns_correct_value(self, mock_run):
        mock_run.return_value = _mock_xtb_sp()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        energy = XTBCalculation().energy(atoms)
        assert energy == pytest.approx(-5.768755329182)

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_returns_float(self, mock_run):
        mock_run.return_value = _mock_xtb_sp()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        assert isinstance(XTBCalculation().energy(atoms), float)

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_uses_gxtb_flag(self, mock_run):
        mock_run.return_value = _mock_xtb_sp()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation().energy(atoms)
        cmd = mock_run.call_args[0][0]
        assert "--gxtb" in cmd

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_uses_sp_flag(self, mock_run):
        mock_run.return_value = _mock_xtb_sp()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation().energy(atoms)
        cmd = mock_run.call_args[0][0]
        assert "--sp" in cmd

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_passes_correct_charge(self, mock_run):
        mock_run.return_value = _mock_xtb_sp()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation(charge=-1).energy(atoms)
        cmd = mock_run.call_args[0][0]
        chrg_idx = cmd.index("--chrg")
        assert cmd[chrg_idx + 1] == "-1"

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_passes_correct_uhf(self, mock_run):
        """--uhf should equal multiplicity - 1."""
        mock_run.return_value = _mock_xtb_sp()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation(multiplicity=3).energy(atoms)
        cmd = mock_run.call_args[0][0]
        uhf_idx = cmd.index("--uhf")
        assert cmd[uhf_idx + 1] == "2"

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_energy_raises_on_nonzero_returncode(self, mock_run):
        mock_run.return_value = _mock_xtb_sp(returncode=1, stderr="fatal error")
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        with pytest.raises(RuntimeError, match="xtb exited"):
            XTBCalculation().energy(atoms)

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    @patch("sparta.calculation.xtb_calc.tempfile.mkdtemp")
    def test_energy_tempdir_cleaned_up(self, mock_mkdtemp, mock_run, tmp_path):
        tmpdir = str(tmp_path / "xtb_scratch")
        os.makedirs(tmpdir)
        mock_mkdtemp.return_value = tmpdir
        mock_run.return_value = _mock_xtb_sp()

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation().energy(atoms)
        assert not os.path.exists(tmpdir)

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    @patch("sparta.calculation.xtb_calc.tempfile.mkdtemp")
    def test_energy_tempdir_cleaned_up_on_failure(self, mock_mkdtemp, mock_run, tmp_path):
        tmpdir = str(tmp_path / "xtb_scratch_fail")
        os.makedirs(tmpdir)
        mock_mkdtemp.return_value = tmpdir
        mock_run.return_value = _mock_xtb_sp(returncode=1, stderr="oops")

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        with pytest.raises(RuntimeError):
            XTBCalculation().energy(atoms)
        assert not os.path.exists(tmpdir)

    # -- optimize() method --

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_optimize_returns_atoms_and_float(self, mock_run, tmp_path):
        opt_atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])

        def side_effect(cmd, **kwargs):
            ase.io.write(os.path.join(kwargs["cwd"], "xtbopt.xyz"), opt_atoms, format="xyz")
            return MagicMock(returncode=0, stdout=XTB_OUTPUT, stderr="")

        mock_run.side_effect = side_effect

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        result_atoms, energy = XTBCalculation().optimize(atoms)
        assert isinstance(result_atoms, Atoms)
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_optimize_uses_opt_flag(self, mock_run, tmp_path):
        opt_atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])

        def side_effect(cmd, **kwargs):
            ase.io.write(os.path.join(kwargs["cwd"], "xtbopt.xyz"), opt_atoms, format="xyz")
            return MagicMock(returncode=0, stdout=XTB_OUTPUT, stderr="")

        mock_run.side_effect = side_effect

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation().optimize(atoms)
        cmd = mock_run.call_args[0][0]
        assert "--opt" in cmd
        assert "--sp" not in cmd

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    def test_optimize_reads_xtbopt_xyz(self, mock_run):
        opt_positions = np.array([[0, 0, 0], [0, 0, 0.777]])
        opt_atoms = Atoms("H2", positions=opt_positions)

        def side_effect(cmd, **kwargs):
            ase.io.write(os.path.join(kwargs["cwd"], "xtbopt.xyz"), opt_atoms, format="xyz")
            return MagicMock(returncode=0, stdout=XTB_OUTPUT, stderr="")

        mock_run.side_effect = side_effect

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        result_atoms, _ = XTBCalculation().optimize(atoms)
        np.testing.assert_allclose(result_atoms.get_positions(), opt_positions, atol=1e-4)

    @patch("sparta.calculation.xtb_calc.subprocess.run")
    @patch("sparta.calculation.xtb_calc.tempfile.mkdtemp")
    def test_optimize_tempdir_cleaned_up(self, mock_mkdtemp, mock_run, tmp_path):
        tmpdir = str(tmp_path / "xtb_opt")
        os.makedirs(tmpdir)
        mock_mkdtemp.return_value = tmpdir
        opt_atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])

        def side_effect(cmd, **kwargs):
            ase.io.write(os.path.join(tmpdir, "xtbopt.xyz"), opt_atoms, format="xyz")
            return MagicMock(returncode=0, stdout=XTB_OUTPUT, stderr="")

        mock_run.side_effect = side_effect

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
        XTBCalculation().optimize(atoms)
        assert not os.path.exists(tmpdir)
