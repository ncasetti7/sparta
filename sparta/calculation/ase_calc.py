from typing import Tuple

import ase
import ase.calculators.calculator
from ase.optimize import LBFGS

from sparta.calculation.base import Calculation


class ASECalculation(Calculation):
    """
    Wraps a generic ASE calculator for single-point energy evaluation and
    geometry optimization.

    The caller is responsible for configuring the calculator (including any
    charge/multiplicity settings it requires) before passing it in.

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Any ASE-compatible calculator instance.
    charge : int
        Total molecular charge (stored as metadata; apply to the calculator
        before passing it in if the calculator requires it).
    multiplicity : int
        Spin multiplicity (stored as metadata).
    fmax : float
        Force convergence criterion for :meth:`optimize` (eV/Å).
    max_steps : int
        Maximum optimization steps.
    """

    def __init__(
        self,
        calculator: ase.calculators.calculator.Calculator,
        charge: int = 0,
        multiplicity: int = 1,
        fmax: float = 0.05,
        max_steps: int = 200,
    ) -> None:
        super().__init__(charge, multiplicity)
        self._calculator = calculator
        self.fmax = fmax
        self.max_steps = max_steps

    def energy(self, atoms: ase.Atoms) -> float:
        atoms = atoms.copy()
        atoms.calc = self._calculator
        return float(atoms.get_potential_energy())

    def optimize(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, float]:
        atoms = atoms.copy()
        atoms.calc = self._calculator
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        return atoms, float(atoms.get_potential_energy())
