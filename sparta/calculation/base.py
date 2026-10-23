from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import ase


class Calculation(ABC):
    """
    Abstract base class for single-point and geometry optimization calculations.

    Subclasses must implement :meth:`energy` and :meth:`optimize`.

    Parameters
    ----------
    charge : int
        Total molecular charge.
    multiplicity : int
        Spin multiplicity (2S + 1).
    """

    def __init__(self, charge: int = 0, multiplicity: int = 1) -> None:
        self.charge = charge
        self.multiplicity = multiplicity

    @abstractmethod
    def energy(self, atoms: "ase.Atoms") -> float:
        """Return the single-point energy (eV or method-native units) for *atoms*."""

    @abstractmethod
    def optimize(self, atoms: "ase.Atoms") -> "Tuple[ase.Atoms, float]":
        """Return an optimized copy of *atoms* and its energy."""
