"""UFF typing and fallback bond-length bounds for metal atoms."""

import warnings
from typing import Dict, Set

import numpy as np
from rdkit import Chem, DistanceGeometry
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdchem import HybridizationType

METAL_ATOMIC_NUMBERS = frozenset({*range(21, 31), *range(39, 49), 57, *range(72, 81)})

# hybridization giving each metal a UFF type (e.g. SP2D -> Pd4+2); others use the fallback
UFF_METAL_HYBRIDIZATION: Dict[str, HybridizationType] = {
    "Pd": HybridizationType.SP2D,
    "Pt": HybridizationType.SP2D,
    "Ni": HybridizationType.SP2D,
    "Au": HybridizationType.SP2D,
    "Co": HybridizationType.SP3D2,
    "Fe": HybridizationType.SP3D2,
    "Mn": HybridizationType.SP3D2,
    "Cr": HybridizationType.SP3D2,
    "Cu": HybridizationType.SP3,
    "Zn": HybridizationType.SP3,
    "Ti": HybridizationType.SP3,
    "Ag": HybridizationType.SP,
}

# half-width (Å) of the fallback bounds window
METAL_BOND_TOLERANCE = 0.15

# approximate metal-ligand bond lengths (Å), used only when UFF cannot type the metal
TYPICAL_METAL_BOND_LENGTHS: Dict[str, Dict[str, float]] = {
    "Pd": {"C": 2.00, "N": 2.05, "O": 2.05, "P": 2.30, "S": 2.30, "Cl": 2.32, "Br": 2.45,
           "I": 2.62, "H": 1.55},
    "Pt": {"C": 2.03, "N": 2.05, "O": 2.03, "P": 2.28, "S": 2.30, "Cl": 2.32, "H": 1.60},
    "Ni": {"C": 1.90, "N": 1.95, "O": 1.95, "P": 2.20, "S": 2.20, "Cl": 2.20, "H": 1.45},
    "Cu": {"C": 1.95, "N": 2.00, "O": 1.95, "P": 2.25, "S": 2.30, "Cl": 2.25, "H": 1.50},
    "Ag": {"C": 2.10, "N": 2.20, "O": 2.30, "P": 2.40, "S": 2.50, "Cl": 2.50, "H": 1.65},
    "Au": {"C": 2.03, "N": 2.05, "O": 2.05, "P": 2.28, "S": 2.30, "Cl": 2.28, "H": 1.60},
    "Rh": {"C": 2.05, "N": 2.10, "O": 2.10, "P": 2.30, "S": 2.35, "Cl": 2.38, "H": 1.55},
    "Ir": {"C": 2.05, "N": 2.10, "O": 2.10, "P": 2.30, "S": 2.35, "Cl": 2.40, "H": 1.60},
    "Ru": {"C": 2.05, "N": 2.10, "O": 2.10, "P": 2.33, "S": 2.35, "Cl": 2.42, "H": 1.60},
    "Co": {"C": 1.95, "N": 1.95, "O": 1.95, "P": 2.22, "S": 2.25, "Cl": 2.25, "H": 1.50},
    "Fe": {"C": 2.00, "N": 2.00, "O": 2.00, "P": 2.25, "S": 2.30, "Cl": 2.30, "H": 1.55},
}


class MetalBondLengthWarning(UserWarning):
    """A metal bond length came from a fallback rather than UFF parameters."""


def is_metal(atom: Chem.Atom) -> bool:
    """Return whether an atom is a d-block metal."""
    return atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS


def assign_metal_hybridization(mol: Chem.Mol) -> Chem.Mol:
    """
    Set the UFF-typable hybridization on every supported metal atom, in place.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule whose metal atoms should be typed.

    Returns
    -------
    Chem.Mol
        The same molecule, for chaining.
    """
    for atom in mol.GetAtoms():
        hybridization = UFF_METAL_HYBRIDIZATION.get(atom.GetSymbol())
        if hybridization is not None:
            atom.SetHybridization(hybridization)
    return mol


def untyped_metal_atoms(mol: Chem.Mol) -> Set[int]:
    """
    Return the indices of metal atoms whose bonds need fallback bounds.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule after :func:`assign_metal_hybridization`.

    Returns
    -------
    set of int
        Atom indices of metals that UFF cannot type.
    """
    metals = {atom.GetIdx() for atom in mol.GetAtoms() if is_metal(atom)}
    if not metals or rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        return set()
    unsupported = {
        index for index in metals
        if mol.GetAtomWithIdx(index).GetSymbol() not in UFF_METAL_HYBRIDIZATION
    }
    return unsupported or metals


def metal_bond_length(mol: Chem.Mol, i: int, j: int) -> float:
    """
    Return a tabulated or covalent-radius length for a bond to a metal, with a warning.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule containing the bond.
    i, j : int
        Atom indices of the bond; at least one is a metal.

    Returns
    -------
    float
        Bond length in Å.
    """
    first, second = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
    metal, ligand = (first, second) if is_metal(first) else (second, first)
    tabulated = TYPICAL_METAL_BOND_LENGTHS.get(metal.GetSymbol(), {}).get(ligand.GetSymbol())
    if tabulated is not None:
        warnings.warn(
            f"UFF has no parameters for {metal.GetSymbol()}; using the tabulated "
            f"{metal.GetSymbol()}-{ligand.GetSymbol()} length {tabulated:.2f} A for atoms "
            f"{i + 1}-{j + 1}",
            MetalBondLengthWarning,
            stacklevel=2,
        )
        return tabulated
    table = Chem.GetPeriodicTable()
    length = table.GetRcovalent(metal.GetAtomicNum()) + table.GetRcovalent(ligand.GetAtomicNum())
    warnings.warn(
        f"UFF has no parameters for {metal.GetSymbol()} and no {metal.GetSymbol()}-"
        f"{ligand.GetSymbol()} length is tabulated; using the covalent-radius sum "
        f"{length:.2f} A for atoms {i + 1}-{j + 1}",
        MetalBondLengthWarning,
        stacklevel=2,
    )
    return float(length)


def tighten_metal_bond_bounds(
    bounds_matrix: np.ndarray, mol: Chem.Mol, metal_indices: Set[int]
) -> np.ndarray:
    """
    Bound each bond to an untyped metal at ``metal_bond_length ± METAL_BOND_TOLERANCE``
    and re-run triangle smoothing.

    Parameters
    ----------
    bounds_matrix : np.ndarray
        RDKit bounds matrix (upper bounds above the diagonal, lower below).
    mol : Chem.Mol
        Molecule the bounds belong to.
    metal_indices : set of int
        Metal atoms whose bonds get fallback bounds.

    Returns
    -------
    np.ndarray
        New, smoothed bounds matrix.

    Raises
    ------
    ValueError
        The tightened bounds cannot be triangle-smoothed consistently.
    """
    bounds = bounds_matrix.copy()
    for bond in mol.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if i not in metal_indices and j not in metal_indices:
            continue
        length = metal_bond_length(mol, i, j)
        bounds[i, j] = length + METAL_BOND_TOLERANCE
        bounds[j, i] = length - METAL_BOND_TOLERANCE
    if not DistanceGeometry.DoTriangleSmoothing(bounds):
        raise ValueError("metal bond bounds are inconsistent with the rest of the molecule")
    return bounds
