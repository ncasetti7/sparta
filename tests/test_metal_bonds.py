"""Tests for metal typing and metal bond bounds (sparta.metal_bonds)."""

import warnings
from copy import copy

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.rdchem import HybridizationType

from sparta import Reaction, cheminformatics
from sparta.metal_bonds import (
    METAL_BOND_TOLERANCE,
    TYPICAL_METAL_BOND_LENGTHS,
    MetalBondLengthWarning,
    tighten_metal_bond_bounds,
    untyped_metal_atoms,
)

# Ligand-free Pd CMD step from the Pistachio benchmark: C20-H21 breaks, Pd8-O9 breaks,
# C20-Pd8 and O9-H21 form.
PD_REACTANT = (
    "[F:1][c:2]1[c:3]([H:4])[c:5]([H:6])[c:7]([Pd:8][O:9][C:10]([C:11]([H:12])([H:13])"
    "[H:14])=[O:15])[c:16]([H:17])[c:18]1[H:19].[c:20]1([H:21])[c:22]([H:23])[s:24]"
    "[c:25]([C:26](=[O:27])[H:28])[c:29]1[H:30]"
)
PD_PRODUCT = (
    "[H:21][O:9][C:10]([C:11]([H:12])([H:13])[H:14])=[O:15].[c:20]1([Pd:8][c:7]2[c:5]"
    "([H:6])[c:3]([H:4])[c:2]([F:1])[c:18]([H:19])[c:16]2[H:17])[c:22]([H:23])[s:24]"
    "[c:25]([C:26](=[O:27])[H:28])[c:29]1[H:30]"
)
# Pd(II) with a dative phosphine, as the benchmark writes ligated complexes
PD_PHOSPHINE = "[CH3:1][Pd:2]([Cl:3])<-[PH3:4]"
# Rh has no UFF type RDKit can assign, so its bonds take the fallback lengths
RH_COMPLEX = "[CH3:1][Rh:2]([Cl:3])[Cl:4]"
RH_SELENIDE = "[CH3:1][Rh:2][SeH:3]"
PD, PD_C, PD_O = 7, 19, 8  # zero-based indices in the CMD fixture


def _bounds(mol: Chem.Mol, i: int, j: int) -> tuple[float, float]:
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    i, j = sorted((i, j))
    return bounds[j, i], bounds[i, j]


def test_pd_is_typed_square_planar_for_uff():
    """make_mol gives Pd the SP2D hybridization UFF types as Pd4+2.

    RDKit perceives SP3D2, which UFF has no Pd type for.
    """
    mol = cheminformatics.make_mol(PD_REACTANT)
    assert mol.GetAtomWithIdx(PD).GetHybridization() == HybridizationType.SP2D
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(mol)
    assert untyped_metal_atoms(mol) == set()


@pytest.mark.parametrize(
    ("smiles", "pair", "expected"),
    [
        (PD_REACTANT, (PD, PD_O), 1.96),
        (PD_PRODUCT, (PD, PD_C), 2.06),
        (PD_PHOSPHINE, (1, 3), 2.43),
    ],
)
def test_pd_bond_bounds_are_physical(smiles, pair, expected):
    """Bonds to typed Pd, dative ones included, get narrow UFF bounds, not ~1 A windows."""
    lower, upper = _bounds(cheminformatics.make_mol(smiles), *pair)
    assert upper - lower < 0.1
    assert lower <= expected <= upper


def test_metal_hybridization_survives_copy_and_embedding():
    """The UFF typing persists through copying and reference-conformer embedding."""
    mol = cheminformatics.make_mol(PD_PRODUCT)
    reference = cheminformatics.generate_conformer(copy(mol), PD_PRODUCT, seed=0)
    for molecule in (copy(mol), reference):
        assert molecule.GetAtomWithIdx(PD).GetHybridization() == HybridizationType.SP2D
    positions = reference.GetConformer().GetPositions()
    assert np.linalg.norm(positions[PD] - positions[PD_C]) == pytest.approx(2.06, abs=0.1)


def test_typed_metal_reaction_emits_no_fallback_warning():
    """A Pd reaction uses UFF throughout, so no fallback length is ever used."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetalBondLengthWarning)
        reaction = Reaction(reactant_smiles=PD_REACTANT, product_smiles=PD_PRODUCT)
    assert reaction.prod_bond_lengths[(PD, PD_C)] == pytest.approx(2.06, abs=0.1)


def test_untyped_metal_gets_tabulated_bounds_with_warning():
    """Bonds to a metal UFF cannot type are bounded around the tabulated length."""
    mol = cheminformatics.make_mol(RH_COMPLEX)
    metals = untyped_metal_atoms(mol)
    assert metals == {1}
    with pytest.warns(MetalBondLengthWarning, match="tabulated"):
        bounds = tighten_metal_bond_bounds(rdDistGeom.GetMoleculeBoundsMatrix(mol), mol, metals)
    length = TYPICAL_METAL_BOND_LENGTHS["Rh"]["C"]
    assert bounds[1, 0] == pytest.approx(length - METAL_BOND_TOLERANCE)
    assert bounds[0, 1] == pytest.approx(length + METAL_BOND_TOLERANCE)


def test_untabulated_metal_pair_falls_back_to_covalent_radii():
    """A metal-ligand pair missing from the table uses the covalent-radius sum."""
    mol = cheminformatics.make_mol(RH_SELENIDE)
    table = Chem.GetPeriodicTable()
    radii = table.GetRcovalent(45) + table.GetRcovalent(34)
    with pytest.warns(MetalBondLengthWarning, match="covalent-radius"):
        bounds = tighten_metal_bond_bounds(
            rdDistGeom.GetMoleculeBoundsMatrix(mol), mol, untyped_metal_atoms(mol)
        )
    assert bounds[1, 2] == pytest.approx(radii + METAL_BOND_TOLERANCE)


def test_untyped_metal_reaction_uses_fallback_reference_lengths():
    """Reference lengths of changing bonds to an untyped metal come from the table."""
    with pytest.warns(MetalBondLengthWarning):
        reaction = Reaction(
            reactant_smiles="[CH3:1][Rh:2]([Cl:3])[Cl:4]",
            product_smiles="[CH3:1].[Rh:2]([Cl:3])[Cl:4]",
        )
    assert reaction.reac_bond_lengths[(0, 1)] == pytest.approx(TYPICAL_METAL_BOND_LENGTHS["Rh"]["C"])


@pytest.fixture(scope="module")
def aligned_pd_reaction():
    reaction = Reaction(
        reactant_smiles=PD_REACTANT,
        product_smiles=PD_PRODUCT,
        n_conformers=2,
        n_seeds=1,
        stretch_factors=1.5,
        top_k=1,
        n_images=5,
    )
    reaction.generate_initial_conformers()
    reaction.run_iterative_alignment(max_iterations=5)
    return reaction


def test_aligned_pd_endpoints_keep_their_bonds(aligned_pd_reaction):
    """Alignment never breaks an endpoint's own Pd bond.

    Before typing Pd and enforcing bounds, the product's Pd-C drifted to 2.9-3.5 A.
    """
    result = aligned_pd_reaction.aligned_results[0]
    product = np.asarray(result["product_coords"])
    reactant = np.asarray(result["reactant_coords"])
    assert result["max_bond_violation"] <= 0.05
    assert np.linalg.norm(product[PD] - product[PD_C]) < 2.4
    assert np.linalg.norm(reactant[PD] - reactant[PD_O]) < 2.3
