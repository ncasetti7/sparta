"""Tests for sparta.cheminformatics."""

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from sparta import cheminformatics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ethane_mol():
    """Ethane with explicit H's and atom maps 1–8."""
    smi = "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])"
    return cheminformatics.make_mol(smi)


def _embed(mol):
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return mol


# ---------------------------------------------------------------------------
# make_mol
# ---------------------------------------------------------------------------

class TestMakeMol:
    def test_atom_ordering_by_map(self):
        # Map 2 first in SMILES, map 1 second → after reorder index 0 = Cl (map 1)
        mol = cheminformatics.make_mol("[F:2][Cl:1]")
        assert mol.GetAtomWithIdx(0).GetSymbol() == "Cl"
        assert mol.GetAtomWithIdx(1).GetSymbol() == "F"

    def test_num_atoms_no_implicit_h(self):
        # removeHs=False: HF has 2 atoms
        mol = cheminformatics.make_mol("[H:1][F:2]")
        assert mol.GetNumAtoms() == 2

    def test_multi_fragment(self):
        mol = cheminformatics.make_mol("[F:1][F:2].[Cl:3][Cl:4]")
        assert mol.GetNumAtoms() == 4


# ---------------------------------------------------------------------------
# add_coords_to_mol / add_more_coords_to_mol
# ---------------------------------------------------------------------------

class TestCoordManagement:
    def test_add_coords_sets_conformer(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        mol = cheminformatics.add_coords_to_mol(coords, mol)
        assert mol.GetNumConformers() == 1
        pos = mol.GetConformer().GetAtomPosition(1)
        assert abs(pos.x - 1.4) < 1e-6

    def test_add_coords_wrong_length_raises(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        with pytest.raises(ValueError):
            cheminformatics.add_coords_to_mol(np.zeros((3, 3)), mol)

    def test_add_more_coords_appends(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        mol = cheminformatics.add_coords_to_mol(coords, mol)
        mol = cheminformatics.add_more_coords_to_mol(coords + 5.0, mol)
        assert mol.GetNumConformers() == 2

    def test_add_more_coords_wrong_length_raises(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        with pytest.raises(ValueError):
            cheminformatics.add_more_coords_to_mol(np.zeros((5, 3)), mol)


# ---------------------------------------------------------------------------
# get_coords_conformer / get_all_coords_all_conformers
# ---------------------------------------------------------------------------

class TestGetCoords:
    def test_get_coords_shape(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        mol = cheminformatics.add_coords_to_mol(coords, mol)
        _, out = cheminformatics.get_coords_conformer(mol)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, coords, atol=1e-6)

    def test_get_all_coords_multiple_conformers(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        mol = cheminformatics.add_coords_to_mol(coords, mol)
        mol = cheminformatics.add_more_coords_to_mol(coords + 3.0, mol)
        all_c = cheminformatics.get_all_coords_all_conformers(mol)
        assert len(all_c) == 2
        assert all_c[0][1].shape == (2, 3)


# ---------------------------------------------------------------------------
# get_bonds
# ---------------------------------------------------------------------------

class TestGetBonds:
    def test_f2_single_bond(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        bonds = cheminformatics.get_bonds(mol)
        assert bonds == {(0, 1)}

    def test_multi_fragment_two_bonds(self):
        mol = cheminformatics.make_mol("[F:1][F:2].[Cl:3][Cl:4]")
        bonds = cheminformatics.get_bonds(mol)
        assert bonds == {(0, 1), (2, 3)}


# ---------------------------------------------------------------------------
# get_dihedral_matches_all
# ---------------------------------------------------------------------------

class TestGetDihedrals:
    def test_no_dihedrals_for_diatomic(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        assert cheminformatics.get_dihedral_matches_all(mol) == []

    def test_ethane_has_one_central_bond_dihedral(self):
        mol = _ethane_mol()
        dihedrals = cheminformatics.get_dihedral_matches_all(mol)
        # The C-C bond should give exactly one dihedral
        assert len(dihedrals) == 1
        a, b, c, d = dihedrals[0]
        # Central bond should be the C-C bond (atoms 0 and 1)
        assert {b, c} == {0, 1}

    def test_no_ring_bonds_included(self):
        # Benzene has only ring bonds — should return no dihedrals
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        dihedrals = cheminformatics.get_dihedral_matches_all(mol)
        # All bonds in benzene ring are ring bonds
        central_bonds = {(min(d[1], d[2]), max(d[1], d[2])) for d in dihedrals}
        for b, c in central_bonds:
            bond = mol.GetBondBetweenAtoms(b, c)
            assert not bond.IsInRing()


# ---------------------------------------------------------------------------
# generate_bonded_atoms_dict
# ---------------------------------------------------------------------------

class TestBondedAtomsDict:
    def test_empty_dihedrals(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        result = cheminformatics.generate_bonded_atoms_dict(mol, [])
        assert result == {}

    def test_ethane_rotating_fragment(self):
        mol = _ethane_mol()
        dihedrals = cheminformatics.get_dihedral_matches_all(mol)
        d = cheminformatics.generate_bonded_atoms_dict(mol, dihedrals)
        assert len(d) == 1
        key = dihedrals[0]
        # Rotating atoms should not include any of a, b, c
        a, b, c, _ = key
        rotating = d[key]
        assert b not in rotating
        assert a not in rotating


# ---------------------------------------------------------------------------
# filter_dihedrals_k_nearest
# ---------------------------------------------------------------------------

class TestFilterDihedrals:
    def test_empty_changing_bonds_returns_unchanged(self):
        mol = _ethane_mol()
        dihedrals = cheminformatics.get_dihedral_matches_all(mol)
        r, p = cheminformatics.filter_dihedrals_k_nearest(
            dihedrals, dihedrals, [], mol, mol, k=2
        )
        assert r == dihedrals
        assert p == dihedrals

    def test_k0_filters_non_center_dihedrals(self):
        # Ethane: one dihedral on the C-C bond. If the reaction center
        # is on that bond (atoms 0,1), k=0 should still include it.
        mol = _ethane_mol()
        dihedrals = cheminformatics.get_dihedral_matches_all(mol)
        changing = [(0, 1)]  # C-C bond is the reaction center
        r, _ = cheminformatics.filter_dihedrals_k_nearest(
            dihedrals, dihedrals, changing, mol, mol, k=0
        )
        assert len(r) == 1  # central bond is at distance 0

    def test_large_k_keeps_all(self):
        mol = _ethane_mol()
        dihedrals = cheminformatics.get_dihedral_matches_all(mol)
        changing = [(0, 1)]
        r, _ = cheminformatics.filter_dihedrals_k_nearest(
            dihedrals, dihedrals, changing, mol, mol, k=999
        )
        assert len(r) == len(dihedrals)


# ---------------------------------------------------------------------------
# kabsch_align
# ---------------------------------------------------------------------------

class TestKabschAlign:
    def test_already_aligned_no_change(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        result = cheminformatics.kabsch_align(coords.copy(), coords)
        np.testing.assert_allclose(result, coords, atol=1e-8)

    def test_rotation_recovery(self):
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        # Apply a 90-degree rotation around Z
        theta = math.pi / 2
        R = np.array([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta),  math.cos(theta), 0],
                      [0,                0,               1]])
        rotated = coords @ R.T
        aligned = cheminformatics.kabsch_align(rotated, coords)
        np.testing.assert_allclose(aligned, coords, atol=1e-7)

    def test_translation_invariance(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        shifted = coords + np.array([5.0, 3.0, -2.0])
        aligned = cheminformatics.kabsch_align(shifted, coords)
        np.testing.assert_allclose(aligned, coords, atol=1e-7)


# ---------------------------------------------------------------------------
# rotate_frag / translate_frag
# ---------------------------------------------------------------------------

class TestFragmentTransforms:
    def test_translate_frag_moves_correct_atoms(self):
        coords = np.zeros((4, 3))
        result = cheminformatics.translate_frag(coords, (1.0, 2.0, 3.0), [0, 1])
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[2], [0.0, 0.0, 0.0])  # untouched

    def test_translate_frags_length_mismatch_raises(self):
        coords = np.zeros((4, 3))
        with pytest.raises(ValueError):
            cheminformatics.translate_frags(coords, [1.0, 2.0], [[0, 1], [2, 3]])

    def test_rotate_frag_preserves_centroid_distance(self):
        coords = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0]])
        before_dist = np.linalg.norm(coords[0] - coords[1])
        result = cheminformatics.rotate_frag(coords.copy(), [45.0, 0.0, 0.0], [0, 1, 2])
        after_dist = np.linalg.norm(result[0] - result[1])
        assert abs(before_dist - after_dist) < 1e-6

    def test_rotate_frags_length_mismatch_raises(self):
        coords = np.zeros((4, 3))
        with pytest.raises(ValueError):
            cheminformatics.rotate_frags(coords, [1.0, 2.0], [[0, 1], [2, 3]])


# ---------------------------------------------------------------------------
# mol_to_ase_atoms
# ---------------------------------------------------------------------------

class TestMolToAse:
    def test_symbols_and_positions(self):
        mol = cheminformatics.make_mol("[F:1][Cl:2]")
        coords = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
        mol = cheminformatics.add_coords_to_mol(coords, mol)
        atoms = cheminformatics.mol_to_ase_atoms(mol)
        assert list(atoms.get_chemical_symbols()) == ["F", "Cl"]
        np.testing.assert_allclose(atoms.get_positions(), coords, atol=1e-6)


# ---------------------------------------------------------------------------
# find_mapped_substructure_match
# ---------------------------------------------------------------------------

class TestFindMappedSubstructureMatch:
    def test_common_bonds_found(self):
        # ethane reactant and product share the C-C bond
        smi = "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])"
        mol = cheminformatics.make_mol(smi)
        components = cheminformatics.find_mapped_substructure_match(mol, mol)
        # At minimum the entire molecule is one connected component
        assert any(len(c) > 1 for c in components)

    def test_disconnected_returns_empty_or_single_atom(self):
        mol1 = cheminformatics.make_mol("[F:1][F:2]")
        mol2 = cheminformatics.make_mol("[F:1].[F:2]")
        components = cheminformatics.find_mapped_substructure_match(mol1, mol2)
        # No common bonds between F2 and two isolated F atoms
        assert all(len(c) <= 1 for c in components)


# ---------------------------------------------------------------------------
# transfer_conformer
# ---------------------------------------------------------------------------

class TestTransferConformer:
    def test_transfer_copies_positions(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        mol = cheminformatics.add_coords_to_mol(coords, mol)
        dst = cheminformatics.make_mol("[F:1][F:2]")
        cheminformatics.transfer_conformer(mol, dst)
        _, dst_coords = cheminformatics.get_coords_conformer(dst)
        np.testing.assert_allclose(dst_coords, coords, atol=1e-6)

    def test_transfer_atom_count_mismatch_raises(self):
        src = cheminformatics.make_mol("[F:1][F:2]")
        dst = cheminformatics.make_mol("[F:1][F:2].[Cl:3][Cl:4]")
        coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        src = cheminformatics.add_coords_to_mol(coords, src)
        with pytest.raises(ValueError):
            cheminformatics.transfer_conformer(src, dst)


# ---------------------------------------------------------------------------
# generate_conformer
# ---------------------------------------------------------------------------

class TestGenerateConformer:
    def test_returns_mol_with_one_conformer(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        mol = cheminformatics.generate_conformer(mol, "[F:1][F:2]", seed=0)
        assert mol.GetNumConformers() == 1

    def test_coordinates_are_finite(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        mol = cheminformatics.generate_conformer(mol, "[F:1][F:2]", seed=0)
        _, coords = cheminformatics.get_coords_conformer(mol)
        assert np.all(np.isfinite(coords))

    def test_coordinate_shape(self):
        mol = cheminformatics.make_mol("[F:1][F:2]")
        mol = cheminformatics.generate_conformer(mol, "[F:1][F:2]", seed=0)
        _, coords = cheminformatics.get_coords_conformer(mol)
        assert coords.shape == (2, 3)

    def test_bond_length_reasonable_for_ff(self):
        """F-F bond should land in a physically plausible range after UFF opt."""
        mol = cheminformatics.make_mol("[F:1][F:2]")
        mol = cheminformatics.generate_conformer(mol, "[F:1][F:2]", seed=0)
        _, coords = cheminformatics.get_coords_conformer(mol)
        dist = float(np.linalg.norm(coords[0] - coords[1]))
        assert 1.0 < dist < 2.5

    def test_multi_fragment_smiles(self):
        """generate_conformer must succeed for disconnected SMILES."""
        smi = "[F:1][F:2].[Cl:3][Cl:4]"
        mol = cheminformatics.make_mol(smi)
        mol = cheminformatics.generate_conformer(mol, smi, seed=0)
        assert mol.GetNumConformers() == 1
        _, coords = cheminformatics.get_coords_conformer(mol)
        assert coords.shape == (4, 3)
        assert np.all(np.isfinite(coords))

    def test_seed_reproducibility(self):
        """Same seed should produce the same conformer."""
        smi = "[F:1][F:2]"
        mol_a = cheminformatics.generate_conformer(cheminformatics.make_mol(smi), smi, seed=7)
        mol_b = cheminformatics.generate_conformer(cheminformatics.make_mol(smi), smi, seed=7)
        _, coords_a = cheminformatics.get_coords_conformer(mol_a)
        _, coords_b = cheminformatics.get_coords_conformer(mol_b)
        np.testing.assert_allclose(coords_a, coords_b, atol=1e-6)
