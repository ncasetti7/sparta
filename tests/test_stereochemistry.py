"""Tests for coordinate-level stereochemistry helpers."""

import numpy as np
from rdkit import Chem

from sparta import Reaction
from sparta import cheminformatics
from sparta.stereochemistry import (
    alkene_stereo_matches,
    alkene_stereo_references,
    atropisomer_stereo_matches,
    atropisomer_stereo_mismatches,
    atropisomer_stereo_references,
    correct_alkene_stereochemistry,
    correct_atropisomer_stereochemistry,
    correct_tetrahedral_stereochemistry,
    reflect_atoms_across_plane,
    rotate_atoms_around_axis,
    stereo_matches,
    tetrahedral_stereo_references,
)
from sparta.stereochemistry.constraints import (
    _movable_atropisomer_substituent,
    atropisomer_axis_value,
)


def _chiral_mol():
    smi = "[C@:1]([F:2])([Cl:3])([Br:4])[H:5]"
    mol = cheminformatics.make_mol(smi)
    return cheminformatics.generate_conformer(mol, smi, seed=0)


def _alkene_mol(smiles="[F:1]/[C:2]([H:6])=[C:3](/[Cl:4])[H:5]"):
    mol = cheminformatics.make_mol(smiles)
    return cheminformatics.generate_conformer(mol, smiles, seed=0)


def _atropisomer_smiles(stereo=Chem.BondStereo.STEREOATROPCW):
    mol = Chem.MolFromSmiles("Clc1ccc(Cl)c(-c2cc(Cl)ccc2Cl)c1")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    for bond in mol.GetBonds():
        if (
            bond.GetBondType() == Chem.BondType.SINGLE
            and bond.GetBeginAtom().GetIsAromatic()
            and bond.GetEndAtom().GetIsAromatic()
        ):
            bond.SetStereo(stereo)
            break
    return Chem.MolToCXSmiles(mol)


def _atropisomer_mol(stereo=Chem.BondStereo.STEREOATROPCW):
    smiles = _atropisomer_smiles(stereo)
    mol = cheminformatics.make_mol(smiles)
    return cheminformatics.generate_conformer(mol, smiles, seed=0)


class TestTetrahedralStereoReferences:
    def test_extracts_label_and_signed_volume(self):
        mol = _chiral_mol()
        refs = tetrahedral_stereo_references(mol)

        assert len(refs) == 1
        assert refs[0].center == 0
        assert refs[0].label in {"R", "S"}
        assert abs(refs[0].volume) > 1e-6

    def test_volume_neighbors_follow_cip_priority(self):
        mol = _chiral_mol()
        ref = tetrahedral_stereo_references(mol)[0]

        assert ref.priority_neighbors == (3, 2, 1, 4)
        assert ref.neighbors == (3, 2, 1)
        assert ref.priority_ranks == tuple(sorted(ref.priority_ranks, reverse=True))

    def test_reaction_builds_separate_reactant_product_references(self):
        reactant = "[C@:1]([F:2])([Cl:3])([Br:4])[H:5]"
        product = "[C@@:1]([F:2])([Cl:3])([Br:4])[H:5]"

        rxn = Reaction(
            reactant_smiles=reactant,
            product_smiles=product,
            n_conformers=1,
            n_seeds=1,
            n_images=3,
        )

        assert rxn.reactant_stereo_refs[0].label == "R"
        assert rxn.product_stereo_refs[0].label == "S"
        assert rxn.reactant_stereo_refs[0].priority_neighbors == (3, 2, 1, 4)
        assert rxn.product_stereo_refs[0].priority_neighbors == (3, 2, 1, 4)
        assert rxn.reactant_stereo_refs[0].sign != rxn.product_stereo_refs[0].sign

    def test_atom_maps_do_not_change_target_cip_priority(self):
        smi = (
            "[O:39]=[C:40]([O:41][C:42]([H:90])([H:91])[H:92])"
            "[C:43](=[O:44])[C@:48]([C:47]([C:46](=[C:45]"
            "([H:93])[H:94])[H:95])([H:96])[H:97])"
            "([C:49]([H:99])([H:100])[H:101])[H:98]"
        )
        mol = cheminformatics.make_mol(smi)
        mol = cheminformatics.generate_conformer(mol, smi, seed=0)

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        mapped_label = mol.GetAtomWithIdx(9).GetProp("_CIPCode")
        refs = tetrahedral_stereo_references(mol)

        assert mapped_label == "R"
        assert len(refs) == 1
        assert refs[0].center == 9
        assert refs[0].label == "S"

    def test_correction_restores_wrong_signed_volume(self):
        mol = _chiral_mol()
        refs = tetrahedral_stereo_references(mol)
        ref = refs[0]
        coords = mol.GetConformer().GetPositions()

        moving = ref.neighbors[0]
        fixed = ref.neighbors[1:]
        wrong = reflect_atoms_across_plane(coords, [moving], (ref.center, fixed[0], fixed[1]))
        assert not stereo_matches(wrong, refs)

        corrected = correct_tetrahedral_stereochemistry(wrong, mol, refs)
        assert stereo_matches(corrected, refs)

    def test_movable_branch_never_carries_another_stereocenter(self):
        """Reflection is improper, so the tetrahedral corrector must not select a
        branch containing a second stereocenter (which it would invert)."""
        from sparta.stereochemistry.constraints import (
            _movable_substituent,
            _branch_has_internal_stereo,
        )

        smi = (
            "[C:1]([H:5])([H:6])([H:7])[C@:2]([Br:3])([H:8])"
            "[C@:9]([Br:10])([H:11])[C:12]([H:13])([H:14])[H:15]"
        )
        mol = cheminformatics.make_mol(smi)
        mol = cheminformatics.generate_conformer(mol, smi, seed=0)
        refs = tetrahedral_stereo_references(mol)

        for ref in refs:
            movable = _movable_substituent(mol, ref)
            if movable is not None:
                _, branch = movable
                assert not _branch_has_internal_stereo(mol, branch)


class TestAlkeneStereoReferences:
    def test_extracts_e_reference_and_side_descriptor(self):
        mol = _alkene_mol()
        refs = alkene_stereo_references(mol)

        assert len(refs) == 1
        assert refs[0].bond == (1, 2)
        assert refs[0].substituents == (0, 3)
        assert refs[0].label == "E"
        assert refs[0].sign == -1.0
        assert abs(refs[0].value) > 0.9

    def test_reaction_builds_separate_reactant_product_alkene_references(self):
        reactant = "[F:1]/[C:2]([H:6])=[C:3](/[Cl:4])[H:5]"
        product = "[F:1]/[C:2]([H:6])=[C:3](\\[Cl:4])[H:5]"

        rxn = Reaction(
            reactant_smiles=reactant,
            product_smiles=product,
            n_conformers=1,
            n_seeds=1,
            n_images=3,
        )

        assert rxn.reactant_alkene_refs[0].label == "E"
        assert rxn.product_alkene_refs[0].label == "Z"
        assert rxn.reactant_alkene_refs[0].sign != rxn.product_alkene_refs[0].sign

    def test_correction_restores_wrong_ez_side(self):
        mol = _alkene_mol()
        refs = alkene_stereo_references(mol)
        ref = refs[0]
        coords = mol.GetConformer().GetPositions()

        wrong = rotate_atoms_around_axis(
            coords,
            [ref.substituents[0]],
            ref.bond,
            np.pi,
        )
        assert not alkene_stereo_matches(wrong, refs)

        corrected = correct_alkene_stereochemistry(wrong, mol, refs)
        assert alkene_stereo_matches(corrected, refs)

    def test_ez_correction_uses_half_turn_rotation(self):
        mol = _alkene_mol()
        refs = alkene_stereo_references(mol)
        ref = refs[0]
        coords = mol.GetConformer().GetPositions()

        wrong = rotate_atoms_around_axis(coords, [ref.substituents[0]], ref.bond, np.pi)
        corrected = correct_alkene_stereochemistry(wrong, mol, refs)
        expected = rotate_atoms_around_axis(wrong, [ref.substituents[0]], ref.bond, np.pi)

        np.testing.assert_allclose(corrected, expected, atol=1e-8)

    def test_product_generation_repairs_product_ez_reference(self):
        reactant = "[F:1]/[C:2]([H:6])=[C:3](/[Cl:4])[H:5]"
        product = "[F:1]/[C:2]([H:6])=[C:3](\\[Cl:4])[H:5]"

        rxn = Reaction(
            reactant_smiles=reactant,
            product_smiles=product,
            n_conformers=2,
            n_seeds=1,
            n_images=3,
            top_k=2,
        ).generate_initial_conformers()

        assert rxn.top_k_conformers
        assert all(
            alkene_stereo_matches(c["product_coords"], rxn.product_alkene_refs)
            for c in rxn.top_k_conformers
        )


class TestAtropisomerStereoReferences:
    def test_make_mol_preserves_atropisomer_cxsmiles(self):
        smiles = _atropisomer_smiles()
        mol = cheminformatics.make_mol(smiles)

        atrop_bonds = [
            bond
            for bond in mol.GetBonds()
            if bond.GetStereo()
            in (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW)
        ]

        assert len(atrop_bonds) == 1

    def test_extracts_atropisomer_reference_and_descriptor(self):
        mol = _atropisomer_mol()
        refs = atropisomer_stereo_references(mol)

        assert len(refs) == 1
        assert refs[0].label == "CW"
        assert refs[0].axis == (6, 7)
        assert len(refs[0].controlling_atoms) == 4
        assert abs(refs[0].value) > 0.5

    def test_opposite_atropisomer_labels_generate_opposite_axis_signs(self):
        cw = _atropisomer_mol(Chem.BondStereo.STEREOATROPCW)
        ccw = _atropisomer_mol(Chem.BondStereo.STEREOATROPCCW)

        cw_ref = atropisomer_stereo_references(cw)[0]
        ccw_ref = atropisomer_stereo_references(ccw)[0]

        assert cw_ref.label == "CW"
        assert ccw_ref.label == "CCW"
        assert cw_ref.sign != ccw_ref.sign

    def test_atropisomer_mismatch_detects_axis_sign_change(self):
        mol = _atropisomer_mol()
        refs = atropisomer_stereo_references(mol)
        ref = refs[0]
        coords = mol.GetConformer().GetPositions()

        wrong = rotate_atoms_around_axis(coords, [ref.descriptor_atoms[0]], ref.axis, np.pi)

        assert not atropisomer_stereo_matches(wrong, refs)


class TestAtropisomerCorrection:
    def test_reflection_lands_on_mirror_and_preserves_branch_geometry(self):
        """The reflection operator flips the axial sign while keeping the moved
        branch rigid (an isometry) — the minimal-strain route to the mirror
        isomer, unlike the pi rotation which lands on a different rotamer."""
        mol = _atropisomer_mol()
        ref = atropisomer_stereo_references(mol)[0]
        coords = mol.GetConformer().GetPositions()
        branch, plane = _movable_atropisomer_substituent(mol, ref)

        a, b = ref.axis
        i, j = ref.descriptor_atoms
        before = atropisomer_axis_value(coords, a, b, i, j)
        reflected = reflect_atoms_across_plane(coords, branch, plane)
        after = atropisomer_axis_value(reflected, a, b, i, j)

        # descriptor sign flips, magnitude preserved (theta -> -theta)
        assert np.sign(before) != np.sign(after)
        assert abs(abs(after) - abs(before)) < 1e-6

        # branch internal geometry is untouched by the reflection
        bl = sorted(branch)
        d0 = np.linalg.norm(coords[bl][:, None] - coords[bl][None, :], axis=-1)
        d1 = np.linalg.norm(reflected[bl][:, None] - reflected[bl][None, :], axis=-1)
        assert np.abs(d0 - d1).max() < 1e-8

    def test_correction_restores_wrong_atropisomer(self):
        mol = _atropisomer_mol()
        refs = atropisomer_stereo_references(mol)
        ref = refs[0]
        coords = mol.GetConformer().GetPositions()

        branch, _ = _movable_atropisomer_substituent(mol, ref)
        wrong = rotate_atoms_around_axis(coords, branch, ref.axis, np.pi)
        assert not atropisomer_stereo_matches(wrong, refs)

        corrected = correct_atropisomer_stereochemistry(wrong, mol, refs)
        assert atropisomer_stereo_matches(corrected, refs)

    def test_correction_preserves_a_tetrahedral_center_in_the_moved_branch(self):
        """When the moved branch carries its own stereocenter, the corrector must
        use the proper (pi-rotation) operator so that center is not inverted."""
        # biaryl-style axis (ring C4-C5 to a chiral carbon) with a tetrahedral
        # center living on the branch that would be moved.
        smi = (
            "[C:1]1([H:9])[C:2]([H:10])=[C:3]([H:11])[C:4]([H:12])"
            "=[C:5]1[C@:6]([F:7])([Cl:8])[Br:13]"
        )
        mol = cheminformatics.make_mol(smi)
        mol = cheminformatics.generate_conformer(mol, smi, seed=0)
        tet_refs = tetrahedral_stereo_references(mol)
        coords = mol.GetConformer().GetPositions()

        # reflecting the {stereocenter + substituents} branch would invert it;
        # a pi rotation about the axis preserves it.
        branch = [5, 6, 7, 12]
        rotated = rotate_atoms_around_axis(coords, branch, (4, 5), np.pi)
        assert stereo_matches(rotated, tet_refs)

    def test_product_generation_repairs_product_atropisomer(self):
        """A reaction whose product is the opposite atropisomer: the pushforward
        must flip the axis and the corrector must recover it, instead of the
        pairs being rejected."""
        reactant = _atropisomer_smiles(Chem.BondStereo.STEREOATROPCW)
        product = _atropisomer_smiles(Chem.BondStereo.STEREOATROPCCW)

        rxn = Reaction(
            reactant_smiles=reactant,
            product_smiles=product,
            n_conformers=4,
            n_seeds=1,
            n_images=3,
            top_k=4,
        ).generate_initial_conformers()

        assert rxn.top_k_conformers
        assert rxn.reactant_atropisomer_refs[0].sign != rxn.product_atropisomer_refs[0].sign
        assert all(
            atropisomer_stereo_matches(c["product_coords"], rxn.product_atropisomer_refs)
            for c in rxn.top_k_conformers
        )
        assert all(
            atropisomer_stereo_matches(c["reactant_coords"], rxn.reactant_atropisomer_refs)
            for c in rxn.top_k_conformers
        )


class TestStereoDebug:
    def test_reaction_is_quiet_by_default(self, capsys, monkeypatch):
        monkeypatch.delenv("SPARTA_STEREO_DEBUG", raising=False)
        smi = "[C@:1]([F:2])([Cl:3])([Br:4])[H:5]"

        Reaction(reactant_smiles=smi, product_smiles=smi)

        assert capsys.readouterr().out == ""

    def test_reaction_logs_when_enabled(self, capsys, monkeypatch):
        monkeypatch.delenv("SPARTA_STEREO_DEBUG", raising=False)
        smi = "[C@:1]([F:2])([Cl:3])([Br:4])[H:5]"

        Reaction(reactant_smiles=smi, product_smiles=smi, stereo_debug=True)

        out = capsys.readouterr().out
        assert "[SPARTA stereo] reactant reference" in out
        assert "[SPARTA stereo] product reference" in out
