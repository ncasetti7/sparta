"""Tests for coordinate-level stereochemistry helpers."""

from rdkit import Chem

from sparta import Reaction
from sparta import cheminformatics
from sparta.stereochemistry import (
    correct_tetrahedral_stereochemistry,
    reflect_atoms_across_plane,
    stereo_matches,
    tetrahedral_stereo_references,
)


def _chiral_mol():
    smi = "[C@:1]([F:2])([Cl:3])([Br:4])[H:5]"
    mol = cheminformatics.make_mol(smi)
    return cheminformatics.generate_conformer(mol, smi, seed=0)


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
