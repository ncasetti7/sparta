"""Shared fixtures for the sparta test suite."""

import numpy as np
import pytest
import ase

from sparta import Reaction
from sparta.cheminformatics import make_mol

# ---------------------------------------------------------------------------
# Reaction SMILES used throughout the test suite.
# F2 + Cl2 → 2 FCl (halogen exchange).
# 4 heavy atoms, no implicit H, all mapped, embeds reliably.
#   atom 0 = F (map 1), atom 1 = F (map 2)
#   atom 2 = Cl (map 3), atom 3 = Cl (map 4)
# Reactant bonds: (0,1) F-F, (2,3) Cl-Cl
# Product  bonds: (0,2) F-Cl, (1,3) F-Cl
# Changing bonds: {(0,1), (2,3), (0,2), (1,3)}
# ---------------------------------------------------------------------------
REACTANT_SMILES = "[F:1][F:2].[Cl:3][Cl:4]"
PRODUCT_SMILES = "[F:1][Cl:3].[F:2][Cl:4]"


@pytest.fixture(scope="session")
def rxn_smiles():
    return REACTANT_SMILES, PRODUCT_SMILES


@pytest.fixture(scope="session")
def simple_reaction():
    """Minimal Reaction object — __init__ only, no conformers generated."""
    return Reaction(
        reactant_smiles=REACTANT_SMILES,
        product_smiles=PRODUCT_SMILES,
        n_conformers=2,
        n_seeds=1,
        stretch_factors=1.5,
        top_k=2,
        n_images=5,
    )


@pytest.fixture(scope="session")
def fast_reaction():
    """Reaction with settings chosen for the fastest possible pipeline run."""
    return Reaction(
        reactant_smiles=REACTANT_SMILES,
        product_smiles=PRODUCT_SMILES,
        n_conformers=1,
        n_seeds=1,
        stretch_factors=1.5,
        top_k=1,
        n_images=3,
    )


@pytest.fixture
def four_atom_coords():
    """Linear arrangement of 4 atoms spaced 1.5 Å apart."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])


@pytest.fixture
def chain_adj():
    """Adjacency matrix for the chain 0–1–2–3."""
    a = np.zeros((4, 4), dtype=int)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        a[i, j] = a[j, i] = 1
    return a


@pytest.fixture
def two_atom_ase_images():
    """5 linearly interpolated ASE images for a 2-atom (F-F) system."""
    initial = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    final = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
    n = 5
    images = []
    for k in range(n):
        alpha = k / (n - 1)
        pos = (1 - alpha) * initial + alpha * final
        images.append(ase.Atoms(symbols=["F", "F"], positions=pos))
    return images
