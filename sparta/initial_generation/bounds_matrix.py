"""
ETKDG-based bounds matrix modification and distance geometry conformer generation.

For each pair of reactant/product conformers:
1. Modify the RDKit bounds matrix to incorporate stretched bond targets for
   bonds that change between reactant and product.
2. Embed multiple conformers using EmbedMultipleConfs with the modified matrix.
3. For each reactant conformer, transform it to approximate product geometry
   (via force_propagation.transform_mol) and then relax it to satisfy the
   product bounds matrix (via dg_optimizer).
"""

from copy import copy
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from sparta.cheminformatics import add_more_coords_to_mol
from sparta.initial_generation.dg_optimizer import optimize_to_bounds_matrix_and_rmsd
from sparta.iterative_alignment.force_propagation import transform_mol

if TYPE_CHECKING:
    import ase
    from sparta.reaction import Reaction
    from sparta.calculation.base import Calculation


def _generate_reaction_conformers_single(
    reaction: "Reaction",
    stretch_factor: float,
    seed: Optional[int],
) -> Tuple[Chem.Mol, Chem.Mol, np.ndarray, np.ndarray]:
    """
    Generate reactant/product conformer pairs for a single (stretch_factor, seed)
    combination using modified bounds matrices.

    The base bounds matrices are pulled from ``reaction.reactant_base_bm`` and
    ``reaction.product_base_bm`` (computed once in ``Reaction.__init__``) and
    modified in-place for the given stretch_factor — no recalculation needed.

    Parameters
    ----------
    reaction : Reaction
        Provides molecules, changing bonds, reference bond lengths, base bounds
        matrices, and ``n_conformers``.
    stretch_factor : float
        Multiplicative factor applied to target bond lengths.
    seed : int or None
        Random seed for embedding reproducibility.

    Returns
    -------
    reactant_mol : Chem.Mol
        Reactant mol with up to ``reaction.n_conformers`` conformers embedded.
    product_mol : Chem.Mol
        Product mol with one conformer per successfully paired reactant conformer.
    reactant_bounds_matrix : np.ndarray
        Modified bounds matrix used for reactant embedding.
    product_bounds_matrix : np.ndarray
        Modified bounds matrix used for product embedding.
    """
    reactant_mol = copy(reaction.reactant_mol)
    product_mol = copy(reaction.product_mol)
    changing_bonds = reaction.changing_bonds
    reac_bond_lengths = reaction.reac_bond_lengths
    prod_bond_lengths = reaction.prod_bond_lengths
    n_conformers = reaction.n_conformers

    reactant_bonds = {
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in reactant_mol.GetBonds()
    }
    product_bonds = {
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in product_mol.GetBonds()
    }

    # Bonds present in product but absent in reactant (forming bonds in reactant side)
    reactant_diff_bonds = [b for b in changing_bonds if b not in reactant_bonds]
    # Bonds present in reactant but absent in product (breaking bonds in product side)
    product_diff_bonds = [b for b in changing_bonds if b not in product_bonds]

    # Copy the pre-computed base bounds matrices and modify for this stretch factor
    reactant_bm = reaction.reactant_base_bm.copy()
    product_bm = reaction.product_base_bm.copy()

    for bond in reactant_diff_bonds:
        atom1, atom2 = bond
        target = prod_bond_lengths[bond] * stretch_factor
        reactant_bm[atom2, atom1] = min(target, reactant_bm[atom2, atom1])
        reactant_bm[atom1, atom2] = min(target + 0.2, reactant_bm[atom1, atom2])

    for bond in product_diff_bonds:
        atom1, atom2 = bond
        target = reac_bond_lengths[bond] * stretch_factor
        product_bm[atom2, atom1] = min(target, product_bm[atom2, atom1])
        product_bm[atom1, atom2] = min(target + 0.2, product_bm[atom1, atom2])

    # Embedding parameters shared by both reactant and product
    embed_params = rdDistGeom.EmbedParameters()
    embed_params.embedFragmentsSeparately = False
    embed_params.maxIterations = 1000
    embed_params.useRandomCoords = True
    embed_params.pruneRmsThresh = 0.01
    if seed is not None:
        embed_params.randomSeed = seed + 1

    # Embed reactant conformers
    embed_params.SetBoundsMat(reactant_bm)
    reactant_mol.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(reactant_mol, n_conformers, embed_params)
    if reactant_mol.GetNumConformers() == 0:
        raise ValueError("Could not embed reactant molecule with the given constraints.")

    reac_conf_ids = [conf.GetId() for conf in reactant_mol.GetConformers()]

    # Embed a single reference product conformer for coordinate transformation
    product_mol.RemoveAllConformers()
    temp_prod = copy(product_mol)
    embed_params.SetBoundsMat(product_bm)
    embed_params.trackFailures = True
    result = AllChem.EmbedMolecule(temp_prod, embed_params)
    if result != 0:
        raise ValueError("Could not embed product molecule with the given constraints.")

    # For each reactant conformer, derive a paired product conformer
    used_reac_conf_ids = []
    for reac_conf_id in reac_conf_ids:
        reactant_coords = reactant_mol.GetConformer(reac_conf_id).GetPositions()
        transformed_coords = transform_mol(reactant_coords, temp_prod, changing_bonds)
        final_product_coords = optimize_to_bounds_matrix_and_rmsd(
            transformed_coords,
            product_bm,
            reactant_coords,
            method="L-BFGS-B",
            max_iter=500,
            weight_rmsd=0.05,
            verbose=False,
        )
        product_mol = add_more_coords_to_mol(final_product_coords, product_mol)
        used_reac_conf_ids.append(reac_conf_id)
        if len(product_mol.GetConformers()) >= n_conformers:
            break

    if len(product_mol.GetConformers()) == 0:
        raise ValueError("Could not generate any valid product conformers.")

    # Remove reactant conformers that did not produce a paired product conformer
    for conf_id in reac_conf_ids:
        if conf_id not in used_reac_conf_ids:
            reactant_mol.RemoveConformer(conf_id)

    return reactant_mol, product_mol, reactant_bm, product_bm


# ---------------------------------------------------------------------------
# Public high-level entry point
# ---------------------------------------------------------------------------

def generate_initial_conformers(
    reaction: "Reaction",
    n_workers: int = 1,
    calculator: "Optional[Calculation]" = None,
) -> "Reaction":
    """
    Generate reactant/product conformer pairs for every (seed, stretch_factor)
    combination, score each pair with a quick NEB, and populate
    ``reaction.all_conformers`` (sorted by score) and ``reaction.top_k_conformers``
    (the best ``reaction.top_k`` pairs).

    Failed embedding attempts are silently skipped.

    Parameters
    ----------
    reaction : Reaction
        A fully initialised :class:`~sparta.reaction.Reaction` object.
    n_workers : int
        Number of parallel worker processes for conformer scoring.  Set > 1
        to use ``concurrent.futures.ProcessPoolExecutor``.
    calculator : Calculation, optional
        If provided, optimise the seed-0 UFF reference conformers stored in
        ``reaction.reac_ref_coords`` and ``reaction.prod_ref_coords`` and store
        the results in ``reaction.reac_ref_optimized``, ``reaction.prod_ref_optimized``,
        ``reaction.reac_ref_energy``, and ``reaction.prod_ref_energy``.

    Returns
    -------
    Reaction
        The same object with ``all_conformers`` and ``top_k_conformers`` populated.
    """
    from sparta import cheminformatics
    from sparta.iterative_alignment.optimization import _score_worker

    all_conformers = []

    for seed in range(reaction.n_seeds):
        for stretch_factor in reaction.stretch_factors:
            try:
                reac_mol, prod_mol, reac_bm, prod_bm = _generate_reaction_conformers_single(
                    reaction, stretch_factor=stretch_factor, seed=seed
                )
            except ValueError:
                continue

            all_reac = cheminformatics.get_all_coords_all_conformers(reac_mol)
            all_prod = cheminformatics.get_all_coords_all_conformers(prod_mol)

            for (_, reac_coords), (_, prod_coords) in zip(all_reac, all_prod):
                all_conformers.append({
                    "reactant_coords": reac_coords,
                    "product_coords": prod_coords,
                    "reactant_bounds_matrix": reac_bm,
                    "product_bounds_matrix": prod_bm,
                    "stretch_factor": stretch_factor,
                    "seed": seed,
                    "score": None,
                })

    symbols = [reaction.reactant_mol.GetAtomWithIdx(i).GetSymbol()
               for i in range(reaction.reactant_mol.GetNumAtoms())]

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        args_list = [
            (symbols, reaction.n_images, c["reactant_coords"], c["product_coords"])
            for c in all_conformers
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            scores = list(executor.map(_score_worker, args_list))
        for conformer, score in zip(all_conformers, scores):
            conformer["score"] = score
    else:
        for conformer in all_conformers:
            try:
                conformer["score"] = _score_worker((
                    symbols, reaction.n_images,
                    conformer["reactant_coords"], conformer["product_coords"],
                ))
            except Exception:
                conformer["score"] = float("inf")

    valid = [c for c in all_conformers if c["score"] != float("inf")]
    valid.sort(key=lambda c: c["score"])

    reaction.all_conformers = valid
    reaction.top_k_conformers = valid[: reaction.top_k]

    if calculator is not None:
        # Separated-reactant reference: each connected fragment is optimised on
        # its own and the energies summed. Optimising a multi-fragment system as
        # one unit would leave an arbitrary inter-fragment interaction in the
        # reference energy (the fragments are co-embedded at an arbitrary
        # separation), which offsets every barrier measured against it -- see
        # sparta.calculation.reference for the measured magnitude. Identical to
        # the previous behaviour for single-fragment species.
        from sparta.calculation.reference import fragment_reference_energy

        reac_pos, reac_energy = fragment_reference_energy(
            reaction.reactant_mol, reaction.reac_ref_coords, calculator
        )
        prod_pos, prod_energy = fragment_reference_energy(
            reaction.product_mol, reaction.prod_ref_coords, calculator
        )

        reaction.reac_ref_optimized = reac_pos
        reaction.prod_ref_optimized = prod_pos
        reaction.reac_ref_energy = reac_energy
        reaction.prod_ref_energy = prod_energy

    return reaction
