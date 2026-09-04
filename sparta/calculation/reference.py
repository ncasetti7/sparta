"""Reference-state energies for reactant and product.

A reaction's barrier is reported relative to the reactant reference energy, so
that reference has to be a well-defined state. Optimising a multi-fragment
reactant as a single ``ase.Atoms`` object does not give one: the fragments
start at whatever separation the distance-geometry embedding happened to
produce (``embedFragmentsSeparately`` is False when the conformers are
embedded), and the optimiser then relaxes into some arbitrary loosely-bound
complex. The resulting energy carries an arbitrary inter-fragment interaction,
which propagates straight into every barrier computed against it.

Measured on four datasets at matched DFT barrier, the co-embedded reference
inflated SPARTA barriers by roughly 12 kcal/mol per additional reactant
fragment (RGD1 medians 80.7 / 93.9 / 105.3 kcal/mol for 1 / 2 / 3 fragments
while the DFT medians stayed at 23.4 / 23.6 / 22.5). That offset pushed
genuinely low-barrier multi-fragment reactions above the pruning threshold.

The fix is the standard separated-reactant reference: optimise each connected
fragment on its own and sum the energies. For a single-fragment species this
is exactly the previous behaviour.
"""
from typing import Optional, Sequence, Tuple

import numpy as np


def fragment_reference_energy(
    mol,
    coords: Sequence[Sequence[float]],
    calculator,
) -> Tuple[np.ndarray, Optional[float]]:
    """Optimise every connected fragment of ``mol`` separately and sum energies.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule (or multi-fragment system) whose atom order matches ``coords``.
    coords : array-like, shape (n_atoms, 3)
        Starting geometry, e.g. ``Reaction.reac_ref_coords``.
    calculator : Calculation
        Anything exposing ``optimize(atoms) -> (atoms, energy)``. If it carries
        ``charge`` / ``multiplicity`` attributes they are set per fragment from
        that fragment's formal charges and radical electrons, then restored.

    Returns
    -------
    (positions, total_energy)
        ``positions`` holds each fragment relaxed in isolation, written back in
        the original atom order. ``total_energy`` is the sum over fragments, or
        None if every fragment failed to evaluate.
    """
    import ase
    from rdkit import Chem

    positions = np.asarray(coords, dtype=float).copy()
    frags = Chem.GetMolFrags(mol)

    saved_charge = getattr(calculator, "charge", None)
    saved_mult = getattr(calculator, "multiplicity", None)

    total: Optional[float] = None
    try:
        for idx in frags:
            idx_list = list(idx)
            symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in idx_list]
            atoms = ase.Atoms(symbols=symbols, positions=positions[idx_list])

            if hasattr(calculator, "charge"):
                calculator.charge = int(
                    sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in idx_list)
                )
            if hasattr(calculator, "multiplicity"):
                calculator.multiplicity = 1 + int(
                    sum(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in idx_list)
                )

            try:
                opt, energy = calculator.optimize(atoms)
                positions[idx_list] = opt.get_positions()
            except Exception:
                # fall back to a single point on the unrelaxed fragment so one
                # bad fragment does not discard the whole reference
                try:
                    energy = calculator.energy(atoms)
                except Exception:
                    continue
            total = float(energy) if total is None else total + float(energy)
    finally:
        if saved_charge is not None and hasattr(calculator, "charge"):
            calculator.charge = saved_charge
        if saved_mult is not None and hasattr(calculator, "multiplicity"):
            calculator.multiplicity = saved_mult

    return positions, total
