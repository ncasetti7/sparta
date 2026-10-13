"""
Iterative IDPP path optimization with force propagation to endpoints.

Functions
---------
score_conformer_pair
    Quick NEB used to rank initial conformers.
run_iterative_alignment
    Main public function: loop over all top-k conformers, run iterative
    NEB + endpoint force propagation for each, store ranked results on the
    Reaction object.
_run_alignment_single
    Single-conformer iterative alignment (called by run_iterative_alignment).
"""

import contextlib
import os
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import numpy as np

from sparta import cheminformatics
from sparta.iterative_alignment.force_propagation import (
    _pairwise_distances,
    BOND_BREAK_TOLERANCE,
    apply_endpoint_adjustment,
    bound_violation,
    compute_endpoint_gradient_barrier,
    compute_endpoint_gradient_total,
)
from sparta.iterative_alignment.idpp import IDPP, create_idpp_calculators

if TYPE_CHECKING:
    from sparta.calculation.base import Calculation
    from sparta.reaction import Reaction


# ---------------------------------------------------------------------------
# Bounds matrix helpers
# ---------------------------------------------------------------------------

def _extract_symmetric_bounds(bounds_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an RDKit asymmetric bounds matrix (lower in lower triangle, upper in
    upper triangle) to two symmetric (n, n) lower-bound and upper-bound matrices.
    """
    n = bounds_matrix.shape[0]
    lower = np.zeros((n, n))
    upper = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                lower[i, j] = bounds_matrix[i, j]
                lower[j, i] = bounds_matrix[i, j]
            elif i < j:
                upper[i, j] = bounds_matrix[i, j]
                upper[j, i] = bounds_matrix[i, j]
    return lower, upper


def _relax_bounds_to_structure(
    lower: np.ndarray, upper: np.ndarray, positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relax any lower or upper bound the current structure already violates to the
    actual distance, so the structure is not immediately penalised during optimisation.
    """
    lower = lower.copy()
    upper = upper.copy()
    n = positions.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[j] - positions[i])
            if d < lower[i, j]:
                lower[i, j] = d
                lower[j, i] = d
            if upper[i, j] > 0 and d > upper[i, j]:
                upper[i, j] = d
                upper[j, i] = d
    return lower, upper


# ---------------------------------------------------------------------------
# NEB path optimisation (fixed endpoints)
# ---------------------------------------------------------------------------

def _build_neb_images(reactant_positions, product_positions, symbols, n_images):
    """Linearly interpolate between endpoints to create NEB images."""
    import ase
    initial = ase.Atoms(symbols=symbols, positions=reactant_positions)
    final = ase.Atoms(symbols=symbols, positions=product_positions)
    images = [initial]
    for k in range(1, n_images - 1):
        alpha = k / (n_images - 1)
        pos = (1.0 - alpha) * reactant_positions + alpha * product_positions
        img = initial.copy()
        img.positions = pos
        images.append(img)
    images.append(final)
    return images


def _build_neb_images_warm(reactant_positions, product_positions, current_images, symbols):
    """
    Build NEB images with updated endpoints but interior positions warm-started
    from current_images. Avoids re-interpolating from scratch after a small
    endpoint perturbation, so FIRE converges in fewer steps.
    """
    import ase
    n_images = len(current_images)
    initial = ase.Atoms(symbols=symbols, positions=reactant_positions)
    final = ase.Atoms(symbols=symbols, positions=product_positions)
    new_images = [initial]
    for k in range(1, n_images - 1):
        img = ase.Atoms(symbols=symbols, positions=current_images[k].get_positions())
        new_images.append(img)
    new_images.append(final)
    return new_images


def _run_neb(images, n_images, fmax, steps, verbose):
    """Attach IDPP calculators and run FIRE NEB. Returns (images, energies, forces)."""
    from ase.mep import NEB
    from ase.optimize import FIRE

    initial = images[0]
    final = images[-1]
    calcs = create_idpp_calculators(initial, final, n_images)
    for k, calc in enumerate(calcs):
        images[k + 1].calc = calc
    # Attach endpoint calculators required by the improvedtangent NEB method
    initial_pos = initial.get_positions()
    final_pos = final.get_positions()
    images[0].calc = IDPP(initial_pos, final_pos, alpha=0.0)
    images[-1].calc = IDPP(initial_pos, final_pos, alpha=1.0)

    neb = NEB(images, method='improvedtangent')
    if verbose:
        FIRE(neb).run(fmax=fmax, steps=steps)
    else:
        with open(os.devnull, "w", encoding="utf-8") as devnull, \
                contextlib.redirect_stdout(devnull):
            FIRE(neb).run(fmax=fmax, steps=steps)

    energies = [
        img.get_potential_energy() if img.calc is not None else None for img in images
    ]
    forces = [
        img.get_forces() if img.calc is not None else None for img in images
    ]
    return images, energies, forces


def optimize_neb_path(
    reaction: "Reaction",
    reactant_coords: np.ndarray,
    product_coords: np.ndarray,
    fmax: float = 0.05,
    steps: int = 100,
    verbose: bool = False,
) -> Tuple[list, List[Optional[float]], list]:
    """
    Run a single IDPP NEB optimisation between a reactant/product conformer pair.

    Parameters
    ----------
    reaction : Reaction
        Provides atomic symbols and the number of NEB images.
    reactant_coords, product_coords : np.ndarray, shape (n_atoms, 3)
    fmax, steps : float, int
        Convergence criterion and step limit.
    verbose : bool

    Returns
    -------
    (images, energies, forces)
    """
    symbols = [reaction.reactant_mol.GetAtomWithIdx(i).GetSymbol()
               for i in range(reaction.reactant_mol.GetNumAtoms())]
    images = _build_neb_images(reactant_coords, product_coords, symbols, reaction.n_images)
    return _run_neb(images, reaction.n_images, fmax, steps, verbose)


# ---------------------------------------------------------------------------
# Conformer scoring (quick NEB for ranking initial conformers)
# ---------------------------------------------------------------------------

def score_conformer_pair(
    reaction: "Reaction",
    reactant_coords: np.ndarray,
    product_coords: np.ndarray,
    fmax: float = 0.1,
    steps: int = 50,
) -> float:
    """
    Score a conformer pair with a quick NEB and return the max interior IDPP energy.
    Lower scores are better (easier path).

    Parameters
    ----------
    reaction : Reaction
        Provides atomic symbols and the number of NEB images.
    reactant_coords, product_coords : np.ndarray, shape (n_atoms, 3)
    fmax, steps : float, int
        Convergence and step limit for the quick NEB.
    """
    symbols = [reaction.reactant_mol.GetAtomWithIdx(i).GetSymbol()
               for i in range(reaction.reactant_mol.GetNumAtoms())]
    images = _build_neb_images(reactant_coords, product_coords, symbols, reaction.n_images)
    _, energies, _ = _run_neb(images, reaction.n_images, fmax, steps, verbose=False)
    interior = [e for e in energies[1:-1] if e is not None]
    return max(interior) if interior else float("inf")


# ---------------------------------------------------------------------------
# Module-level worker functions for multiprocessing
# ---------------------------------------------------------------------------

def _score_worker(args: tuple) -> float:
    """Picklable worker for parallel conformer scoring."""
    symbols, n_images, reactant_coords, product_coords = args
    images = _build_neb_images(reactant_coords, product_coords, symbols, n_images)
    _, energies, _ = _run_neb(images, n_images, 0.1, 50, verbose=False)
    interior = [e for e in energies[1:-1] if e is not None]
    return max(interior) if interior else float("inf")


def _alignment_worker(args: tuple) -> Optional[Dict]:
    """Picklable worker for parallel single-conformer alignment."""
    reaction, conformer, kwargs = args
    try:
        return _run_alignment_single(reaction, conformer, **kwargs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main iterative alignment
# ---------------------------------------------------------------------------

def _run_alignment_single(
    reaction: "Reaction",
    conformer: Dict,
    d_power: int = 4,
    neb_fmax: float = 0.001,
    neb_steps: int = 200,
    backtrack_fmax: Optional[float] = None,
    backtrack_steps: Optional[int] = None,
    propagation_threshold: float = 0.0001,
    propagation_step_size: float = 2.0,
    max_iterations: int = 40,
    objective: Literal["total", "barrier"] = "barrier",
    verbose: bool = False,
) -> Dict:
    """
    Iterative IDPP path optimisation with force propagation to endpoints.

    Alternates between:
      - Phase 2: Standard IDPP NEB with fixed endpoints.
      - Phase 3: Propagate IDPP energy gradient to endpoints (via
        :mod:`sparta.iterative_alignment.force_propagation`) and adjust their
        positions while respecting the bounds matrix.

    Continues until the propagated force magnitude falls below
    ``propagation_threshold`` or no step size produces an improvement.

    Parameters
    ----------
    reaction : Reaction
        Provides atomic symbols, number of NEB images, and other settings.
    conformer : dict
        A conformer dict from ``Reaction.top_k_conformers``, containing at
        minimum the keys ``'reactant_coords'``, ``'product_coords'``,
        ``'reactant_bounds_matrix'``, and ``'product_bounds_matrix'``.
    d_power : int
        IDPP distance weighting exponent.
    neb_fmax : float
        Force convergence for each inner NEB run.
    neb_steps : int
        Maximum steps for each inner NEB run.
    backtrack_fmax : float, optional
        Force convergence for cheap backtracking NEB evaluations.
        Defaults to ``neb_fmax * 10``.
    backtrack_steps : int, optional
        Maximum steps for cheap backtracking NEB evaluations.
        Defaults to ``max(neb_steps // 4, 20)``.
    propagation_threshold : float
        Stop when max propagated force magnitude < this value.
    propagation_step_size : float
        Initial endpoint adjustment step size (Å). Halved on rejection.
    max_iterations : int
        Maximum phase-2/3 iterations.
    objective : {'total', 'barrier'}
        'total'   — minimise sum of interior image energies.
        'barrier' — minimise max interior image energy.
    verbose : bool

    Returns
    -------
    dict with keys:
        'reactant_coords'  : np.ndarray — final reactant endpoint positions
        'product_coords'   : np.ndarray — final product endpoint positions
        'images'           : list of ase.Atoms — full optimised path
        'energies'         : list of float or None — per-image IDPP energies
        'score'            : float — max interior IDPP energy after optimisation
        'max_bond_violation' : float — largest violation (Å) of either endpoint's bonds,
                                     against the bounds relaxed to its starting structure
    """
    import ase

    if backtrack_fmax is None:
        backtrack_fmax = neb_fmax * 10.0
    if backtrack_steps is None:
        backtrack_steps = max(neb_steps // 4, 20)

    reactant_coords = conformer["reactant_coords"]
    product_coords = conformer["product_coords"]
    reactant_bounds_matrix = conformer["reactant_bounds_matrix"]
    product_bounds_matrix = conformer["product_bounds_matrix"]
    n_images = reaction.n_images

    symbols = [reaction.reactant_mol.GetAtomWithIdx(i).GetSymbol()
               for i in range(reaction.reactant_mol.GetNumAtoms())]

    # Extract symmetric bounds matrices and relax to current structures
    bl_init, bu_init = _extract_symmetric_bounds(reactant_bounds_matrix)
    bl_final, bu_final = _extract_symmetric_bounds(product_bounds_matrix)
    bl_init, bu_init = _relax_bounds_to_structure(bl_init, bu_init, reactant_coords)
    bl_final, bu_final = _relax_bounds_to_structure(bl_final, bu_final, product_coords)

    # Working ASE atoms for the endpoints (mutated during phase 3)
    reac_atoms = ase.Atoms(symbols=symbols, positions=reactant_coords.copy())
    prod_atoms = ase.Atoms(symbols=symbols, positions=product_coords.copy())

    reac_bonds = np.array(sorted(cheminformatics.get_bonds(reaction.reactant_mol)), dtype=int)
    prod_bonds = np.array(sorted(cheminformatics.get_bonds(reaction.product_mol)), dtype=int)

    # State tracking
    best_score = float("inf")
    best_reac_pos = reac_atoms.get_positions().copy()
    best_prod_pos = prod_atoms.get_positions().copy()
    best_images = None
    best_energies = None

    current_step = propagation_step_size
    min_step = propagation_step_size * 0.01

    # Endpoint distance matrices for the gradient functions. Invalidated after
    # each accepted endpoint adjustment so they are recomputed once on next use.
    d_initial_cache: Optional[np.ndarray] = None
    d_final_cache: Optional[np.ndarray] = None

    # Interior image positions from the last accepted backtracking NEB, used to
    # warm-start the next Phase 2 NEB instead of re-interpolating from scratch.
    warm_images: Optional[list] = None

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n=== Iteration {iteration + 1}: Phase 2 — NEB with fixed endpoints ===")

        # Phase 2: NEB with current endpoints.
        # Warm-start from the last accepted backtracking solution when available.
        if warm_images is not None:
            images = warm_images
            warm_images = None
        else:
            images = _build_neb_images(
                reac_atoms.get_positions(),
                prod_atoms.get_positions(),
                symbols, n_images,
            )
        images, energies, _ = _run_neb(images, n_images, neb_fmax, neb_steps, verbose)

        interior = [e for e in energies[1:-1] if e is not None]
        if objective == "total":
            current_score = sum(interior) if interior else float("inf")
        else:  # barrier
            current_score = max(interior) if interior else float("inf")

        if verbose:
            print(f"  Score ({objective}): {current_score:.6f}")

        if current_score < best_score - 1e-8:
            best_score = current_score
            best_reac_pos = reac_atoms.get_positions().copy()
            best_prod_pos = prod_atoms.get_positions().copy()
            best_images = [img.copy() for img in images]
            best_energies = list(energies)
            if verbose:
                print(f"  Improved! Best: {best_score:.6f}")

        # Phase 3: Compute propagated endpoint forces.
        # Reuse cached endpoint distance matrices when endpoints haven't changed.
        if d_initial_cache is None:
            d_initial_cache = _pairwise_distances(reac_atoms.get_positions())
            d_final_cache = _pairwise_distances(prod_atoms.get_positions())

        if objective == "total":
            init_force, final_force = compute_endpoint_gradient_total(
                images, d_power,
                d_initial=d_initial_cache, d_final=d_final_cache,
            )
        else:
            init_force, final_force, max_idx = compute_endpoint_gradient_barrier(
                images, d_power,
                d_initial=d_initial_cache, d_final=d_final_cache,
            )
            if verbose:
                print(f"  Gradient from image {max_idx}")

        max_prop = max(
            np.linalg.norm(init_force, axis=1).max(),
            np.linalg.norm(final_force, axis=1).max(),
        )
        if verbose:
            print(f"  Max propagated force: {max_prop:.6f}")

        if max_prop < propagation_threshold:
            if verbose:
                print(f"  Converged (force {max_prop:.6f} < {propagation_threshold}).")
            break

        # Try endpoint adjustment with backtracking.
        # Backtracking trials use a cheaper NEB (backtrack_fmax / backtrack_steps)
        # with warm-started interior positions, so each trial is much faster.
        # The full-quality NEB runs in the next Phase 2 after acceptance.
        accepted = False
        while current_step >= min_step:
            reac_pos_before = reac_atoms.get_positions().copy()
            prod_pos_before = prod_atoms.get_positions().copy()

            apply_endpoint_adjustment(
                reac_atoms, init_force, bl_init, bu_init,
                step_size=current_step, max_displacement=current_step,
            )
            apply_endpoint_adjustment(
                prod_atoms, final_force, bl_final, bu_final,
                step_size=current_step, max_displacement=current_step,
            )
            # reject steps that break an endpoint bond
            if (
                bound_violation(reac_atoms.get_positions(), bl_init, bu_init, reac_bonds)
                > BOND_BREAK_TOLERANCE
                or bound_violation(prod_atoms.get_positions(), bl_final, bu_final, prod_bonds)
                > BOND_BREAK_TOLERANCE
            ):
                if verbose:
                    print(f"  Rejected step {current_step:.4f}: endpoints leave their bounds")
                reac_atoms.set_positions(reac_pos_before)
                prod_atoms.set_positions(prod_pos_before)
                current_step *= 0.5
                continue

            # Cheap evaluation: warm-started interior + loose convergence
            test_images = _build_neb_images_warm(
                reac_atoms.get_positions(),
                prod_atoms.get_positions(),
                images, symbols,
            )
            test_images, test_energies, _ = _run_neb(
                test_images, n_images, backtrack_fmax, backtrack_steps, verbose=False,
            )
            test_interior = [e for e in test_energies[1:-1] if e is not None]
            if objective == "total":
                test_score = sum(test_interior) if test_interior else float("inf")
            else:
                test_score = max(test_interior) if test_interior else float("inf")

            if test_score < best_score - 1e-8:
                if verbose:
                    print(f"  Accepted step {current_step:.4f}: {test_score:.6f} < {best_score:.6f}")
                accepted = True
                current_step = propagation_step_size  # reset after acceptance
                # Warm-start next Phase 2 from this (cheap) NEB solution
                warm_images = [img.copy() for img in test_images]
                # Invalidate distance cache — endpoints have moved
                d_initial_cache = None
                d_final_cache = None
                break
            else:
                if verbose:
                    print(f"  Rejected step {current_step:.4f}: {test_score:.6f} >= {best_score:.6f}")
                reac_atoms.set_positions(reac_pos_before)
                prod_atoms.set_positions(prod_pos_before)
                current_step *= 0.5

        if not accepted:
            if verbose:
                print("  No improvement found. Stopping.")
            reac_atoms.set_positions(best_reac_pos)
            prod_atoms.set_positions(best_prod_pos)
            break

    else:
        if verbose:
            print(f"Reached max iterations ({max_iterations}).")

    # Final NEB with best endpoints (full quality, fresh linear interpolation)
    reac_atoms.set_positions(best_reac_pos)
    prod_atoms.set_positions(best_prod_pos)
    if verbose:
        print("\n=== Final NEB ===")
    images = _build_neb_images(best_reac_pos, best_prod_pos, symbols, n_images)
    images, energies, _ = _run_neb(images, n_images, neb_fmax, neb_steps, verbose)

    interior = [e for e in energies[1:-1] if e is not None]
    score = max(interior) if interior else float("inf")

    return {
        "reactant_coords": images[0].get_positions(),
        "product_coords": images[-1].get_positions(),
        "images": images,
        "energies": energies,
        "score": score,
        "max_bond_violation": max(
            bound_violation(images[0].get_positions(), bl_init, bu_init, reac_bonds),
            bound_violation(images[-1].get_positions(), bl_final, bu_final, prod_bonds),
        ),
    }


# ---------------------------------------------------------------------------
# Public high-level entry point
# ---------------------------------------------------------------------------

def run_iterative_alignment(
    reaction: "Reaction",
    neb_fmax: float = 0.001,
    neb_steps: int = 200,
    backtrack_fmax: Optional[float] = None,
    backtrack_steps: Optional[int] = None,
    propagation_threshold: float = 0.0001,
    propagation_step_size: float = 2.0,
    max_iterations: int = 40,
    objective: str = "barrier",
    d_power: int = 4,
    verbose: bool = False,
    n_workers: int = 1,
    calculator: "Optional[Calculation]" = None,
) -> "Reaction":
    """
    Run iterative IDPP path optimisation on each conformer in
    ``reaction.top_k_conformers`` and populate ``reaction.aligned_results``
    (sorted by max interior IDPP energy, ascending).

    Requires ``reaction.top_k_conformers`` to be populated — call
    :func:`~sparta.initial_generation.bounds_matrix.generate_initial_conformers` first.

    Bounds matrices stored in each conformer dict are reused directly; nothing
    is recalculated from scratch.

    Parameters
    ----------
    reaction : Reaction
        A :class:`~sparta.reaction.Reaction` object with ``top_k_conformers``
        already populated.
    neb_fmax : float
        Force convergence criterion for each inner NEB run.
    neb_steps : int
        Maximum steps per inner NEB run.
    backtrack_fmax : float, optional
        Force convergence for cheap backtracking NEB evaluations.
        Defaults to ``neb_fmax * 10``.
    backtrack_steps : int, optional
        Maximum steps for cheap backtracking NEB evaluations.
        Defaults to ``max(neb_steps // 4, 20)``.
    propagation_threshold : float
        Stop iterating when the max propagated force falls below this value.
    propagation_step_size : float
        Initial endpoint step size (Å). Halved on rejection.
    max_iterations : int
        Maximum phase-2/3 iterations per conformer.
    objective : {'barrier', 'total'}
        'barrier' — minimise max interior IDPP energy.
        'total'   — minimise sum of interior IDPP energies.
    d_power : int
        IDPP distance weighting exponent.
    verbose : bool
        Print per-conformer progress.
    n_workers : int
        Number of parallel worker processes.  Set > 1 to use
        ``concurrent.futures.ProcessPoolExecutor`` for parallel alignment.
    calculator : Calculation, optional
        If provided, run single-point energy calculations on the reactant and
        product endpoints and the highest-IDPP-energy interior image of the
        best aligned result, storing results in ``reaction.best_reac_sp_energy``,
        ``reaction.best_prod_sp_energy``, and ``reaction.best_midpoint_sp_energy``.

    Returns
    -------
    Reaction
        The same object with ``aligned_results`` populated.

    Raises
    ------
    ValueError
        If ``reaction.top_k_conformers`` is empty.
    """
    if not reaction.top_k_conformers:
        raise ValueError(
            "reaction.top_k_conformers is empty. "
            "Call generate_initial_conformers(reaction) first."
        )

    kwargs = dict(
        d_power=d_power,
        neb_fmax=neb_fmax,
        neb_steps=neb_steps,
        backtrack_fmax=backtrack_fmax,
        backtrack_steps=backtrack_steps,
        propagation_threshold=propagation_threshold,
        propagation_step_size=propagation_step_size,
        max_iterations=max_iterations,
        objective=objective,
        verbose=verbose,
    )

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        args_list = [(reaction, conformer, kwargs)
                     for conformer in reaction.top_k_conformers]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            raw_results = list(executor.map(_alignment_worker, args_list))
    else:
        raw_results = []
        for conformer in reaction.top_k_conformers:
            raw_results.append(_alignment_worker((reaction, conformer, kwargs)))

    results = []
    for i, (result, conformer) in enumerate(
        zip(raw_results, reaction.top_k_conformers)
    ):
        if result is None:
            continue
        result["conformer_idx"] = i
        result["stretch_factor"] = conformer["stretch_factor"]
        result["seed"] = conformer["seed"]
        results.append(result)

    results.sort(key=lambda r: r["score"])
    reaction.aligned_results = results

    if calculator is not None and reaction.aligned_results:
        import ase

        best = reaction.aligned_results[0]
        symbols = [
            reaction.reactant_mol.GetAtomWithIdx(i).GetSymbol()
            for i in range(reaction.reactant_mol.GetNumAtoms())
        ]

        reaction.best_reac_sp_energy = calculator.energy(
            ase.Atoms(symbols=symbols, positions=best["reactant_coords"])
        )
        reaction.best_prod_sp_energy = calculator.energy(
            ase.Atoms(symbols=symbols, positions=best["product_coords"])
        )

        interior_energies = best["energies"][1:-1]
        max_interior_idx = int(np.argmax(interior_energies)) + 1
        reaction.best_midpoint_sp_energy = calculator.energy(best["images"][max_interior_idx])

    return reaction


# ---------------------------------------------------------------------------
# Stretch-factor search
# ---------------------------------------------------------------------------

def find_optimal_stretch_factor(
    reaction: "Reaction",
    start: float = 1.0,
    step: float = 0.1,
    max_stretch: float = 3.0,
    n_workers: int = 1,
    neb_fmax: float = 0.001,
    neb_steps: int = 200,
    backtrack_fmax: Optional[float] = None,
    backtrack_steps: Optional[int] = None,
    propagation_threshold: float = 0.0001,
    propagation_step_size: float = 2.0,
    max_iterations: int = 40,
    objective: str = "barrier",
    d_power: int = 4,
    verbose: bool = False,
) -> "Reaction":
    """
    Search for the optimal stretch factor by incrementally increasing from
    ``start`` until the best alignment score stops improving.

    For each stretch factor a fresh :class:`~sparta.reaction.Reaction` is
    created (copying all settings from *reaction*) and the full
    generate→align pipeline is run.  The search stops as soon as the best
    score is no longer lower than the previous best, or ``max_stretch`` is
    reached.

    Parameters
    ----------
    reaction : Reaction
        Template reaction whose settings are copied for each trial.
    start : float
        Initial stretch factor (default 1.0).
    step : float
        Increment between successive stretch factors (default 0.1).
    max_stretch : float
        Upper bound on the stretch factor (default 3.0).
    n_workers : int
        Parallel worker processes passed to ``generate_initial_conformers``
        and ``run_iterative_alignment``.
    neb_fmax, neb_steps, backtrack_fmax, backtrack_steps,
    propagation_threshold, propagation_step_size,
    max_iterations, objective, d_power, verbose :
        Alignment hyperparameters forwarded to ``run_iterative_alignment``.

    Returns
    -------
    Reaction
        The trial :class:`~sparta.reaction.Reaction` with the lowest
        ``aligned_results[0]["score"]``.

    Raises
    ------
    ValueError
        If no valid results are found across all tried stretch factors.
    """
    from sparta.reaction import Reaction
    from sparta.initial_generation import generate_initial_conformers

    alignment_kwargs = dict(
        neb_fmax=neb_fmax,
        neb_steps=neb_steps,
        backtrack_fmax=backtrack_fmax,
        backtrack_steps=backtrack_steps,
        propagation_threshold=propagation_threshold,
        propagation_step_size=propagation_step_size,
        max_iterations=max_iterations,
        objective=objective,
        d_power=d_power,
        verbose=verbose,
        n_workers=n_workers,
    )

    best_score = float("inf")
    best_reaction = None

    stretch = start
    while stretch <= max_stretch + 1e-9:
        trial = Reaction(
            reactant_smiles=reaction.reactant_smiles,
            product_smiles=reaction.product_smiles,
            charge=reaction.charge,
            multiplicity=reaction.multiplicity,
            n_conformers=reaction.n_conformers,
            n_seeds=reaction.n_seeds,
            stretch_factors=[round(stretch, 10)],
            top_k=reaction.top_k,
            n_images=reaction.n_images,
            filter_dihedrals=reaction.filter_dihedrals,
            k_nearest_dihedrals=reaction.k_nearest_dihedrals,
            cutoff=reaction.cutoff,
            rotation=reaction.rotation,
            translation=reaction.translation,
        )

        try:
            trial = generate_initial_conformers(trial, n_workers=n_workers)
            if not trial.top_k_conformers:
                stretch = round(stretch + step, 10)
                continue
            trial = run_iterative_alignment(trial, **alignment_kwargs)
        except Exception:
            stretch = round(stretch + step, 10)
            continue

        if not trial.aligned_results:
            stretch = round(stretch + step, 10)
            continue

        score = trial.aligned_results[0]["score"]

        if verbose:
            print(f"stretch_factor={stretch:.2f}  best_score={score:.6f}")

        if score < best_score - 1e-8:
            best_score = score
            best_reaction = trial
        else:
            break

        stretch = round(stretch + step, 10)

    if best_reaction is None:
        raise ValueError(
            "find_optimal_stretch_factor: no valid results found for any "
            f"stretch factor in [{start}, {max_stretch}]."
        )

    return best_reaction
