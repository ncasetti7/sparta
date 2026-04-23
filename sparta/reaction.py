"""
Reaction — central data class for reactant-product alignment.

Typical usage::

    from sparta import Reaction

    rxn = Reaction(
        reactant_smiles="[CH4:1].[F:2][F:3]>>[CH3:1][F:2].[H:4][F:3]",
        product_smiles="[CH3:1][F:2].[H:4][F:3]",
        n_conformers=10,
        n_seeds=3,
        stretch_factors=[1.3, 1.5, 1.8],
        top_k=5,
    )
    rxn = rxn.generate_initial_conformers().run_iterative_alignment()

    # Results are stored on the object
    best = rxn.aligned_results[0]  # sorted by alignment score
    reactant_xyz = best["reactant_coords"]
    product_xyz  = best["product_coords"]
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdmolops

from sparta import cheminformatics


class Reaction:
    """
    Stores reaction SMILES, hyperparameters, and all results of conformer
    generation and iterative alignment.

    Parameters
    ----------
    reactant_smiles : str
        Atom-mapped SMILES for the reactant(s).
    product_smiles : str
        Atom-mapped SMILES for the product(s).
    n_conformers : int
        Number of conformers to generate per (seed, stretch_factor) combination.
    n_seeds : int
        Number of random seeds to use (seeds 0 … n_seeds-1).
    stretch_factors : float or list of float
        Stretch factor(s) applied to changing bond target lengths in the bounds
        matrix.  A list causes generation to loop over all values.
    top_k : int
        Number of best initial conformer pairs (ranked by NEB score) to carry
        forward into iterative alignment.
    n_images : int
        Number of images for NEB path scoring and overlap evaluation.
    filter_dihedrals : bool
        If True, only dihedrals within k_nearest bonds of the reaction center
        are included in the optimization.
    k_nearest_dihedrals : int
        Bond-distance cutoff for dihedral filtering.
    cutoff : float
        Neighbour-pair cutoff (Å) used during overlap evaluation.
    rotation : bool or None
        Whether to include fragment rotations in the alignment optimization.
        Defaults to True.
    translation : bool or None
        Whether to include fragment translations.  Defaults to True when there
        are multiple fragments, False otherwise.
    """

    def __init__(
        self,
        reactant_smiles: str,
        product_smiles: str,
        n_conformers: int = 10,
        n_seeds: int = 1,
        stretch_factors: Union[float, List[float]] = 1.5,
        top_k: int = 5,
        n_images: int = 15,
        filter_dihedrals: bool = True,
        k_nearest_dihedrals: int = 3,
        cutoff: float = 10.0,
        rotation: Optional[bool] = None,
        translation: Optional[bool] = None,
    ) -> None:
        self.reactant_smiles = reactant_smiles
        self.product_smiles = product_smiles
        self.n_conformers = n_conformers
        self.n_seeds = n_seeds
        self.stretch_factors = (
            [float(stretch_factors)]
            if isinstance(stretch_factors, (int, float))
            else [float(sf) for sf in stretch_factors]
        )
        self.top_k = top_k
        self.n_images = n_images
        self.filter_dihedrals = filter_dihedrals
        self.k_nearest_dihedrals = k_nearest_dihedrals
        self.cutoff = cutoff

        # ── Molecule objects ──────────────────────────────────────────────
        self.reactant_mol: Chem.Mol = cheminformatics.make_mol(reactant_smiles)
        self.product_mol: Chem.Mol = cheminformatics.make_mol(product_smiles)

        # ── Reaction topology ─────────────────────────────────────────────
        self.changing_bonds: List[Tuple[int, int]] = self._compute_changing_bonds()

        # ── Reference conformers for bond-length initialisation ───────────
        # A quick UFF conformer (seed=0) is embedded here solely to obtain
        # reference bond lengths.  The proper ETKDG conformers are generated
        # in generate_initial_conformers().
        reac_ref = cheminformatics.make_mol(reactant_smiles)
        prod_ref = cheminformatics.make_mol(product_smiles)
        reac_ref = cheminformatics.generate_conformer(reac_ref, reactant_smiles, seed=0)
        prod_ref = cheminformatics.generate_conformer(prod_ref, product_smiles, seed=0)
        _, reac_ref_coords = cheminformatics.get_coords_conformer(reac_ref)
        _, prod_ref_coords = cheminformatics.get_coords_conformer(prod_ref)

        self.reac_ref_coords: np.ndarray = reac_ref_coords
        self.prod_ref_coords: np.ndarray = prod_ref_coords

        self.reac_bond_lengths: Dict[Tuple[int, int], float] = {}
        self.prod_bond_lengths: Dict[Tuple[int, int], float] = {}
        for i, j in self.changing_bonds:
            self.reac_bond_lengths[(i, j)] = float(
                np.linalg.norm(reac_ref_coords[i] - reac_ref_coords[j])
            )
            self.prod_bond_lengths[(i, j)] = float(
                np.linalg.norm(prod_ref_coords[i] - prod_ref_coords[j])
            )

        # ── Base bounds matrices (computed once; reused across all seeds/stretches) ──
        self.reactant_base_bm: np.ndarray = rdDistGeom.GetMoleculeBoundsMatrix(
            self.reactant_mol
        )
        self.product_base_bm: np.ndarray = rdDistGeom.GetMoleculeBoundsMatrix(
            self.product_mol
        )

        # ── Dihedrals and fragment topology ──────────────────────────────
        self.reactant_dihedrals = cheminformatics.get_dihedral_matches_all(self.reactant_mol)
        self.product_dihedrals = cheminformatics.get_dihedral_matches_all(self.product_mol)

        if filter_dihedrals and self.changing_bonds:
            self.reactant_dihedrals, self.product_dihedrals = (
                cheminformatics.filter_dihedrals_k_nearest(
                    self.reactant_dihedrals,
                    self.product_dihedrals,
                    self.changing_bonds,
                    self.reactant_mol,
                    self.product_mol,
                    k=k_nearest_dihedrals,
                )
            )

        self.reactant_bonded_atoms_dict = cheminformatics.generate_bonded_atoms_dict(
            self.reactant_mol, self.reactant_dihedrals
        )
        self.product_bonded_atoms_dict = cheminformatics.generate_bonded_atoms_dict(
            self.product_mol, self.product_dihedrals
        )

        self.reactant_frags: List[List[int]] = [
            list(f) for f in rdmolops.GetMolFrags(self.reactant_mol)
        ]
        self.product_frags: List[List[int]] = [
            list(f) for f in rdmolops.GetMolFrags(self.product_mol)
        ]

        # ── Rotation / translation flags ──────────────────────────────────
        self.rotation: bool = True if rotation is None else rotation
        if translation is None:
            self.translation = (
                len(self.reactant_frags) > 1 or len(self.product_frags) > 1
            )
        else:
            self.translation = translation
        if self.translation and not self.rotation:
            raise ValueError("translation=True requires rotation=True.")

        # ── Result containers (populated by the two main methods) ─────────
        # Each element of all_conformers is a dict:
        #   reactant_coords, product_coords,
        #   reactant_bounds_matrix, product_bounds_matrix,
        #   stretch_factor, seed, score
        self.all_conformers: List[Dict] = []
        self.top_k_conformers: List[Dict] = []

        # Each element of aligned_results is a dict:
        #   reactant_coords, product_coords, score,
        #   conformer_idx, stretch_factor, seed
        self.aligned_results: List[Dict] = []

        # ── Calculator results (populated when a Calculation is supplied) ──
        # Optimised ref conformer geometry and energy (from generate_initial_conformers)
        self.reac_ref_optimized: Optional[np.ndarray] = None
        self.prod_ref_optimized: Optional[np.ndarray] = None
        self.reac_ref_energy: Optional[float] = None
        self.prod_ref_energy: Optional[float] = None
        # Single-point energies on the best aligned result (from run_iterative_alignment)
        self.best_reac_sp_energy: Optional[float] = None
        self.best_prod_sp_energy: Optional[float] = None
        self.best_midpoint_sp_energy: Optional[float] = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _compute_changing_bonds(self) -> List[Tuple[int, int]]:
        """Symmetric difference of bond sets between reactant and product."""
        reac_bonds = cheminformatics.get_bonds(self.reactant_mol)
        prod_bonds = cheminformatics.get_bonds(self.product_mol)
        return sorted(reac_bonds.symmetric_difference(prod_bonds))

    # ── Public API ────────────────────────────────────────────────────────

    def generate_initial_conformers(
        self,
        n_workers: int = 1,
        calculator=None,
    ) -> "Reaction":
        """
        Generate reactant/product conformer pairs for every (seed, stretch_factor)
        combination, score each pair with a quick NEB, and populate
        ``self.all_conformers`` and ``self.top_k_conformers``.

        Parameters
        ----------
        n_workers : int
            Parallel worker processes for conformer scoring.
        calculator : Calculation, optional
            If provided, optimise the seed-0 UFF reference conformers and store
            the optimised geometries and energies in ``reac_ref_optimized``,
            ``prod_ref_optimized``, ``reac_ref_energy``, and ``prod_ref_energy``.

        Returns ``self`` to allow method chaining.
        """
        from sparta.initial_generation import generate_initial_conformers
        return generate_initial_conformers(self, n_workers=n_workers, calculator=calculator)

    def run_iterative_alignment(
        self,
        neb_fmax: float = 0.001,
        neb_steps: int = 200,
        propagation_threshold: float = 0.0001,
        propagation_step_size: float = 2.0,
        max_iterations: int = 40,
        objective: str = "barrier",
        d_power: int = 4,
        verbose: bool = False,
        n_workers: int = 1,
        calculator=None,
    ) -> "Reaction":
        """
        Run iterative IDPP path optimisation on each conformer in
        ``self.top_k_conformers`` and populate ``self.aligned_results``.

        Must be called after :meth:`generate_initial_conformers`.
        Returns ``self`` to allow method chaining.

        Parameters
        ----------
        neb_fmax : float
            Force convergence criterion for each inner NEB run.
        neb_steps : int
            Maximum steps per inner NEB run.
        propagation_threshold : float
            Stop iterating when the max propagated force falls below this value.
        propagation_step_size : float
            Initial endpoint step size (Å). Halved on rejection.
        max_iterations : int
            Maximum phase-2/3 iterations per conformer.
        objective : {'barrier', 'total'}
            'barrier' — minimise max interior IDPP energy (barrier height).
            'total'   — minimise sum of interior IDPP energies.
        d_power : int
            IDPP distance weighting exponent.
        verbose : bool
            Print progress for each conformer.
        n_workers : int
            Parallel worker processes for per-conformer alignment.
        calculator : Calculation, optional
            If provided, run single-point energy calculations on the endpoints
            and highest-IDPP-energy image of the best aligned result, storing
            results in ``best_reac_sp_energy``, ``best_prod_sp_energy``, and
            ``best_midpoint_sp_energy``.
        """
        from sparta.iterative_alignment import run_iterative_alignment
        return run_iterative_alignment(
            self,
            neb_fmax=neb_fmax,
            neb_steps=neb_steps,
            propagation_threshold=propagation_threshold,
            propagation_step_size=propagation_step_size,
            max_iterations=max_iterations,
            objective=objective,
            d_power=d_power,
            verbose=verbose,
            n_workers=n_workers,
            calculator=calculator,
        )

    def find_optimal_stretch_factor(
        self,
        start: float = 1.0,
        step: float = 0.1,
        max_stretch: float = 3.0,
        n_workers: int = 1,
        neb_fmax: float = 0.001,
        neb_steps: int = 200,
        propagation_threshold: float = 0.0001,
        propagation_step_size: float = 2.0,
        max_iterations: int = 40,
        objective: str = "barrier",
        d_power: int = 4,
        verbose: bool = False,
    ) -> "Reaction":
        """
        Search for the optimal stretch factor, starting at ``start`` and
        incrementing by ``step`` until the alignment score stops improving.

        Returns a new :class:`Reaction` (not ``self``) configured with the
        best stretch factor and fully aligned results populated.

        Parameters
        ----------
        start : float
            Initial stretch factor (default 1.0).
        step : float
            Increment per trial (default 0.1).
        max_stretch : float
            Upper bound on stretch factor (default 3.0).
        n_workers : int
            Parallel worker processes.
        neb_fmax, neb_steps, propagation_threshold, propagation_step_size,
        max_iterations, objective, d_power, verbose :
            Alignment hyperparameters.
        """
        from sparta.iterative_alignment import find_optimal_stretch_factor
        return find_optimal_stretch_factor(
            self,
            start=start,
            step=step,
            max_stretch=max_stretch,
            n_workers=n_workers,
            neb_fmax=neb_fmax,
            neb_steps=neb_steps,
            propagation_threshold=propagation_threshold,
            propagation_step_size=propagation_step_size,
            max_iterations=max_iterations,
            objective=objective,
            d_power=d_power,
            verbose=verbose,
        )
