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
from rdkit import Chem, DistanceGeometry
from rdkit.Chem import rdDistGeom, rdmolops

from sparta import cheminformatics
from sparta.stereochemistry import (
    alkene_stereo_references,
    atropisomer_stereo_references,
    log_alkene_state,
    log_atropisomer_state,
    log_stereo_state,
    tetrahedral_stereo_references,
)


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
    charge : int, optional
        Total molecular charge.  If omitted, inferred from formal charges in
        the reactant SMILES (reactant and product must agree).  Passed to any
        calculator supplied to :meth:`generate_initial_conformers` or
        :meth:`run_iterative_alignment`, overriding whatever charge the
        calculator was initialised with.
    multiplicity : int, optional
        Spin multiplicity (2S + 1).  If omitted, inferred from the number of
        radical electrons in the reactant SMILES (``n_radical_electrons + 1``).
        Handled the same way as ``charge``.
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
    catalyst_bias : float
        If > 0, bias non-reactive (spectator) fragments — e.g. a catalyst or
        additive — toward the reaction center during conformer embedding by
        capping their distance to it with a loose upper bound in the bounds
        matrices.  Ranges from 0 (no bias; spectator placed freely) to 1
        (tightest leash).  The spectator still samples position and orientation
        freely within the leash radius, so conformational diversity is largely
        preserved.  Name the catalyst with ``catalyst_atoms`` and vary the tether
        geometry with ``tether_mode`` / ``tether_anchor``.
    catalyst_atoms : list of int, optional
        Atom indices (atom-map number − 1) identifying the catalyst.  Any atom in
        the catalyst suffices; the whole fragment(s) containing those atoms is
        treated as the catalyst, and the reaction center becomes the changing-bond
        atoms *not* in the catalyst (for a multi-component reaction, the shared
        center of the reacting molecules).  When None, the catalyst is
        auto-detected as every fragment with no changing-bond atoms.
    tether_mode : {'center', 'explore'}
        How the catalyst is tethered to the reaction center.  'center' tethers the
        catalyst's centroid-nearest heavy atom to the reaction-center centroid (a
        single head-on pose).  'explore' covers both the catalyst and the reaction
        center with ~4 Å spheres and emits one tether configuration per
        (catalyst-sphere, reaction-sphere) pair, sampling many approach poses for
        broad coverage; ``n_seeds`` then adds random replicates per pose.  Ignored
        when ``tether_anchor`` is given.
    tether_anchor : int or list of int, optional
        Explicit catalyst atom index or indices to tether to the reaction center,
        overriding ``tether_mode``. A list is treated as alternative explicit
        anchors, one tether configuration per atom — use when several catalyst
        atoms could contact the substrate.
    diversity_rmsd : float
        RMSD threshold (Å) for diversity filtering of ``top_k_conformers``.
        Conformer pairs are grouped into "categories" by best-fit RMSD and only
        the best-scoring pair from each category is kept, so the top_k carry
        distinct geometries rather than near-duplicates.  Two pairs share a
        category when both their reactant and product RMSDs fall below this
        value.  Set to 0 to disable (keep the best top_k purely by score).
    selection : {'coverage', 'score'}
        How ``top_k_conformers`` is chosen from the scored pool.  'coverage'
        (default) is a farthest-point selection that maximises spread in
        symmetry-aware RMSD space: the best-scoring pair anchors the set and each
        further pick is the pose farthest from those already chosen, so the budget
        spans diverse geometries for transition-state seeding rather than the
        lowest-barrier ones.  'score' keeps the best-scoring representative of each
        of the first ``top_k`` distinct RMSD clusters.  Both stop early once only
        near-duplicates (within ``diversity_rmsd``) remain.
    score_cap : float or None
        Feasibility gate applied before ``selection``: conformer pairs whose NEB
        score exceeds ``score_cap`` × the lowest (best) score in the pool are
        excluded, so coverage/score selection never reaches into the strained,
        high-barrier tail.  Must be >= 1.  Defaults to 3.0; set to None to
        disable.  ``all_conformers`` still holds the full scored pool.
    stereo_debug : bool
        If True, print opt-in stereochemistry diagnostics during reference
        creation, initial product generation, and iterative alignment. Can also
        be enabled with the ``SPARTA_STEREO_DEBUG=1`` environment variable.
    n_workers : int
        Default number of worker processes used by method calls when
        ``n_workers`` is not passed explicitly.
    """

    def __init__(
        self,
        reactant_smiles: str,
        product_smiles: str,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
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
        catalyst_bias: float = 0.0,
        catalyst_atoms: Optional[List[int]] = None,
        tether_mode: str = "center",
        tether_anchor: Optional[Union[int, List[int]]] = None,
        diversity_rmsd: float = 0.5,
        selection: str = "coverage",
        score_cap: Optional[float] = 3.0,
        stereo_debug: bool = False,
        n_workers: int = 1,
    ) -> None:
        self.reactant_smiles = reactant_smiles
        self.product_smiles = product_smiles
        self.stereo_debug = stereo_debug
        self.n_workers = n_workers
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
        self.diversity_rmsd = diversity_rmsd
        if selection not in ("coverage", "score"):
            raise ValueError(
                f"selection must be 'coverage' or 'score', got {selection!r}."
            )
        self.selection = selection
        if score_cap is not None and score_cap < 1.0:
            raise ValueError(f"score_cap must be >= 1 or None, got {score_cap}.")
        self.score_cap = score_cap

        # ── Molecule objects ──────────────────────────────────────────────
        self.reactant_mol: Chem.Mol = cheminformatics.make_mol(reactant_smiles)
        self.product_mol: Chem.Mol = cheminformatics.make_mol(product_smiles)

        # ── Charge and multiplicity ───────────────────────────────────────
        reac_charge = rdmolops.GetFormalCharge(self.reactant_mol)
        prod_charge = rdmolops.GetFormalCharge(self.product_mol)
        if reac_charge != prod_charge:
            raise ValueError(
                f"Reactant charge ({reac_charge}) differs from product charge "
                f"({prod_charge}): charge is not conserved."
            )
        self.charge: int = reac_charge if charge is None else charge
        self.multiplicity: int = (
            sum(a.GetNumRadicalElectrons() for a in self.reactant_mol.GetAtoms()) + 1
            if multiplicity is None else multiplicity
        )

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
        self.reactant_stereo_refs = tetrahedral_stereo_references(reac_ref, reac_ref_coords)
        self.product_stereo_refs = tetrahedral_stereo_references(prod_ref, prod_ref_coords)
        self.reactant_alkene_refs = alkene_stereo_references(reac_ref, reac_ref_coords)
        self.product_alkene_refs = alkene_stereo_references(prod_ref, prod_ref_coords)
        self.reactant_atropisomer_refs = atropisomer_stereo_references(reac_ref, reac_ref_coords)
        self.product_atropisomer_refs = atropisomer_stereo_references(prod_ref, prod_ref_coords)
        log_stereo_state(self, "reactant reference", reac_ref_coords, self.reactant_stereo_refs)
        log_stereo_state(self, "product reference", prod_ref_coords, self.product_stereo_refs)
        log_alkene_state(self, "reactant reference", reac_ref_coords, self.reactant_alkene_refs)
        log_alkene_state(self, "product reference", prod_ref_coords, self.product_alkene_refs)
        log_atropisomer_state(
            self,
            "reactant reference",
            reac_ref_coords,
            self.reactant_atropisomer_refs,
        )
        log_atropisomer_state(
            self,
            "product reference",
            prod_ref_coords,
            self.product_atropisomer_refs,
        )

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

        # ── Optional catalyst leash ───────────────────────────────────────
        # When a bias is requested, build one or more tether configurations and
        # precompute the matching leashed base bounds matrices (one per config).
        # Nothing below runs when catalyst_bias == 0.
        self.catalyst_bias: float = catalyst_bias
        self.catalyst_atoms = None
        self.tether_mode = tether_mode
        self.tether_anchor = None
        self._leashed_reac_bms: List[np.ndarray] = []
        self._leashed_prod_bms: List[np.ndarray] = []
        if catalyst_bias > 0.0:
            if tether_mode not in ("center", "explore"):
                raise ValueError(
                    f"tether_mode must be 'center' or 'explore', got {tether_mode!r}."
                )
            self.catalyst_atoms = (
                list(catalyst_atoms) if catalyst_atoms is not None else None
            )
            if isinstance(tether_anchor, (list, tuple, set)):
                self.tether_anchor = [int(anchor) for anchor in tether_anchor]
            elif tether_anchor is None:
                self.tether_anchor = None
            else:
                self.tether_anchor = int(tether_anchor)
            self._build_catalyst_leash(catalyst_bias)
        self.n_tether_configs: int = max(1, len(self._leashed_reac_bms))

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

    # Bias → leash-radius mapping (Å): bias 1.0 → R_near, bias → 0 → R_far.
    _LEASH_R_NEAR: float = 4.0   # tightest leash; catalyst docked near center
    _LEASH_R_FAR: float = 12.0   # effectively unconstrained for typical systems
    _COVER_RADIUS: float = 4.0   # 'explore' sphere radius (Å) covering each piece

    @staticmethod
    def _central_atom(atom_indices: List[int], coords: np.ndarray) -> int:
        """Return the atom in atom_indices closest to their geometric centroid."""
        idx = np.asarray(atom_indices)
        pts = coords[idx]
        dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        return int(idx[int(np.argmin(dists))])

    @staticmethod
    def _sphere_cover(
        atom_indices: List[int], coords: np.ndarray, radius: float
    ) -> List[int]:
        """
        Greedy farthest-point cover: return a small set of atoms (sphere centers)
        such that every atom in ``atom_indices`` lies within ``radius`` of one.

        The first center is the centroid-nearest atom; each subsequent center is
        the atom farthest from all current centers, until everything is covered.
        Yields ~O(extent / radius) representatives — modest molecular coverage
        without enumerating every atom.
        """
        idx = list(atom_indices)
        if not idx:
            return []
        pts = coords[np.asarray(idx)]
        centroid = pts.mean(axis=0)
        first = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
        centers = [first]
        # nearest[i] = distance from atom i to its closest chosen center, updated
        # incrementally (O(N) per added center) rather than rebuilding the full
        # (N, C) distance matrix each iteration.
        nearest = np.linalg.norm(pts - pts[first], axis=1)
        while True:
            far = int(np.argmax(nearest))
            if nearest[far] <= radius:
                break
            centers.append(far)
            nearest = np.minimum(nearest, np.linalg.norm(pts - pts[far], axis=1))
        return [idx[c] for c in centers]

    def _tether_configs(self) -> List[List[Tuple[int, int]]]:
        """
        Build the tether configurations implied by ``tether_mode`` /
        ``tether_anchor``.  Each configuration is a list of
        (catalyst_atom, reaction_center_atom) edges.

        The catalyst is the fragment(s) named by ``catalyst_atoms`` or, when that
        is None, every fragment with no changing-bond atom.  The reaction center
        is the changing-bond atoms *not* in the catalyst — for a multi-component
        reaction, the shared center of the reacting molecules.

        - explicit ``tether_anchor``: one config per anchor, one edge each.
        - 'center': one config tethering each catalyst fragment's centroid heavy
          atom to the reaction-center centroid (head-on).
        - 'explore': cover the catalyst and the reaction center with
          ``_COVER_RADIUS`` spheres and emit one config per
          (catalyst-sphere, reaction-sphere) pair, sampling many poses.

        Every edge spans two different fragments in both reactant and product, so
        a molecule is never tethered to itself.
        """
        rxn_atoms = {a for bond in self.changing_bonds for a in bond}
        if not rxn_atoms:
            return []

        if self.catalyst_atoms is not None:
            cat_set = set(self.catalyst_atoms)
            cat_frags = [f for f in self.reactant_frags if cat_set & set(f)]
            if not cat_frags:
                raise ValueError(
                    f"catalyst_atoms {sorted(cat_set)} match no reactant fragment."
                )
            cat_all = [a for f in cat_frags for a in f]
            center_atoms = sorted(rxn_atoms - set(cat_all))
        else:
            cat_frags = [f for f in self.reactant_frags if not (set(f) & rxn_atoms)]
            cat_all = [a for f in cat_frags for a in f]
            center_atoms = sorted(rxn_atoms)
        if not cat_frags or not center_atoms:
            return []

        frag_reac = {a: fi for fi, frag in enumerate(self.reactant_frags) for a in frag}
        frag_prod = {a: fi for fi, frag in enumerate(self.product_frags) for a in frag}

        def heavy(atoms: List[int]) -> List[int]:
            h = [a for a in atoms
                 if self.reactant_mol.GetAtomWithIdx(a).GetAtomicNum() > 1]
            return h or list(atoms)

        rxn_center = self._central_atom(center_atoms, self.reac_ref_coords)

        if self.tether_anchor is not None:
            anchors = (
                self.tether_anchor
                if isinstance(self.tether_anchor, list)
                else [self.tether_anchor]
            )
            raw = [[(anchor, rxn_center)] for anchor in anchors]
        elif self.tether_mode == "center":
            raw = [[(self._central_atom(heavy(f), self.reac_ref_coords), rxn_center)
                    for f in cat_frags]]
        else:  # 'explore'
            cat_reps = self._sphere_cover(heavy(cat_all), self.reac_ref_coords, self._COVER_RADIUS)
            rxn_reps = self._sphere_cover(center_atoms, self.reac_ref_coords, self._COVER_RADIUS)
            raw = [[(c, t)] for c in cat_reps for t in rxn_reps]

        # Cross-fragment guard: drop within-molecule edges and now-empty configs.
        configs = []
        for cfg in raw:
            edges = [(a, b) for (a, b) in cfg
                     if frag_reac.get(a) != frag_reac.get(b)
                     and frag_prod.get(a) != frag_prod.get(b)]
            if edges:
                configs.append(edges)
        return configs

    def _build_catalyst_leash(self, bias: float) -> None:
        """
        Precompute one leashed, triangle-smoothed base bounds-matrix pair per
        tether configuration, stored in ``_leashed_reac_bms`` /
        ``_leashed_prod_bms``.

        Each configuration caps the distance between its catalyst atom(s) and the
        reaction center at the bias-derived radius.  ``base_bms_for_config`` hands
        these out per embedding batch, and ``generate_initial_conformers`` cycles
        through all of them, so 'explore' samples the configured poses.
        """
        configs = self._tether_configs()
        if not configs:
            return
        radius = self._LEASH_R_FAR - bias * (self._LEASH_R_FAR - self._LEASH_R_NEAR)
        for cfg in configs:
            reac_bm = self.reactant_base_bm.copy()
            prod_bm = self.product_base_bm.copy()
            for anchor, target in cfg:
                a, b = min(anchor, target), max(anchor, target)
                reac_bm[a, b] = min(reac_bm[a, b], radius)
                prod_bm[a, b] = min(prod_bm[a, b], radius)
            if not (DistanceGeometry.DoTriangleSmoothing(reac_bm)
                    and DistanceGeometry.DoTriangleSmoothing(prod_bm)):
                raise ValueError(
                    f"catalyst_bias={bias} gives leash radius {radius:.2f} Å, which is "
                    f"geometrically infeasible; reduce catalyst_bias."
                )
            self._leashed_reac_bms.append(reac_bm)
            self._leashed_prod_bms.append(prod_bm)

    def base_bms_for_config(self, config_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Base (reactant, product) bounds matrices for tether configuration
        ``config_idx`` — the precomputed leashed pair when a catalyst leash is
        active, else the plain base matrices (``catalyst_bias == 0``).
        """
        if not self._leashed_reac_bms:
            return self.reactant_base_bm, self.product_base_bm
        return self._leashed_reac_bms[config_idx], self._leashed_prod_bms[config_idx]

    # ── Public API ────────────────────────────────────────────────────────

    def generate_initial_conformers(
        self,
        n_workers: Optional[int] = None,
        calculator=None,
    ) -> "Reaction":
        """
        Generate reactant/product conformer pairs for every (seed, stretch_factor)
        combination, score each pair with a quick NEB, and populate
        ``self.all_conformers`` and ``self.top_k_conformers``.

        Parameters
        ----------
        n_workers : int
            Parallel worker processes for conformer generation batches and
            conformer scoring. Defaults to ``self.n_workers``.
        calculator : Calculation, optional
            If provided, optimise the seed-0 UFF reference conformers and store
            the optimised geometries and energies in ``reac_ref_optimized``,
            ``prod_ref_optimized``, ``reac_ref_energy``, and ``prod_ref_energy``.

        Returns ``self`` to allow method chaining.
        """
        if calculator is not None and hasattr(calculator, "charge"):
            calculator.charge = self.charge
            calculator.multiplicity = self.multiplicity
        from sparta.initial_generation import generate_initial_conformers
        workers = self.n_workers if n_workers is None else n_workers
        return generate_initial_conformers(self, n_workers=workers, calculator=calculator)

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
        n_workers: Optional[int] = None,
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
            Parallel worker processes for per-conformer alignment. Defaults to
            ``self.n_workers``.
        calculator : Calculation, optional
            If provided, run single-point energy calculations on the endpoints
            and highest-IDPP-energy image of the best aligned result, storing
            results in ``best_reac_sp_energy``, ``best_prod_sp_energy``, and
            ``best_midpoint_sp_energy``.
        """
        if calculator is not None and hasattr(calculator, "charge"):
            calculator.charge = self.charge
            calculator.multiplicity = self.multiplicity
        from sparta.iterative_alignment import run_iterative_alignment
        workers = self.n_workers if n_workers is None else n_workers
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
            n_workers=workers,
            calculator=calculator,
        )

    def find_optimal_stretch_factor(
        self,
        start: float = 1.0,
        step: float = 0.1,
        max_stretch: float = 3.0,
        n_workers: Optional[int] = None,
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
            Parallel worker processes. Defaults to ``self.n_workers``.
        neb_fmax, neb_steps, propagation_threshold, propagation_step_size,
        max_iterations, objective, d_power, verbose :
            Alignment hyperparameters.
        """
        from sparta.iterative_alignment import find_optimal_stretch_factor
        workers = self.n_workers if n_workers is None else n_workers
        return find_optimal_stretch_factor(
            self,
            start=start,
            step=step,
            max_stretch=max_stretch,
            n_workers=workers,
            neb_fmax=neb_fmax,
            neb_steps=neb_steps,
            propagation_threshold=propagation_threshold,
            propagation_step_size=propagation_step_size,
            max_iterations=max_iterations,
            objective=objective,
            d_power=d_power,
            verbose=verbose,
        )
