<p align="center">
  <img src="sparta_logo.png" alt="SPARTA logo" width="400"/>
</p>

# SPARTA

**S**teric **P**ath **A**lignment for **R**eactive **T**ransformation **A**ssessment

SPARTA generates reactant-product conformer pairs and iteratively aligns them using an Image-Dependent Pair Potential (IDPP) nudged-elastic-band (NEB) path optimization.

Path generation and alignment are purely geometric and need no electronic-structure program. Evaluating electronic energies along the aligned path gives a barrier estimate that acts as a fast proxy for kinetic feasibility, at a small fraction of the cost of locating a transition state. SPARTA will use any calculator you give it; see [Electronic energies](#electronic-energies).

---

## Installation

```bash
pip install sparta-chem
```

The import name is `sparta`:

```python
from sparta import Reaction
```

| Requirement | Minimum version |
|-------------|----------------|
| Python      | 3.11           |
| numpy       | 1.24           |
| scipy       | 1.16           |
| rdkit       | 2025.9.1       |
| ase         | 3.26           |

---

## Python API

### Quick start

```python
from sparta import Reaction, XTBCalculation, run_sparta

rxn = Reaction(
    reactant_smiles="[C:1]([H:2])([H:3])([H:4])[H:5].[F:6][F:7]",
    product_smiles="[C:1]([H:2])([H:3])([H:4])[F:6].[H:5][F:7]",
)

rxn = run_sparta(rxn, calculator=XTBCalculation())
print(rxn.barrier_kcal)          # SPARTA barrier, kcal/mol
```

`run_sparta` is the whole pipeline in one call. It scans stretch factors from
1.0 to 2.0, aligns one endpoint pair at each, evaluates electronic energies
along the path, and keeps the factor giving the lowest barrier.

Both SMILES must be **fully atom-mapped, including hydrogens**, and the two
mappings must cover the same set of map numbers, that correspondence is what
defines which bonds change. Hydrogens have to be written as explicit mapped
atoms (`[C:1]([H:2])...`) rather than left implicit (`[CH4:1]`): SPARTA parses
with `removeHs=False`, so an implicit hydrogen is not an atom and will be
missing from the generated geometry.

### Geometric scoring only

Without a calculator nothing electronic is computed, and the result carries
the IDPP score alone. This is the cheaper, purely geometric mode:

```python
from sparta import Reaction, generate_initial_conformers, run_iterative_alignment

rxn = Reaction(reactant_smiles=..., product_smiles=..., stretch_factors=[1.3, 1.5, 1.8])
rxn = generate_initial_conformers(rxn)
rxn = run_iterative_alignment(rxn)

print(rxn.aligned_results[0]["score"])     # IDPP score, not an energy
```

Each function returns the reaction, so these also chain:
`Reaction(...).generate_initial_conformers().run_iterative_alignment()`.

### Accessing results

```python
# aligned_results is sorted by score (ascending, lower is better)
best = rxn.aligned_results[0]

print(best["score"])            # float: max interior IDPP energy
print(best["reactant_coords"])  # np.ndarray (n_atoms, 3)
print(best["product_coords"])   # np.ndarray (n_atoms, 3)
print(best["images"])           # list of ase.Atoms, full NEB path
print(best["energies"])         # list of float, per-image IDPP energies
print(best["seed"])             # int
print(best["stretch_factor"])   # float
print(best["conformer_idx"])    # int: index into rxn.top_k_conformers

# present only when a calculator was supplied
print(best["barrier_kcal"])         # SPARTA barrier, kcal/mol
print(best["electronic_energies"])  # list of float, per interior image
print(best["peak_image_idx"])       # index of the highest-energy image
print(rxn.barrier_kcal)             # same as aligned_results[0]["barrier_kcal"]
print(rxn.reference_energy)         # relaxed reactant reference, Hartree
```

Two different quantities live here and they are not interchangeable:

| | meaning | units |
|---|---|---|
| `score` | highest interior **IDPP** energy, purely geometric | IDPP |
| `barrier_kcal` | highest interior **electronic** energy minus the relaxed reactant reference | kcal/mol |

`aligned_results` is sorted by `barrier_kcal` when a calculator was given, and
by `score` otherwise, so `aligned_results[0]` is always the best result by
whichever metric was available.

The reactant reference relaxes each connected fragment separately and sums the
energies. Optimizing a multi-fragment reactant as one unit would fold an
arbitrary inter-fragment interaction into every barrier.

All intermediate conformers are also available:

```python
rxn.all_conformers      # every generated conformer pair, sorted by initial NEB score
rxn.top_k_conformers    # the top-k subset passed to iterative alignment
```

### Electronic energies

The IDPP score is geometric. To obtain an electronic barrier, evaluate single-point energies on the images of the aligned path with a calculator of your choice.

SPARTA ships an `XTBCalculation` wrapper for convenience, which shells out to the [xtb](https://github.com/grimme-lab/xtb) executable. **xtb is not a Python dependency and is not installed by pip**, install it separately and make sure it is on `PATH`, or pass an explicit path via `xtb_cmd`. Nothing else in SPARTA requires it.

```python
from sparta.calculation import XTBCalculation

calc = XTBCalculation(charge=0, multiplicity=1)       # or xtb_cmd="/path/to/xtb"
# XTBCalculation also takes `timeout` (seconds, default 120)
energies = [calc.energy(img) for img in best["images"]]
```

`ASECalculation` wraps any ASE calculator, which covers most machine-learned potentials:

```python
from sparta.calculation import ASECalculation

calc = ASECalculation(ase_calculator)                 # also takes `fmax`, `max_steps`
```

Any object exposing the same `energy(atoms)` interface works in their place, so the path code never needs to know what is underneath.

### `Reaction` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reactant_smiles` | `str` |, | Atom-mapped SMILES for the reactant(s) |
| `product_smiles` | `str` |, | Atom-mapped SMILES for the product(s) |
| `charge` | `int` | auto | Total charge; inferred from the SMILES when omitted |
| `multiplicity` | `int` | auto | Spin multiplicity; inferred from the SMILES when omitted |
| `n_conformers` | `int` | `10` | Conformers per (seed, stretch_factor) combination |
| `n_seeds` | `int` | `1` | Number of random seeds (0 … n_seeds−1) |
| `stretch_factors` | `float` or `list` | `1.5` | Stretch factor(s) applied to changing bond targets in the bounds matrix |
| `top_k` | `int` | `5` | Number of best initial conformers to carry into iterative alignment |
| `n_images` | `int` | `15` | NEB images (including endpoints) |
| `filter_dihedrals` | `bool` | `True` | Restrict optimization to dihedrals near the reaction center |
| `k_nearest_dihedrals` | `int` | `3` | Bond-distance cutoff for dihedral filtering |
| `cutoff` | `float` | `10.0` | Neighbor-pair cutoff (Å) for overlap evaluation |
| `rotation` | `bool` | `True` | Include fragment rotations |
| `translation` | `bool` | auto | Include fragment translations (default: `True` when multiple fragments exist) |

### `generate_initial_conformers` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_workers` | `1` | Conformer pairs generated and scored in parallel |
| `calculator` | `None` | Optional `Calculation`; when given, initial scoring uses it instead of the IDPP energy |

### `run_iterative_alignment` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neb_fmax` | `0.001` | Force convergence criterion for each inner NEB run |
| `neb_steps` | `200` | Maximum steps per inner NEB run |
| `propagation_threshold` | `0.0001` | Stop when max propagated endpoint force < this value |
| `propagation_step_size` | `2.0` | Initial endpoint adjustment step size (Å) |
| `max_iterations` | `40` | Maximum NEB + propagation iterations per conformer |
| `objective` | `"peak"` | Which IDPP quantity to minimize: `"peak"` the highest interior image, `"total"` the sum. Not the electronic barrier |
| `d_power` | `4` | IDPP distance weighting exponent |
| `backtrack_fmax` | `None` | Force criterion for the cheap NEB runs during endpoint stepping; falls back to `neb_fmax` |
| `backtrack_steps` | `None` | Step cap for those runs; falls back to `neb_steps` |
| `n_workers` | `1` | Conformers aligned in parallel |
| `calculator` | `None` | Optional `Calculation` for electronic energies along the path |
| `sp_images` | `"all"` | `"all"` evaluates every interior image; `"peak"` evaluates only the image the IDPP path puts highest, which is cheaper but assumes the two peaks coincide |
| `verbose` | `False` | Print per-iteration progress |

### Choosing a stretch factor automatically

A single stretch factor is not optimal for every reaction. `find_optimal_stretch_factor` scans upward until the score stops improving and leaves the best result on the reaction object:

```python
from sparta import find_optimal_stretch_factor

rxn = find_optimal_stretch_factor(rxn, start=1.0, step=0.1, max_stretch=3.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` | `1.0` | First stretch factor tried |
| `step` | `0.1` | Increment between factors |
| `max_stretch` | `3.0` | Upper bound on the factor |
| `patience` | `3` | Stop after this many consecutive factors fail to improve. `1` stops at the first, which is cheap but leaves gains behind when the score is not monotonic |
| `calculator` | `None` | When given, factors are ranked by electronic barrier rather than IDPP score |
| `rank_by` | `"auto"` | `"auto"` ranks by barrier when a calculator is present and by IDPP score otherwise; `"idpp"` forces geometric ranking |
| `sp_images` | `"all"` | As for `run_iterative_alignment` |

It also accepts every alignment parameter of `run_iterative_alignment`.

### Getting the barrier for an aligned reaction

If a reaction has already been aligned, `sparta_barrier` returns its barrier in
kcal/mol, computing the electronic energies only if they are not already there:

```python
from sparta import sparta_barrier

barrier = sparta_barrier(rxn, calculator=XTBCalculation())
```

### Reproducing the published results

The defaults of `run_sparta` are the published settings. Stated explicitly:

```python
from sparta import Reaction, XTBCalculation, run_sparta

rxn = Reaction(
    reactant_smiles=...,
    product_smiles=...,
    n_conformers=30,          # per stretch factor
    n_seeds=1,
)

rxn = run_sparta(
    rxn,
    calculator=XTBCalculation(),
    stretch_factors=None,     # scan 1.0 to 2.0 in steps of 0.1
    patience=3,               # stop after 3 factors without improvement
)
```

`run_sparta` sets `top_k=1`, so one endpoint pair per stretch factor carries
through to alignment. The reported barriers used g-xTB; `XTBCalculation`
invokes whichever xtb build is on `PATH`, and GFN2 and g-xTB give different
numbers for the same reaction.

To evaluate a fixed set of factors instead of scanning, pass them:
`run_sparta(rxn, calc, stretch_factors=[1.0, 1.1, ..., 2.0])`.

---

## Command line

Installing the package provides a `sparta` command that processes a CSV of reactions and writes a results CSV.

### Input file format

A CSV with at minimum these two columns:

```
reactant_smiles,product_smiles
[C:1]([H:2])([H:3])([H:4])[H:5].[F:6][F:7],[C:1]([H:2])([H:3])([H:4])[F:6].[H:5][F:7]
[C:1]([H:3])([H:4])=[C:2]([H:5])[H:6].[H:7][Cl:8],[C:1]([H:3])([H:4])([H:7])[C:2]([H:5])([H:6])[Cl:8]
```

As above, every atom carries a map number and hydrogens are explicit.

Additional columns are preserved in the output.

### Basic usage

```bash
# Defaults
sparta reactions.csv

# Specify output file
sparta reactions.csv -o results.csv

# Override hyperparameters via flags
sparta reactions.csv \
    --n-conformers 10 --n-seeds 3 \
    --stretch-factors 1.3 1.5 1.8 \
    --top-k 5 --idpp-objective peak

# Load hyperparameters from a JSON config
sparta reactions.csv --config config.json

# Compute SPARTA barriers rather than geometric scores alone
sparta reactions.csv --calculator xtb

# Save the best aligned conformer pair as XYZ files
sparta reactions.csv --save-conformers --conformer-dir ./conformers
```

### Config file format

CLI flags override config file values. All keys are optional.

```json
{
    "n_conformers": 10,
    "n_seeds": 3,
    "stretch_factors": [1.3, 1.5, 1.8],
    "top_k": 5,
    "n_images": 15,
    "neb_fmax": 0.001,
    "neb_steps": 200,
    "propagation_threshold": 0.0001,
    "propagation_step_size": 2.0,
    "max_iterations": 40,
    "objective": "peak",
    "d_power": 4
}
```

### Output

The output CSV contains all original columns plus:

| Column | Description |
|--------|-------------|
| `status` | `ok`, `no_results`, or `error: <message>` |
| `best_score` | IDPP score of the best aligned conformer (lower = better) |
| `all_scores` | Semicolon-separated scores for all aligned conformers |
| `barrier_kcal` | SPARTA barrier in kcal/mol, present when `--calculator` is not `none` |

### All CLI flags

```
positional arguments:
  input                       CSV file with reactant_smiles and product_smiles columns

options:
  -o, --output FILE           Output CSV (default: <input>_results.csv)
  -c, --config FILE           JSON config file for hyperparameters

  --n-conformers INT          Conformers per (seed, stretch_factor) [default: 10]
  --n-seeds INT               Number of random seeds [default: 1]
  --stretch-factors SF [...]  Stretch factor(s) [default: 1.5]
  --top-k INT                 Top-k conformers to align [default: 5]
  --n-images INT              NEB images [default: 15]

  --neb-fmax FLOAT            NEB force convergence [default: 0.001]
  --neb-steps INT             Max NEB steps per run [default: 200]
  --propagation-threshold F   Endpoint force convergence [default: 0.0001]
  --propagation-step-size F   Initial endpoint step size Å [default: 2.0]
  --max-iterations INT        Max NEB+propagation iterations [default: 40]
  --d-power INT               IDPP distance exponent [default: 4]

  --save-conformers           Save best XYZ conformer pair per reaction
  --conformer-dir DIR         Directory for XYZ files [default: conformers]
  --verbose                   Print per-conformer optimization progress
```

---

## License

MIT, see [LICENSE](LICENSE).
