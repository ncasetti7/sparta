<p align="center">
  <img src="sparta_logo.png" alt="SPARTA logo" width="400"/>
</p>

# SPARTA

**S**teric **P**ath **A**lignment for **R**eactive **T**ransformation **A**ssessment

SPARTA generates reactant–product conformer pairs and iteratively aligns them using an Image-Dependent Pair Potential (IDPP) nudged-elastic-band (NEB) path optimisation. The resulting IDPP barrier score provides a fast, geometry-based proxy for kinetic feasibility — reactions with lower scores have less steric clash along the path connecting reactant and product.

---

## Installation

### Default (pip)

```bash
pip install sparta
```

> SPARTA will be available on PyPI. Until then, use the development install below.

### Development install (clone from repo)

```bash
git clone https://github.com/your-org/sparta.git
cd sparta
pip install .
```

Or with [uv](https://docs.astral.sh/uv/) for a fully reproducible environment:

```bash
git clone https://github.com/your-org/sparta.git
cd sparta
uv sync
```

#### Requirements

| Package | Minimum version |
|---------|----------------|
| Python  | 3.12           |
| numpy   | 1.24           |
| scipy   | 1.16           |
| rdkit   | 2025.9.1       |
| ase     | 3.26           |

---

## Python API

### Quick start

```python
from sparta import Reaction, generate_initial_conformers, run_iterative_alignment

rxn = Reaction(
    reactant_smiles="[CH4:1].[F:2][F:3]",
    product_smiles="[CH3:1][F:2].[H:4][F:3]",
    n_conformers=10,
    n_seeds=3,
    stretch_factors=[1.3, 1.5, 1.8],
    top_k=5,
)

# Functional style — each function returns the reaction object
rxn = generate_initial_conformers(rxn)
rxn = run_iterative_alignment(rxn)

# Method-chaining style (equivalent)
rxn = Reaction(...).generate_initial_conformers().run_iterative_alignment()
```

### Accessing results

```python
# aligned_results is sorted by score (ascending — lower is better)
best = rxn.aligned_results[0]

print(best["score"])            # float: max interior IDPP energy
print(best["reactant_coords"])  # np.ndarray (n_atoms, 3)
print(best["product_coords"])   # np.ndarray (n_atoms, 3)
print(best["images"])           # list of ase.Atoms — full NEB path
print(best["energies"])         # list of float — per-image IDPP energies
print(best["seed"])             # int
print(best["stretch_factor"])   # float
```

All intermediate conformers are also available:

```python
rxn.all_conformers      # every generated conformer pair, sorted by initial NEB score
rxn.top_k_conformers    # the top-k subset passed to iterative alignment
```

### `Reaction` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reactant_smiles` | `str` | — | Atom-mapped SMILES for the reactant(s) |
| `product_smiles` | `str` | — | Atom-mapped SMILES for the product(s) |
| `n_conformers` | `int` | `10` | Conformers per (seed, stretch_factor) combination |
| `n_seeds` | `int` | `1` | Number of random seeds (0 … n_seeds−1) |
| `stretch_factors` | `float` or `list` | `1.5` | Stretch factor(s) applied to changing bond targets in the bounds matrix |
| `top_k` | `int` | `5` | Number of best initial conformers to carry into iterative alignment |
| `n_images` | `int` | `15` | NEB images (including endpoints) |
| `filter_dihedrals` | `bool` | `True` | Restrict optimisation to dihedrals near the reaction centre |
| `k_nearest_dihedrals` | `int` | `3` | Bond-distance cutoff for dihedral filtering |
| `cutoff` | `float` | `10.0` | Neighbour-pair cutoff (Å) for overlap evaluation |
| `rotation` | `bool` | `True` | Include fragment rotations |
| `translation` | `bool` | auto | Include fragment translations (default: `True` when multiple fragments exist) |

### `run_iterative_alignment` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neb_fmax` | `0.001` | Force convergence criterion for each inner NEB run |
| `neb_steps` | `200` | Maximum steps per inner NEB run |
| `propagation_threshold` | `0.0001` | Stop when max propagated endpoint force < this value |
| `propagation_step_size` | `2.0` | Initial endpoint adjustment step size (Å) |
| `max_iterations` | `40` | Maximum NEB + propagation iterations per conformer |
| `objective` | `"barrier"` | `"barrier"` minimises the max interior image energy; `"total"` minimises the sum |
| `d_power` | `4` | IDPP distance weighting exponent |
| `verbose` | `False` | Print per-iteration progress |

---

## Batch processing (CLI)

The `workflows/process_reactions_from_file.py` script processes a CSV of reactions and writes a results CSV.

### Input file format

A CSV with at minimum these two columns:

```
reactant_smiles,product_smiles
[CH4:1].[F:2][F:3],[CH3:1][F:2].[H:4][F:3]
[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[Br:7][Br:8],[c:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[Br:7].[H:9][Br:8]
```

Additional columns are preserved in the output.

### Basic usage

```bash
# Defaults
python workflows/process_reactions_from_file.py reactions.csv

# Specify output file
python workflows/process_reactions_from_file.py reactions.csv -o results.csv

# Override hyperparameters via flags
python workflows/process_reactions_from_file.py reactions.csv \
    --n-conformers 10 --n-seeds 3 \
    --stretch-factors 1.3 1.5 1.8 \
    --top-k 5 --objective barrier

# Load hyperparameters from a JSON config
python workflows/process_reactions_from_file.py reactions.csv --config config.json

# Save the best aligned conformer pair as XYZ files
python workflows/process_reactions_from_file.py reactions.csv \
    --save-conformers --conformer-dir ./conformers
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
    "objective": "barrier",
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
  --objective {barrier,total} Optimisation objective [default: barrier]
  --d-power INT               IDPP distance exponent [default: 4]

  --save-conformers           Save best XYZ conformer pair per reaction
  --conformer-dir DIR         Directory for XYZ files [default: conformers]
  --verbose                   Print per-conformer optimisation progress
```
