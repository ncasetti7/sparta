# Stereochemistry Handling in SPARTA

This document describes how SPARTA represents, checks, and enforces
stereochemistry while generating reactant-product conformer pairs for
transition-state search seeding.

SPARTA uses atom-mapped reactant and product SMILES as the source of truth for
reaction topology. Those atom maps are essential for matching coordinates across
the reaction, but they must not be allowed to change chemical stereochemical
priority. The stereochemistry code therefore treats atom maps as bookkeeping and
computes stereochemical reference labels on map-stripped molecule copies.

## Design Goals

- Preserve the stereochemistry requested by the reactant and product SMILES.
- Use chemically meaningful ligand priority, not atom ordering or neighbor
  iteration order.
- Keep atom indices stable for SPARTA coordinates: atom index = atom-map number
  minus one.
- Reject or repair conformers that flip specified stereocenters before they feed
  transition-state searches.
- Keep diagnostics off by default, with explicit opt-in logging for debugging.
- Centralize stereochemistry code under `sparta/stereochemistry/` so additional
  stereochemical element types can be added without scattering logic through the
  pipeline.

## Current Scope

Implemented now:

- Tetrahedral stereocenters with defined SMILES stereochemistry.
- Reference extraction from independently embedded reactant and product graphs.
- Signed-volume constraints for initial product conformer relaxation.
- Conservative branch reflection to repair flipped tetrahedral centers.
- Iterative-alignment guards that reject endpoint steps which invert recorded
  tetrahedral signed volumes.
- Opt-in debug logging through `Reaction(..., stereo_debug=True)` or
  `SPARTA_STEREO_DEBUG=1`.

Not implemented yet:

- E/Z alkene reference extraction and enforcement.
- Atropisomeric axis reference extraction and enforcement.
- Square planar, trigonal bipyramidal, octahedral, or other non-tetrahedral
  stereochemical elements.
- Full stereochemical validation of TS or IRC calculations after SPARTA hands off
  conformer seeds.

The package layout intentionally leaves room for these future elements. New
reference types should live in `sparta/stereochemistry/` and expose the same
kind of operations: extract reference from graph plus coordinates, compute a
coordinate descriptor, validate current coordinates, and provide optimizer or
step-rejection constraints when possible.

## Source of Truth

Use both reactant and product SMILES as independent stereochemical references.

- Reactant references come from the reactant SMILES and a reactant reference
  conformer.
- Product references come from the product SMILES and a product reference
  conformer.
- Product conformers created by pushing forward reactant coordinates must match
  the product references, not the reactant references.

This matters because the reaction can change CIP priority. A center can keep the
same physical handedness while its R/S label changes, or invert physically while
its label appears unchanged, if substituent priorities differ between reactant
and product graphs. Therefore SPARTA stores separate reactant and product
references on `Reaction`:

```python
rxn.reactant_stereo_refs
rxn.product_stereo_refs
```

These are built in `sparta/reaction.py` using
`tetrahedral_stereo_references(...)`.

## Atom Maps and CIP Labels

Atom maps are not chemistry. They are reaction bookkeeping.

RDKit can allow atom-map labels on explicit-H mapped molecules to affect CIP
diagnostics. For example, the same local SMILES winding can produce opposite
reported `_CIPCode` values before and after map labels are removed:

```python
from rdkit import Chem
import re

mapped = (
    "[O:39]=[C:40]([O:41][C:42]([H:90])([H:91])[H:92])"
    "[C:43](=[O:44])[C@:48]([C:47]([C:46](=[C:45]"
    "([H:93])[H:94])[H:95])([H:96])[H:97])"
    "([C:49]([H:99])([H:100])[H:101])[H:98]"
)
unmapped = re.sub(r":\d+", "", mapped)

ps = Chem.SmilesParserParams()
ps.removeHs = False

for smiles in (mapped, unmapped):
    mol = Chem.MolFromSmiles(smiles, ps)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    print([(a.GetIdx(), a.GetProp("_CIPCode"))
           for a in mol.GetAtoms() if a.HasProp("_CIPCode")])
```

In this case the mapped molecule reports `R`, while the map-stripped molecule
reports `S`. SPARTA should enforce the chemically meaningful map-stripped
interpretation while keeping mapped atom indices for coordinates.

The helper `_unmapped_stereo_copy(...)` in
`sparta/stereochemistry/constraints.py` makes a molecule copy and clears
`atom.SetAtomMapNum(0)` before assigning stereochemistry. Reference extraction
then uses the map-stripped copy for CIP labels and priorities, but the atom
indices remain aligned with the original molecule because only map numbers are
changed.

## Explicit Hydrogens

SPARTA reactions commonly move hydrogens. Those hydrogens must be explicit and
mapped on both reactant and product sides.

Do not use raw `Chem.MolFromSmiles(smiles)` for SPARTA reaction validation if
explicit hydrogens matter. RDKit's default parser removes many explicit
hydrogens into implicit-H counts. Use:

```python
ps = Chem.SmilesParserParams()
ps.removeHs = False
mol = Chem.MolFromSmiles(smiles, ps)
```

Inside SPARTA, prefer `sparta.cheminformatics.make_mol(smiles)`. It preserves
explicit hydrogens and renumbers atoms so index `i` corresponds to atom-map
number `i + 1`.

## Tetrahedral Reference Model

The current tetrahedral reference is represented by
`TetrahedralStereoReference`:

```python
center: int
neighbors: tuple[int, int, int]
priority_neighbors: tuple[int, ...]
priority_ranks: tuple[int, ...]
label: str
sign: float
volume: float
margin: float
```

Reference extraction:

1. Copy the molecule and clear atom maps.
2. Assign stereochemistry on the map-stripped copy.
3. Skip atoms with unspecified chirality or no `_CIPCode`.
4. Sort explicit neighbors by descending RDKit `_CIPRank`.
5. Use the three highest-priority explicit neighbors to define a signed volume.
6. Store the sign and label from the reference conformer.

The signed volume is:

```python
V = (r_i - r_c) dot ((r_j - r_c) cross (r_k - r_c))
```

where `c` is the stereocenter and `i, j, k` are the three highest-priority
explicit neighbors. Once the priority order is fixed, maintaining the sign of
`V` maintains the handedness of that center.

This is why the code never uses arbitrary neighbor order to define chirality.
Neighbor order can change with parsing, renumbering, or graph edits. Priority
order is the chemically meaningful choice.

## Initial Generation

Initial conformer generation happens in
`sparta/initial_generation/bounds_matrix.py`.

Relevant steps:

1. RDKit embeds reactant conformers with `embed_params.enforceChirality = True`.
2. RDKit embeds one product reference conformer with the same chirality setting.
3. Product stereo references are taken from `reaction.product_stereo_refs`, not
   inferred from each pushed-forward conformer.
4. Reactant coordinates are pushed forward into product topology with
   `transform_mol(...)`.
5. Product coordinates are relaxed to the product bounds matrix with RMSD tethering
   to the paired reactant.
6. Tetrahedral signed-volume penalties are passed as `chiral_constraints`.
7. If the relaxed product mismatches the target signed volumes, SPARTA attempts a
   conservative branch reflection using `correct_tetrahedral_stereochemistry(...)`.
8. The corrected geometry is relaxed again with the same chiral constraints.
9. Product conformers that still mismatch are rejected.

The chiral penalty lives in
`sparta/initial_generation/dg_optimizer.py` and has the form:

```text
weight_chiral * max(0, margin - target_sign * V)^2
```

This creates a barrier as a center approaches or crosses the wrong signed-volume
side. It does not replace full chemical force-field stereochemistry, but it gives
the distance-geometry optimizer a coordinate-level reason not to pass through a
planar or inverted center.

## Iterative Alignment

Iterative alignment happens in
`sparta/iterative_alignment/optimization.py`.

IDPP and distance-bounds endpoint propagation are mostly geometric. They do not
know that changing the signed volume at a tetrahedral center is chemically
forbidden. SPARTA therefore uses a guard:

1. At alignment start, load `reaction.reactant_stereo_refs` and
   `reaction.product_stereo_refs`.
2. After each proposed endpoint adjustment, compute current signed volumes.
3. If any reactant or product reference flips sign, reject the step.
4. Restore the previous endpoints and backtrack to a smaller step size.

This guard prevents an endpoint adjustment from locking in a stereochemical flip
while still allowing IDPP to improve the path geometry.

## Debugging

Stereo logging is off by default. Enable it either way:

```python
rxn = Reaction(reactant_smiles=reactant, product_smiles=product, stereo_debug=True)
```

or:

```bash
SPARTA_STEREO_DEBUG=1 python your_script.py
```

Debug output includes:

- the number of tetrahedral references,
- atom index,
- target label,
- signed-volume neighbors,
- priority neighbors and ranks,
- reference sign and volume,
- current volume,
- match or mismatch status.

Use this to find the first stage where a center changes sign:

- `reactant reference` / `product reference`: target construction.
- `reactant conf=...`: RDKit reactant embedding.
- `product pushforward...`: topology pushforward from reactant to product.
- `product first relax...`: distance-geometry plus RMSD relaxation.
- `product corrected...`: branch reflection repair.
- `product corrected relax...`: final repaired relaxation.
- `alignment start...`: endpoints entering IDPP alignment.
- `alignment rejected...`: endpoint propagation step rejected for a flip.

## Validating Coordinates

Avoid using XYZ plus `rdDetermineBonds.DetermineBonds(...)` as the primary
stereo validator for SPARTA outputs. XYZ has no bond order, atom-map, or stereo
metadata. Reconstructed molecules can have different bond insertion order and
different stereochemical diagnostics even when the coordinate array is the same.

Prefer validation against the known graph:

```python
from rdkit import Chem
from sparta.cheminformatics import make_mol, add_coords_to_mol

def assign_from_3d_on_known_graph(smiles, coords):
    mol = make_mol(smiles)
    mol = add_coords_to_mol(coords, mol)

    # Use a copy for assignment so the input graph remains untouched.
    probe = Chem.Mol(mol)
    Chem.RemoveStereochemistry(probe)

    # Optional for chemical CIP diagnostics: ignore reaction maps.
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(0)

    Chem.AssignStereochemistryFrom3D(probe, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)

    return [
        (atom.GetIdx(), atom.GetSymbol(), atom.GetProp("_CIPCode"))
        for atom in probe.GetAtoms()
        if atom.HasProp("_CIPCode")
    ]
```

For SPARTA internal enforcement, prefer the signed-volume helpers:

```python
from sparta.stereochemistry import stereo_matches

stereo_matches(coords, rxn.product_stereo_refs)
```

## Drawing Molecules

When visually inspecting conformers, prefer drawing an SDF/MolBlock generated
from the known graph plus coordinates instead of an XYZ file reconstructed by
bond perception:

```python
from rdkit import Chem
from sparta.cheminformatics import make_mol, add_coords_to_mol

mol = make_mol(product_smiles)
mol = add_coords_to_mol(product_coords, mol)
block = Chem.MolToMolBlock(mol)

view.addModel(block, "sdf")
view.setStyle({"stick": {}})
view.zoomTo()
```

XYZ is still useful for quantum chemistry handoff, but it is a weak format for
debugging stereochemical identity.

## Transition-State and IRC Searches

SPARTA conformers are TS search seeds. They are not a guarantee that downstream
optimizers or IRC calculations will preserve stereochemistry.

In practice, cheap molecular mechanics and quantum optimizers usually will not
flip a well-defined tetrahedral center, E/Z alkene, or atropisomer unless the
path crosses a high barrier or the geometry is poorly constrained. However, a TS
search explores a saddle region and may alter bonding, planarity, or conformer
identity. IRC follows the potential-energy surface from the located TS; it does
not know that SPARTA intended a particular stereochemical endpoint unless the
surface and the initial TS geometry lead there.

Recommended downstream policy:

- Validate the optimized TS geometry against the intended reactant and product
  stereochemical references where meaningful.
- Run IRC in both directions.
- Assign endpoints on known reactant/product graphs, not from XYZ bond perception
  alone.
- Reject TS results whose IRC endpoints do not match the intended reactant and
  product stereochemistry.
- For reactions where stereochemistry changes at the reacting center, validate
  against the correct endpoint graph on each side rather than requiring one
  global R/S label throughout.

## Extending Beyond Tetrahedral Centers

Add new stereochemical element types under `sparta/stereochemistry/` using the
same pattern as tetrahedral references.

For E/Z alkenes:

- Extract double bonds with defined stereo from the map-stripped graph.
- Identify the highest-priority substituent on each alkene atom.
- Store a signed dihedral or side-of-plane descriptor.
- Enforce sign during product relaxation and reject sign-changing alignment steps.

For atropisomerism:

- Define the stereogenic axis and the priority substituent on each side.
- Store a signed torsion descriptor or other robust axis descriptor.
- Add barriers or guards against crossing through the interconversion geometry.
- Be careful with cases where the axis is only configurationally stable under
  specific steric or electronic conditions.

For all new reference types:

- Build references independently for reactant and product graphs.
- Strip atom maps before assigning chemical priority.
- Keep original atom indices for coordinate descriptors.
- Add tests with atom-mapped examples where mapped and unmapped RDKit diagnostics
  differ.
- Add debug logging that identifies the atom or bond indices, target descriptor,
  current descriptor, and match status.

## Tests

Stereo behavior is covered by `tests/test_stereochemistry.py`.

Run:

```bash
python -m pytest tests/test_stereochemistry.py -q
python -m pytest tests/ -q
```

The regression test `test_atom_maps_do_not_change_target_cip_priority` captures
the mapped-product case where raw RDKit `_CIPCode` on the atom-mapped molecule
reports the opposite label from the chemically map-stripped interpretation.
