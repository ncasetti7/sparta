"""Batch reaction alignment from a SMILES input file.

Installed as the ``sparta`` console script, so the examples below assume it is
on PATH after ``pip install sparta-chem``.

Input file format (CSV, one reaction per row):
    reactant_smiles,product_smiles
    [CH4:1].[F:2][F:3],[CH3:1][F:2].[H:4][F:3]
    ...

Usage examples::

    # Minimal -- hyperparameters from defaults
    sparta reactions.csv

    # With hyperparameters as flags
    sparta reactions.csv \\
        --n-conformers 10 --n-seeds 3 \\
        --stretch-factors 1.3 1.5 1.8 --top-k 5 \\
        --output results.csv

    # Load hyperparameters from a JSON config file
    sparta reactions.csv --config config.json
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# XYZ helpers
# ---------------------------------------------------------------------------

def _write_xyz(path: str, symbols: list, coords, title: str = "") -> None:
    """Write a single geometry to an XYZ file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(symbols)}\n{title}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            fh.write(f"{sym:4s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def _save_conformers(reaction, out_dir: str, reaction_idx: int) -> None:
    """Save the best aligned reactant/product pair as XYZ files."""
    if not reaction.aligned_results:
        return
    best = reaction.aligned_results[0]
    symbols = [
        reaction.reactant_mol.GetAtomWithIdx(i).GetSymbol()
        for i in range(reaction.reactant_mol.GetNumAtoms())
    ]
    os.makedirs(out_dir, exist_ok=True)
    _write_xyz(
        os.path.join(out_dir, f"rxn{reaction_idx:04d}_reactant.xyz"),
        symbols,
        best["reactant_coords"],
        title=f"reaction {reaction_idx} reactant  score={best['score']:.6f}",
    )
    _write_xyz(
        os.path.join(out_dir, f"rxn{reaction_idx:04d}_product.xyz"),
        symbols,
        best["product_coords"],
        title=f"reaction {reaction_idx} product  score={best['score']:.6f}",
    )


# ---------------------------------------------------------------------------
# Per-reaction worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_reaction(task: tuple) -> tuple:
    """
    Process a single reaction end-to-end.

    Parameters
    ----------
    task : tuple
        (idx, row, cfg, alignment_kwargs, save_conformers, conformer_dir,
         find_optimal_stretch, stretch_start, stretch_step, max_stretch,
         n_workers, total, calc_spec)

    Returns
    -------
    (idx, output_row, log_lines)
    """
    (idx, row, cfg, alignment_kwargs, save_conformers, conformer_dir,
     find_optimal_stretch, stretch_start, stretch_step, max_stretch,
     n_workers, total, calc_spec) = task

    # built inside the worker: a live calculator does not need to be pickled
    calculator = None
    if calc_spec and calc_spec.get("kind") == "xtb":
        from sparta import XTBCalculation
        calculator = XTBCalculation(xtb_cmd=calc_spec.get("xtb_cmd", "xtb"))

    from sparta import (
        Reaction,
        generate_initial_conformers,
        run_iterative_alignment,
        find_optimal_stretch_factor,
    )

    reactant_smiles = row["reactant_smiles"].strip()
    product_smiles = row["product_smiles"].strip()
    log = [f"  [{idx + 1}/{total}] {reactant_smiles} >> {product_smiles}"]

    try:
        rxn = Reaction(
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            n_conformers=cfg["n_conformers"],
            n_seeds=cfg["n_seeds"],
            stretch_factors=cfg["stretch_factors"],
            top_k=cfg["top_k"],
            n_images=cfg["n_images"],
        )

        if find_optimal_stretch:
            rxn = find_optimal_stretch_factor(
                rxn,
                start=stretch_start,
                step=stretch_step,
                max_stretch=max_stretch,
                calculator=calculator,
                **alignment_kwargs,
            )
        else:
            rxn = generate_initial_conformers(rxn, n_workers=n_workers)

            if not rxn.top_k_conformers:
                raise RuntimeError("No conformers generated.")

            rxn = run_iterative_alignment(
                rxn, calculator=calculator, **alignment_kwargs
            )

        if rxn.aligned_results:
            best_score = rxn.aligned_results[0]["score"]
            all_scores = [r["score"] for r in rxn.aligned_results]
            barrier = rxn.aligned_results[0].get("barrier_kcal")
            status = "ok"
        else:
            best_score = None
            all_scores = []
            barrier = None
            status = "no_results"

        if save_conformers and rxn.aligned_results:
            _save_conformers(rxn, conformer_dir, idx)

    except Exception as exc:
        log.append(f"    ERROR: {exc}")
        best_score = None
        all_scores = []
        barrier = None
        status = f"error: {exc}"

    log.append(f"    score = {best_score}"
               + ("" if barrier is None else f"  barrier = {barrier:.2f} kcal/mol"))

    out = dict(row)
    out["status"] = status
    out["best_score"] = "" if best_score is None else f"{best_score:.8f}"
    out["all_scores"] = "" if not all_scores else ";".join(f"{s:.8f}" for s in all_scores)
    if barrier is not None and barrier == barrier:
        out["barrier_kcal"] = f"{barrier:.4f}"
    elif calc_spec:
        out["barrier_kcal"] = ""

    return idx, out, log


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch reactant-product alignment from a SMILES CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("input", help="CSV file with columns: reactant_smiles, product_smiles")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output CSV file. Defaults to <input_stem>_results.csv.",
    )
    p.add_argument(
        "--config", "-c", default=None,
        help="JSON config file for hyperparameters. CLI flags override config values.",
    )

    # Conformer generation hyperparameters
    p.add_argument("--n-conformers", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument(
        "--stretch-factors", type=float, nargs="+", default=None,
        metavar="SF",
        help="One or more stretch factors, e.g. --stretch-factors 1.3 1.5 1.8",
    )
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--n-images", type=int, default=None)

    # Iterative alignment hyperparameters
    p.add_argument("--neb-fmax", type=float, default=None)
    p.add_argument("--neb-steps", type=int, default=None)
    p.add_argument("--propagation-threshold", type=float, default=None)
    p.add_argument("--propagation-step-size", type=float, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument(
        "--idpp-objective", dest="objective",
        choices=["peak", "total"], default=None,
        help="which IDPP quantity alignment minimizes: 'peak' the highest "
             "interior image, 'total' the sum. Not the electronic barrier.",
    )
    p.add_argument("--d-power", type=int, default=None)

    # Parallelism
    p.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel worker processes for scoring and alignment within each reaction.",
    )
    p.add_argument(
        "--n-reaction-workers", type=int, default=1,
        help=(
            "Number of reactions to process in parallel. "
            "Each reaction still uses --n-workers for internal parallelism. "
            "Note: setting both >1 will oversubscribe CPUs."
        ),
    )

    # Stretch-factor search
    p.add_argument(
        "--find-optimal-stretch", action="store_true",
        help=(
            "Search for the optimal stretch factor instead of using a fixed "
            "--stretch-factors list. Starts at --stretch-start and increments "
            "by --stretch-step until the alignment score stops improving."
        ),
    )
    p.add_argument("--stretch-start", type=float, default=1.0,
                   help="Starting stretch factor for --find-optimal-stretch [default: 1.0].")
    p.add_argument("--stretch-step", type=float, default=0.1,
                   help="Increment for --find-optimal-stretch [default: 0.1].")
    p.add_argument("--max-stretch", type=float, default=3.0,
                   help="Upper bound on stretch factor for --find-optimal-stretch [default: 3.0].")

    # Conformer saving
    p.add_argument(
        "--calculator", choices=["none", "xtb"], default="none",
        help="electronic energies for the SPARTA barrier. 'none' reports the "
             "geometric IDPP score only.",
    )
    p.add_argument(
        "--xtb-cmd", default="xtb",
        help="path to the xtb executable when --calculator xtb is used.",
    )
    p.add_argument(
        "--save-conformers", action="store_true",
        help="Save the best aligned conformer pair as XYZ files for each reaction.",
    )
    p.add_argument(
        "--conformer-dir", default="conformers",
        help="Directory for saved XYZ files (created if absent).",
    )

    p.add_argument("--verbose", action="store_true", help="Print per-conformer progress.")

    return p


def _merge_config(args: argparse.Namespace) -> dict:
    """
    Return a single dict of hyperparameters, with JSON config as base and
    CLI flags taking precedence wherever provided.
    """
    defaults = {
        "n_conformers": 10,
        "n_seeds": 1,
        "stretch_factors": 1.5,
        "top_k": 5,
        "n_images": 15,
        "neb_fmax": 0.001,
        "neb_steps": 200,
        "propagation_threshold": 0.0001,
        "propagation_step_size": 2.0,
        "max_iterations": 40,
        "objective": "peak",
        "d_power": 4,
    }

    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            file_cfg = json.load(fh)
        defaults.update(file_cfg)

    # CLI overrides (only when explicitly set, i.e. not None)
    cli_map = {
        "n_conformers": args.n_conformers,
        "n_seeds": args.n_seeds,
        "stretch_factors": args.stretch_factors,
        "top_k": args.top_k,
        "n_images": args.n_images,
        "neb_fmax": args.neb_fmax,
        "neb_steps": args.neb_steps,
        "propagation_threshold": args.propagation_threshold,
        "propagation_step_size": args.propagation_step_size,
        "max_iterations": args.max_iterations,
        "objective": args.objective,
        "d_power": args.d_power,
    }
    for key, val in cli_map.items():
        if val is not None:
            defaults[key] = val

    return defaults


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _merge_config(args)

    # Resolve output path
    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + "_results.csv"))

    # Read input CSV
    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "reactant_smiles" not in (reader.fieldnames or []) or \
                "product_smiles" not in (reader.fieldnames or []):
            sys.exit(
                "Error: input CSV must have 'reactant_smiles' and 'product_smiles' columns."
            )
        rows = list(reader)

    n_reaction_workers = args.n_reaction_workers
    if args.n_workers > 1 and n_reaction_workers > 1:
        print(
            f"Warning: --n-workers={args.n_workers} and --n-reaction-workers={n_reaction_workers} "
            "are both >1; this will oversubscribe CPUs.",
            flush=True,
        )

    print(
        f"Processing {len(rows)} reaction(s) with {n_reaction_workers} reaction worker(s) …",
        flush=True,
    )

    alignment_kwargs = dict(
        neb_fmax=cfg["neb_fmax"],
        neb_steps=cfg["neb_steps"],
        propagation_threshold=cfg["propagation_threshold"],
        propagation_step_size=cfg["propagation_step_size"],
        max_iterations=cfg["max_iterations"],
        objective=cfg["objective"],
        d_power=cfg["d_power"],
        verbose=args.verbose,
        n_workers=args.n_workers,
    )

    tasks = [
        (
            idx, row, cfg, alignment_kwargs,
            args.save_conformers, args.conformer_dir,
            args.find_optimal_stretch, args.stretch_start, args.stretch_step, args.max_stretch,
            args.n_workers, len(rows),
            (None if args.calculator == "none"
             else {"kind": args.calculator, "xtb_cmd": args.xtb_cmd}),
        )
        for idx, row in enumerate(rows)
    ]

    results = []
    if n_reaction_workers > 1:
        with multiprocessing.Pool(n_reaction_workers) as pool:
            for idx, out, log in pool.imap_unordered(_process_reaction, tasks):
                for msg in log:
                    print(msg, flush=True)
                results.append((idx, out))
    else:
        for task in tasks:
            idx, out, log = _process_reaction(task)
            for msg in log:
                print(msg, flush=True)
            results.append((idx, out))

    results.sort(key=lambda x: x[0])
    output_rows = [out for _, out in results]

    # Write output CSV
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\nResults written to {output_path}")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
