import os
import re
import shutil
import subprocess
import tempfile
from typing import Tuple

import ase
import ase.io

from sparta.calculation.base import Calculation


class XTBCalculation(Calculation):
    """
    Runs single-point and optimization calculations via the xtb binary using
    the GFN-xTB method (--gxtb flag).

    Each call writes a temporary XYZ file, invokes xtb in a scratch directory,
    parses the output, then removes the scratch directory.

    Parameters
    ----------
    charge : int
        Total molecular charge passed to xtb via --chrg.
    multiplicity : int
        Spin multiplicity; the number of unpaired electrons (multiplicity - 1)
        is passed via --uhf.
    xtb_cmd : str
        Path or name of the xtb executable (default: "xtb").
    """

    def __init__(
        self,
        charge: int = 0,
        multiplicity: int = 1,
        xtb_cmd: str = "xtb",
        timeout: int = 120,
    ) -> None:
        super().__init__(charge, multiplicity)
        self._xtb_cmd = xtb_cmd
        self._timeout = timeout

    def _run_xtb(self, tmpdir: str, extra_args: list) -> str:
        cmd = [
            self._xtb_cmd, "input.xyz",
            "--gxtb",
            "--chrg", str(self.charge),
            "--uhf", str(self.multiplicity - 1),
        ] + extra_args
        # xtb is an external program, not a Python dependency, so a missing
        # binary is the single most likely first-run failure. Say so plainly
        # rather than letting FileNotFoundError surface from subprocess.
        if shutil.which(self._xtb_cmd) is None and not os.path.isfile(self._xtb_cmd):
            raise FileNotFoundError(
                f"xtb executable {self._xtb_cmd!r} not found on PATH. SPARTA "
                "does not bundle a quantum-chemistry program; install xtb "
                "(https://github.com/grimme-lab/xtb) and make sure it is on "
                "PATH, or pass xtb_cmd with an explicit path. Only the "
                "calculator-backed features need it -- path generation and "
                "alignment do not."
            )
        try:
            result = subprocess.run(
                cmd, cwd=tmpdir, capture_output=True, text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"xtb timed out after {self._timeout}s"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"xtb exited with code {result.returncode}:\n{result.stderr}"
            )
        return result.stdout

    def _parse_energy(self, stdout: str) -> float:
        for line in stdout.splitlines():
            if "TOTAL ENERGY" in line:
                match = re.search(r"(-?\d+\.\d+)\s+Eh", line)
                if match:
                    return float(match.group(1))
        raise ValueError("Could not parse TOTAL ENERGY from xtb output")

    def energy(self, atoms: ase.Atoms) -> float:
        tmpdir = tempfile.mkdtemp()
        try:
            ase.io.write(os.path.join(tmpdir, "input.xyz"), atoms, format="xyz")
            stdout = self._run_xtb(tmpdir, ["--sp"])
            return self._parse_energy(stdout)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def optimize(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, float]:
        tmpdir = tempfile.mkdtemp()
        try:
            ase.io.write(os.path.join(tmpdir, "input.xyz"), atoms, format="xyz")
            stdout = self._run_xtb(tmpdir, ["--opt"])
            energy = self._parse_energy(stdout)
            optimized = ase.io.read(os.path.join(tmpdir, "xtbopt.xyz"), format="xyz")
            return optimized, energy
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
