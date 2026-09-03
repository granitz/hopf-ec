"""Build and run an fMRIPrep command (Docker / Apptainer / Singularity / native install).

fMRIPrep is the state-of-the-art, robust BIDS-App for fMRI preprocessing (Esteban et al. 2019).
We only assemble the command line from the configuration; nothing is hard-coded to a machine.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from ..utils import LOG, ensure_dir, which


def detect_runtime(preferred: str = "auto", sif: str | None = None) -> str:
    if preferred != "auto":
        return preferred
    if which("fmriprep"):
        return "native"
    if which("fmriprep-docker"):
        return "fmriprep-docker"
    if sif and which("apptainer"):
        return "apptainer"
    if sif and which("singularity"):
        return "singularity"
    if which("docker"):
        return "docker"
    if which("apptainer"):
        return "apptainer"
    if which("singularity"):
        return "singularity"
    raise RuntimeError("no fMRIPrep runtime found (install Docker, Apptainer/Singularity or fmriprep itself)")


def build_fmriprep_command(bids_dir: str | Path, output_dir: str | Path, participants: list[str] | None, cfg: dict,
                           work_dir: str | Path | None = None, fs_license: str | None = None, runtime: str | None = None) -> list[str]:
    """Return the argv list for one fMRIPrep participant-level run."""
    bids_dir = Path(bids_dir).resolve()
    output_dir = Path(output_dir).resolve()
    work_dir = Path(work_dir).resolve() if work_dir else output_dir.parent / "fmriprep_work"
    fs_license = fs_license or cfg.get("fs_license") or os.environ.get("FS_LICENSE")
    runtime = detect_runtime(runtime or cfg.get("runtime", "auto"), cfg.get("sif"))
    labels = [p.replace("sub-", "") for p in (participants or [])]
    spaces = cfg.get("output_spaces") or ["MNI152NLin2009cAsym:res-2"]
    nprocs = int(cfg.get("nprocs", 8))
    omp = int(cfg.get("omp_nthreads", min(4, nprocs)))
    mem = int(cfg.get("mem_mb", 16000))
    tf_home = cfg.get("templateflow_home") or os.environ.get("TEMPLATEFLOW_HOME")

    app_args = ["participant", "--output-spaces", *spaces, "--nprocs", str(nprocs), "--omp-nthreads", str(omp),
                "--mem-mb", str(mem), "--notrack"]
    if labels:
        app_args += ["--participant-label", *labels]
    if cfg.get("skip_bids_validation", True):
        app_args.append("--skip-bids-validation")
    if cfg.get("fs_no_reconall"):
        app_args.append("--fs-no-reconall")
    app_args += [str(x) for x in (cfg.get("extra_args") or [])]

    if runtime == "native":
        cmd = ["fmriprep", str(bids_dir), str(output_dir), *app_args, "-w", str(work_dir)]
        if fs_license:
            cmd += ["--fs-license-file", str(fs_license)]
        return cmd
    if runtime == "fmriprep-docker":
        cmd = ["fmriprep-docker", str(bids_dir), str(output_dir), *app_args, "-w", str(work_dir)]
        if fs_license:
            cmd += ["--fs-license-file", str(fs_license)]
        return cmd
    if runtime == "docker":
        image = cfg.get("image", "nipreps/fmriprep:24.1.1")
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000",
               "-v", f"{bids_dir}:/data:ro", "-v", f"{output_dir}:/out", "-v", f"{work_dir}:/work"]
        if fs_license:
            cmd += ["-v", f"{Path(fs_license).resolve()}:/opt/freesurfer/license.txt:ro"]
        if tf_home:
            cmd += ["-v", f"{Path(tf_home).resolve()}:/home/fmriprep/.cache/templateflow", "-e", "TEMPLATEFLOW_HOME=/home/fmriprep/.cache/templateflow"]
        cmd += [image, "/data", "/out", *app_args, "-w", "/work"]
        if fs_license:
            cmd += ["--fs-license-file", "/opt/freesurfer/license.txt"]
        return cmd
    if runtime in ("apptainer", "singularity"):
        sif = cfg.get("sif")
        if not sif:
            raise ValueError("fmriprep.sif must point to the fMRIPrep image (.sif) for apptainer/singularity")
        cmd = [runtime, "run", "--cleanenv", "-B", f"{bids_dir}:/data:ro", "-B", f"{output_dir}:/out", "-B", f"{work_dir}:/work"]
        if fs_license:
            cmd += ["-B", f"{Path(fs_license).resolve()}:/opt/freesurfer/license.txt:ro"]
        if tf_home:
            cmd += ["-B", f"{Path(tf_home).resolve()}:/templateflow", "--env", "TEMPLATEFLOW_HOME=/templateflow"]
        cmd += [str(sif), "/data", "/out", *app_args, "-w", "/work"]
        if fs_license:
            cmd += ["--fs-license-file", "/opt/freesurfer/license.txt"]
        return cmd
    raise ValueError(f"unknown runtime {runtime!r}")


def run_fmriprep(bids_dir, output_dir, participants, cfg, work_dir=None, fs_license=None, runtime=None, dry_run=False) -> int:
    cmd = build_fmriprep_command(bids_dir, output_dir, participants, cfg, work_dir, fs_license, runtime)
    LOG.info("fMRIPrep command:\n  %s", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    ensure_dir(output_dir)
    if work_dir:
        ensure_dir(work_dir)
    if not (fs_license or cfg.get("fs_license") or os.environ.get("FS_LICENSE")):
        LOG.warning("no FreeSurfer license given (paths.fs_license / FS_LICENSE); fMRIPrep will fail without it")
    return subprocess.call(cmd)
