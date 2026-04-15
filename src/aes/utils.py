"""Utility helpers: seeding, hashing, environment capture.

Reproducibility is enforced by (a) seeding all RNGs, (b) hashing
input data files, and (c) capturing the full software environment
into MLflow as params. Implements peer-review demands 13c-1, 13c-2.
"""
from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch missing in CI / lint env
        pass


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 of a file in streaming mode."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class EnvSnapshot:
    python: str
    torch: str
    cuda: str | None
    driver: str | None
    platform: str
    cpu_count: int

    def as_mlflow_params(self) -> dict[str, str]:
        return {
            "env.python": self.python,
            "env.torch": self.torch,
            "env.cuda": self.cuda or "n/a",
            "env.driver": self.driver or "n/a",
            "env.platform": self.platform,
            "env.cpu_count": str(self.cpu_count),
        }


def capture_env() -> EnvSnapshot:
    """Record software + driver environment for MLflow."""
    try:
        import torch

        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda
    except ImportError:
        torch_ver, cuda_ver = "n/a", None

    driver = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            driver = out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return EnvSnapshot(
        python=sys.version.split()[0],
        torch=torch_ver,
        cuda=cuda_ver,
        driver=driver,
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
    )


def data_hashes(data_dir: str | os.PathLike) -> dict[str, str]:
    """Hash ASAP files per ADR-001-rev1 Annex B."""
    d = Path(data_dir)
    out: dict[str, str] = {}
    mapping = {
        "asap1": "training_set_rel3.tsv",
        "asap2": "ASAP2_train_sourcetexts.csv",
    }
    for tag, fname in mapping.items():
        p = d / fname
        if p.exists():
            out[f"data.hash.{tag}"] = sha256_file(p)
    return out
