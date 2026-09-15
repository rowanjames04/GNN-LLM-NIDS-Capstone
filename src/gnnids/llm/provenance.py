"""What a report was generated *on*, recorded at run time (D35).

A cloud model is identified by its API model id. A self-hosted result is not:
an Ollama tag like `llama3.1:8b` is re-pointed whenever the model is
re-quantised or re-published, and a free Colab runtime is assigned whatever GPU
is available that day. Neither can be recovered after the fact, so both are
captured when the run starts -- and again on every resume, because a resumed
Colab session can land on different hardware.

Every probe here is best-effort: a missing `nvidia-smi` on the Mac or an
unreachable git binary records `None`, never an exception. Provenance that
crashes a run is worse than provenance that says "unknown".
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def gpus() -> list[dict] | None:
    """NVIDIA GPUs visible to this process, or None where there are none."""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    if not out:
        return None
    rows = []
    for line in out.splitlines():
        name, mem, driver = (part.strip() for part in line.split(",", 2))
        rows.append({"name": name, "memory": mem, "driver": driver})
    return rows


def git_commit(repo: Path) -> dict:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    return {"commit": commit,
            # A dirty tree means the commit alone does not reproduce the run.
            "dirty": bool(status) if commit else None}


def session(repo: Path) -> dict:
    """The machine and code a run (or a resumed part of one) executed on."""
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": gpus(),
    }
