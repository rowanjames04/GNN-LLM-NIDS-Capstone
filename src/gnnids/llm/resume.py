"""Append-as-you-go report logging, so a run can survive losing its machine (D35).

`generate_reports.py` used to hold every report in memory and write one file
after the last pack. On the Mac that was fine. The self-hosted arm now runs in a
Google Colab session, which ends at 12 hours at most and sooner when idle -- a
disconnect at pack 180 of 200 would have thrown away 180 generated reports.

So each report is appended to a JSON-lines file the moment it is scored, and a
re-run of the same command skips every detection already on disk. The final
results file is only written once every pack has a row, and the partial log is
then removed.

**A resumed run must be the same run.** The log's first line records the
provider, model, prompt version and pack source. A mismatch is refused rather
than merged: a results file mixing two prompt versions would pass every check
and measure nothing.

**Each session is recorded separately.** A resumed Colab run can land on a
different GPU, so every start appends a session line (hardware, commit, model
build -- see `provenance.py`) and every row carries the index of the session
that generated it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

IDENTITY_KEYS = ("provider", "model", "prompt_version", "packs_source")


class ResumeMismatch(RuntimeError):
    """The partial log on disk belongs to a different run."""


class ReportLog:
    def __init__(self, path: Path, identity: dict) -> None:
        self.path = Path(path)
        self.identity = {k: identity[k] for k in IDENTITY_KEYS}
        self.sessions: list[dict] = []

    def load(self, pack_ids: list[str], retry_failed: bool = False
             ) -> tuple[dict[str, dict], int]:
        """Rows already generated for this run, keyed by detection id.

        Returns the rows to keep and how many failed rows were dropped for
        retry. Retrying is opt-in and counted, because silently regenerating
        failures until they succeed would make a model look more reliable than
        it is. It exists for infrastructure failures -- an Ollama server that
        died mid-run fails every remaining pack in milliseconds.
        """
        if not self.path.exists():
            return {}, 0
        lines = [ln for ln in self.path.read_text().splitlines() if ln.strip()]
        if not lines:
            return {}, 0
        header = json.loads(lines[0]).get("header", {})
        if {k: header.get(k) for k in IDENTITY_KEYS} != self.identity:
            raise ResumeMismatch(
                f"{self.path.name} was written by a different run "
                f"({header}) than this one ({self.identity}). "
                f"Delete it or pass --fresh.")

        rows: dict[str, dict] = {}
        for ln in lines[1:]:
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                # A session killed mid-write leaves at most one torn last line.
                # Dropping it costs one regenerated report, nothing more.
                continue
            if "session" in row and "detection_id" not in row:
                self.sessions.append(row["session"])
                continue
            rows[row["detection_id"]] = row

        unknown = set(rows) - set(pack_ids)
        if unknown:
            raise ResumeMismatch(
                f"{self.path.name} holds {len(unknown)} detection(s) not in the "
                f"current pack set -- the packs changed since it was written. "
                f"Delete it or pass --fresh.")

        retried = 0
        if retry_failed:
            failed = [k for k, r in rows.items() if not r.get("ok")]
            for k in failed:
                del rows[k]
            retried = len(failed)
        return rows, retried

    def start(self, keep: dict[str, dict], session: dict | None = None) -> int:
        """Rewrite the log as header + earlier sessions + kept rows, then open a
        new session. Returns the new session's index, for stamping rows."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions.append(session or {})
        with self.path.open("w") as f:
            f.write(json.dumps({"header": self.identity}) + "\n")
            for sess in self.sessions:
                f.write(json.dumps({"session": sess}) + "\n")
            for row in keep.values():
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return len(self.sessions) - 1

    def append(self, row: dict) -> None:
        # fsync per row: a few milliseconds against a generation that takes
        # seconds, and it is what makes "on disk" true when the VM is reclaimed.
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
