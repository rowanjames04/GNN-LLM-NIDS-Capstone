"""One interface, several backends -- the seam the comparative study needs (7a).

V5 compares cloud and self-hosted models on the *same* task with the *same*
input. That only means anything if the models differ and nothing else does, so
every backend goes through one interface and the prompt, the evidence pack and
the scoring are identical across them.

`StubAdapter` exists so the whole pipeline can be exercised without spending
money or needing a network. It is not a mock in the testing sense -- it produces
a deterministic report from the pack, which makes it a useful control in the
study itself: the floor a real model has to beat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMResponse:
    """One generated report, with everything the study needs to score it."""

    text: str
    model: str
    provider: str
    prompt_version: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class LLMAdapter(Protocol):
    """What every backend must provide. Deliberately tiny.

    A wider interface would tempt provider-specific features into the study, and
    a feature only one provider has is a confound: the comparison would then be
    measuring the feature rather than the model.
    """

    provider: str
    model: str

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        ...


class StubAdapter:
    """Deterministic, offline, free. The control arm.

    It writes a report by **template** from the same prompt a real model gets.
    Two uses:

    - every smoke run and every test exercises the full path without a network
      call or a cent spent;
    - in the study proper it is the floor. A model that scores no better than a
      template on groundedness is not adding anything, and having that number
      makes the claim checkable instead of rhetorical.
    """

    provider = "stub"

    def __init__(self, model: str = "template-v1") -> None:
        self.model = model

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        t0 = time.perf_counter()
        # Echo back only what the prompt actually contains. A stub that invented
        # detail would score badly on groundedness for the wrong reason and make
        # the control useless.
        lines = [ln.strip() for ln in user.splitlines() if ln.strip()]
        facts = [ln for ln in lines if ln.startswith("- ")][:12]
        text = (
            "## Incident summary\n\n"
            "A network flow was flagged by the detection system. "
            "The evidence supplied is reproduced below without interpretation.\n\n"
            + ("\n".join(facts) if facts else "- No structured facts were supplied.")
            + "\n\n## Recommended action\n\n"
            "Review the evidence above against the host's normal behaviour "
            "before acting.\n"
        )
        return LLMResponse(
            text=text, model=self.model, provider=self.provider,
            prompt_version=prompt_version,
            latency_seconds=round(time.perf_counter() - t0, 6),
            input_tokens=len(user.split()), output_tokens=len(text.split()),
            cost_usd=0.0,
        )
