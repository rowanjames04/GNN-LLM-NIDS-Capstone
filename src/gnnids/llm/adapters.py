"""Cloud and self-hosted backends behind the one interface (7a/7b).

**No adapter here is called during a smoke run or a test.** They are constructed
lazily and their SDKs are imported inside `generate`, so the pipeline runs
end-to-end on `StubAdapter` with no key, no network and no cost. Running a real
model is an explicit choice, and it spends Rowan's money -- see
[[LLM Comparative Study Design]] and the note in [[Compute Access Risk]].

Each adapter reports tokens and cost where the provider exposes them, because
cost per report is one of the five metric families in the study and a
self-hosted model's *not* having a per-token price is itself part of that
comparison.
"""

from __future__ import annotations

import os
import time

from .base import LLMResponse

# Anthropic list prices, USD per million tokens. Recorded here rather than
# fetched so a study run is reproducible from the repo alone; re-check before
# quoting a cost figure in the report, as list prices move.
ANTHROPIC_PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _determinism(temperature, seed, note: str | None = None) -> dict:
    """What determinism controls were actually applied to this call.

    Recorded per response rather than assumed, because the roster is not
    uniform: OpenAI, Gemini and Ollama accept a temperature, while current
    Claude models reject it outright. A study claiming "all models at
    temperature 0" would be wrong; one that records what each call could
    actually be given is defensible.
    """
    out = {"temperature": temperature, "seed": seed,
           "supported": temperature is not None or seed is not None}
    if note:
        out["note"] = note
    return out


def _cost(model: str, table: dict, n_in: int | None, n_out: int | None) -> float | None:
    if n_in is None or n_out is None or model not in table:
        return None
    price_in, price_out = table[model]
    return round(n_in / 1e6 * price_in + n_out / 1e6 * price_out, 6)


class AnthropicAdapter:
    """Claude via the official SDK.

    Defaults to `claude-opus-5`. Thinking is left at its default (adaptive) and
    `effort` is exposed rather than hardcoded: a report-writing task is not
    reasoning-heavy, so the study should be free to compare effort levels as a
    cost lever without editing code.
    """

    provider = "anthropic"

    # Sampling controls were removed on the current Claude generation: passing
    # `temperature`, `top_p` or `top_k` to these models returns a 400. They are
    # therefore not sent, and the response records that the study's determinism
    # control could not be applied here. `effort` is the available lever.
    NO_SAMPLING_CONTROLS = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                            "claude-sonnet-5", "claude-fable-5")

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 2000,
                 effort: str | None = None, temperature: float | None = None,
                 seed: int | None = None) -> None:
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        # Accepted from config so the roster can be configured uniformly, then
        # dropped for models that reject them -- a silent 400 mid-sweep would be
        # a worse failure than an honest "unsupported" in the results.
        self.temperature = None if model in self.NO_SAMPLING_CONTROLS else temperature
        self.seed = seed
        self.sampling_dropped = (model in self.NO_SAMPLING_CONTROLS
                                 and temperature is not None)

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        import anthropic

        t0 = time.perf_counter()
        try:
            client = anthropic.Anthropic()
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if self.effort:
                kwargs["output_config"] = {"effort": self.effort}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            msg = client.messages.create(**kwargs)
            # stop_details is populated only on a refusal; guard before reading.
            if msg.stop_reason == "refusal":
                why = getattr(msg.stop_details, "category", None)
                return self._fail(f"refused ({why})", prompt_version, t0)
            text = "".join(b.text for b in msg.content if b.type == "text")
            n_in, n_out = msg.usage.input_tokens, msg.usage.output_tokens
            return LLMResponse(
                text=text, model=self.model, provider=self.provider,
                prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                input_tokens=n_in, output_tokens=n_out,
                cost_usd=_cost(self.model, ANTHROPIC_PRICING, n_in, n_out),
                extra={"stop_reason": msg.stop_reason,
                       "determinism": _determinism(
                           self.temperature, self.seed,
                           note=("sampling controls unavailable on this model"
                                 if self.sampling_dropped or
                                 self.model in self.NO_SAMPLING_CONTROLS else None))},
            )
        except Exception as e:                       # noqa: BLE001
            return self._fail(f"{type(e).__name__}: {e}", prompt_version, t0)

    def _fail(self, why: str, prompt_version: str, t0: float) -> LLMResponse:
        # A failed generation is recorded, never dropped. A study that silently
        # omits the runs where a model refused or errored reports that model as
        # better than it is.
        return LLMResponse(
            text="", model=self.model, provider=self.provider,
            prompt_version=prompt_version,
            latency_seconds=round(time.perf_counter() - t0, 3), error=why)


class OpenAIAdapter:
    """The second cloud arm of the comparison."""

    provider = "openai"

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 2000,
                 temperature: float | None = None, seed: int | None = None) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.seed = temperature, seed

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        from openai import OpenAI

        t0 = time.perf_counter()
        try:
            kwargs = {"model": self.model, "max_tokens": self.max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.seed is not None:
                kwargs["seed"] = self.seed
            r = OpenAI().chat.completions.create(**kwargs)
            u = r.usage
            return LLMResponse(
                text=r.choices[0].message.content or "", model=self.model,
                provider=self.provider, prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                extra={"determinism": _determinism(self.temperature, self.seed)},
            )
        except Exception as e:                       # noqa: BLE001
            return LLMResponse(
                text="", model=self.model, provider=self.provider,
                prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                error=f"{type(e).__name__}: {e}")


class GeminiAdapter:
    """Google Gemini via the `google-genai` SDK.

    The third cloud arm. Worth having beyond roster size: Gemini exposes both
    `temperature` and `seed`, which current Claude models no longer do, so it is
    one of the models where the study's determinism control can actually be
    applied. That asymmetry is itself reportable.
    """

    provider = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", max_tokens: int = 2000,
                 temperature: float | None = None, seed: int | None = None) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.seed = temperature, seed

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        from google import genai
        from google.genai import types

        t0 = time.perf_counter()
        try:
            # Credentials come from GEMINI_API_KEY / GOOGLE_API_KEY in the
            # environment; nothing is passed in code (secrets live in .env).
            client = genai.Client()
            cfg = {"system_instruction": system, "max_output_tokens": self.max_tokens}
            if self.temperature is not None:
                cfg["temperature"] = self.temperature
            if self.seed is not None:
                cfg["seed"] = self.seed
            r = client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(**cfg))
            u = getattr(r, "usage_metadata", None)
            return LLMResponse(
                text=r.text or "", model=self.model, provider=self.provider,
                prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                input_tokens=getattr(u, "prompt_token_count", None),
                output_tokens=getattr(u, "candidates_token_count", None),
                # No per-token price is recorded here. Reporting 0.0 would let
                # this model win a cost comparison it is not competing in; the
                # aggregation reports "unknown" instead.
                cost_usd=None,
                extra={"determinism": _determinism(self.temperature, self.seed)},
            )
        except Exception as e:                       # noqa: BLE001
            return LLMResponse(
                text="", model=self.model, provider=self.provider,
                prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                error=f"{type(e).__name__}: {e}")


class OllamaAdapter:
    """Self-hosted, local. The V5 arm that iHPC's decommissioning threatened.

    With no cluster available the working assumption is small quantised models
    on the M4 ([[Compute Access Risk]] fallback 2), which changes V5 from
    *cloud vs large local* to *cloud vs small local*. That is a stated ceiling,
    not a silent downgrade, and `cost_usd` stays None here on purpose -- a
    self-hosted model has no per-token price, and recording 0.0 would let it win
    a cost comparison it is not actually competing in.
    """

    provider = "ollama"

    def __init__(self, model: str = "llama3.2:3b", host: str | None = None,
                 temperature: float | None = None, seed: int | None = None) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST")
        self.temperature, self.seed = temperature, seed

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        import ollama

        t0 = time.perf_counter()
        try:
            client = ollama.Client(host=self.host) if self.host else ollama
            options = {}
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if self.seed is not None:
                options["seed"] = self.seed
            r = client.chat(model=self.model, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}], options=options or None)
            return LLMResponse(
                text=r["message"]["content"], model=self.model,
                provider=self.provider, prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                input_tokens=r.get("prompt_eval_count"),
                output_tokens=r.get("eval_count"),
                cost_usd=None,
                extra={"determinism": _determinism(self.temperature, self.seed)},
            )
        except Exception as e:                       # noqa: BLE001
            return LLMResponse(
                text="", model=self.model, provider=self.provider,
                prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                error=f"{type(e).__name__}: {e}")


ADAPTERS = {
    "stub": None,          # resolved in build_adapter to avoid a circular import
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
}


def build_adapter(provider: str, **kwargs):
    """Construct one backend by name. Construction never touches the network."""
    from .base import StubAdapter

    if provider == "stub":
        return StubAdapter(**kwargs)
    if provider not in ADAPTERS:
        raise KeyError(f"unknown provider {provider!r}; have {sorted(ADAPTERS)}")
    return ADAPTERS[provider](**kwargs)
