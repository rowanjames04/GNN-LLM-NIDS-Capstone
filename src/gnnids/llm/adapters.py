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

import json
import os
import re
import time

from .base import LLMResponse

# Set in `LLMResponse.extra` when a call failed for a reason outside the model --
# an exhausted quota, an unreachable server. `generate_reports.py` stops the run
# on it instead of recording a failed report, and a re-run resumes.
INFRASTRUCTURE_ERROR = "infrastructure_error"

# Anthropic list prices, USD per million tokens. Recorded here rather than
# fetched so a study run is reproducible from the repo alone; re-check before
# quoting a cost figure in the report, as list prices move.
ANTHROPIC_PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


# Gemini paid-tier list prices, USD per million tokens (text, standard), read
# from ai.google.dev/gemini-api/docs/pricing on 2026-09-15. The study uses the
# FREE tier (D36), so these are never charged: they give the cost comparison a
# reference figure for what a deployment on a billed key would pay. The
# gemini-3.8-flash price is marked "through Dec 31, 2026" on that page -- a
# promotional rate, so quote it with its date.
GEMINI_PRICING = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-2.5-flash": (0.30, 2.50),
}

# List price per provider, whatever this study's billing. `cost_usd` is what a
# call actually cost; `reference_cost_usd` is what it costs at list price. They
# are kept apart so a free-tier call is neither reported as costing $0 nor as
# costing money it did not.
LIST_PRICES = {"anthropic": ANTHROPIC_PRICING, "gemini": GEMINI_PRICING}


def reference_cost(provider: str, model: str, n_in: int | None,
                   n_out: int | None) -> float | None:
    """List-price cost of one call, or None where no per-token price exists
    (self-hosted models, unpriced models). The stub is genuinely free."""
    if provider == "stub":
        return 0.0
    return _cost(model, LIST_PRICES.get(provider, {}), n_in, n_out)


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
    """Google Gemini via the `google-genai` SDK, used on the free tier (D36).

    Worth having beyond roster size: Gemini exposes both `temperature` and
    `seed`, which current Claude models no longer do, so it is one of the models
    where the study's determinism control can actually be applied. That
    asymmetry is itself reportable.

    **Rate limits are infrastructure, not model behaviour.** The free tier caps
    requests per minute and per day. A 429 is retried with backoff (honouring
    the server's suggested delay), calls are spaced by `min_interval_seconds`,
    and neither wait is counted in latency. If the quota is still exhausted --
    or it is a *daily* quota, which no wait inside a run will clear -- the
    response is marked as an infrastructure error, and `generate_reports.py`
    stops without recording it. Recording it as a failed report would score
    Gemini worse for Google's quota, not for its reports.
    """

    provider = "gemini"
    RETRYABLE = (429, 500, 503)
    sleep = staticmethod(time.sleep)          # replaced in tests

    def __init__(self, model: str = "gemini-3.8-flash", max_tokens: int = 2000,
                 temperature: float | None = None, seed: int | None = None,
                 min_interval_seconds: float = 0.0, max_retries: int = 5,
                 backoff_seconds: float = 10.0) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.seed = temperature, seed
        self.min_interval_seconds = min_interval_seconds
        self.max_retries, self.backoff_seconds = max_retries, backoff_seconds
        self._last_call: float | None = None

    def generate(self, system: str, user: str, prompt_version: str) -> LLMResponse:
        from google.genai import errors

        retries, waited = 0, 0.0
        while True:
            self._throttle()
            t0 = time.perf_counter()
            try:
                r = self._call(system, user, prompt_version, t0)
            except errors.APIError as e:
                code = getattr(e, "code", None)
                details = json.dumps(getattr(e, "details", None) or {}, default=str)
                daily = "PerDay" in details
                if code in self.RETRYABLE and not daily and retries < self.max_retries:
                    delay = max(self.backoff_seconds * 2 ** retries,
                                _suggested_retry_delay(details) or 0.0)
                    self.sleep(delay)
                    retries, waited = retries + 1, waited + delay
                    continue
                r = self._error(e, prompt_version, t0)
                if code in self.RETRYABLE:
                    r.extra[INFRASTRUCTURE_ERROR] = (
                        "daily quota exhausted" if daily else
                        f"HTTP {code} after {retries} retries")
            except Exception as e:                   # noqa: BLE001
                r = self._error(e, prompt_version, t0)
            r.extra["rate_limit_retries"] = retries
            r.extra["rate_limit_wait_seconds"] = round(waited, 1)
            return r

    def _throttle(self) -> None:
        if self._last_call is not None and self.min_interval_seconds:
            gap = time.monotonic() - self._last_call
            if gap < self.min_interval_seconds:
                self.sleep(self.min_interval_seconds - gap)
        self._last_call = time.monotonic()

    def _call(self, system: str, user: str, prompt_version: str,
              t0: float) -> LLMResponse:
        from google import genai
        from google.genai import types

        # Credentials come from GEMINI_API_KEY / GOOGLE_API_KEY in the
        # environment; nothing is passed in code, and nothing loads a .env
        # file -- export the key in the shell first (see the README).
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
        answer = getattr(u, "candidates_token_count", None)
        # Gemini 3.x thinks by default. Thinking tokens are billed as output
        # but are NOT in candidates_token_count, and they count against
        # max_output_tokens -- so they are added here, and recorded apart.
        thoughts = getattr(u, "thoughts_token_count", None) or 0
        n_out = None if answer is None else answer + thoughts
        finish = next((str(c.finish_reason) for c in (r.candidates or [])
                       if c.finish_reason is not None), None)
        text = r.text or ""
        return LLMResponse(
            text=text, model=self.model, provider=self.provider,
            prompt_version=prompt_version,
            latency_seconds=round(time.perf_counter() - t0, 3),
            input_tokens=getattr(u, "prompt_token_count", None),
            output_tokens=n_out,
            # None, not 0.0: on the free tier nothing is charged, but that is
            # not a price, and 0.0 would win a cost comparison this model is
            # not competing in. The list price is reference_cost_usd.
            cost_usd=None,
            # An empty answer is a failed report, and saying why matters:
            # MAX_TOKENS here usually means thinking used the output budget,
            # which is a configuration problem, not the model refusing.
            error=(None if text.strip()
                   else f"empty response (finish_reason={finish})"),
            extra={"determinism": _determinism(self.temperature, self.seed),
                   "thoughts_tokens": thoughts, "finish_reason": finish,
                   "model_version": getattr(r, "model_version", None)},
        )

    def _error(self, e: Exception, prompt_version: str, t0: float) -> LLMResponse:
        return LLMResponse(
            text="", model=self.model, provider=self.provider,
            prompt_version=prompt_version,
            latency_seconds=round(time.perf_counter() - t0, 3),
            error=f"{type(e).__name__}: {e}")


def _suggested_retry_delay(details: str) -> float | None:
    """The `retryDelay` a 429 carries (e.g. "37s"), if any."""
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', details)
    return float(m.group(1)) if m else None


class OllamaAdapter:
    """Self-hosted. The V5 arm that iHPC's decommissioning threatened.

    Served by Ollama inside a Google Colab GPU runtime and reached over
    localhost from the same notebook (D35, [[Colab Self-Hosted Arm]]). A free
    Colab GPU holds ~8-14B at 4-bit where the M4 held ~3B, so V5 is *cloud vs
    mid-size open-weight*. `cost_usd` stays None on purpose -- a self-hosted
    model has no per-token price, and recording 0.0 would let it win a cost
    comparison it is not actually competing in.

    `num_ctx` is sent explicitly on every call: Ollama truncates a prompt longer
    than its context window rather than raising, so leaving it to the server's
    default would make an over-long prompt a silent defect.
    """

    provider = "ollama"

    def __init__(self, model: str = "llama3.1:8b", host: str | None = None,
                 temperature: float | None = None, seed: int | None = None,
                 num_ctx: int | None = None) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST")
        self.temperature, self.seed, self.num_ctx = temperature, seed, num_ctx

    def describe(self) -> dict:
        """The exact model build and server this run used, queried from Ollama.

        The **digest** is the identity to quote in the report; the tag is only a
        name, and it is re-pointed when a model is re-published. Never raises --
        an unreachable server records the error and the run itself will then
        fail loudly on its first call.
        """
        import ollama

        out: dict = {"tag": self.model, "num_ctx": self.num_ctx,
                     "digest": None, "parameter_size": None,
                     "quantization_level": None, "family": None,
                     "server_version": None}
        try:
            client = ollama.Client(host=self.host) if self.host else ollama.Client()
            d = client.show(self.model).details
            if d is not None:
                out.update(parameter_size=d.parameter_size,
                           quantization_level=d.quantization_level,
                           family=d.family)
            for m in client.list().models:
                if m.model == self.model:
                    out["digest"] = m.digest
            out["server_version"] = _ollama_server_version(self.host)
        except Exception as e:                       # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
        return out

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
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
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
                extra={"determinism": _determinism(self.temperature, self.seed),
                       "timing": _ollama_timing(r)},
            )
        except Exception as e:                       # noqa: BLE001
            r = LLMResponse(
                text="", model=self.model, provider=self.provider,
                prompt_version=prompt_version,
                latency_seconds=round(time.perf_counter() - t0, 3),
                error=f"{type(e).__name__}: {e}")
            # The Ollama client raises the builtin ConnectionError when the
            # server is gone -- in Colab, usually a restarted runtime. Every
            # remaining pack would fail in milliseconds and be recorded as the
            # model failing, so the run stops instead and resumes later.
            if isinstance(e, ConnectionError):
                r.extra[INFRASTRUCTURE_ERROR] = "Ollama server unreachable"
            return r


def _ollama_timing(r) -> dict:
    """Where the time went, as the Ollama server measured it (nanoseconds in,
    seconds out).

    `latency_seconds` is wall-clock from the client and includes loading the
    model into memory on the first call. `generation_seconds` is the time the
    model spent producing tokens -- on a GPU runtime, the GPU-seconds the study
    design asks for on the self-hosted arm. Until 2026-10-03 the server returned
    these with every response and the adapter discarded them.
    """
    def seconds(key):
        v = r.get(key)
        return round(v / 1e9, 3) if isinstance(v, (int, float)) else None
    return {
        "generation_seconds": seconds("eval_duration"),
        "prompt_seconds": seconds("prompt_eval_duration"),
        "load_seconds": seconds("load_duration"),
        "total_seconds": seconds("total_duration"),
    }


def _ollama_server_version(host: str | None) -> str | None:
    """`GET /api/version`. The Python client has no call for it."""
    import urllib.request

    base = host or "http://127.0.0.1:11434"
    if "://" not in base:
        base = f"http://{base}"
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/api/version", timeout=5) as r:
            return json.loads(r.read()).get("version")
    except (OSError, ValueError):
        return None


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
