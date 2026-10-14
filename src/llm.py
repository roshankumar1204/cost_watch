"""Thin wrapper around Groq's chat API that also keeps an honest token log."""
import json
import os
import time
from dataclasses import dataclass, field

import config


@dataclass
class TokenLog:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_purpose: dict = field(default_factory=dict)

    def add(self, purpose: str, inp: int, out: int) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        p = self.by_purpose.setdefault(purpose, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        p["calls"] += 1
        p["input_tokens"] += inp
        p["output_tokens"] += out

    def summary(self) -> dict:
        cost = (self.input_tokens * config.PRICE_PER_M_INPUT
                + self.output_tokens * config.PRICE_PER_M_OUTPUT) / 1_000_000
        return {
            "model": config.LLM_MODEL,
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "price_usd_per_1M": {"input": config.PRICE_PER_M_INPUT, "output": config.PRICE_PER_M_OUTPUT},
            "estimated_cost_usd_at_published_rate": round(cost, 6),
            "actual_cost_on_free_tier_usd": 0.0,
            "by_purpose": self.by_purpose,
        }


class LLM:
    """Returns None from every call when no GROQ_API_KEY is set, so callers fall back
    to deterministic templates and the pipeline still runs end to end."""

    def __init__(self, enabled: bool = True):
        self.log = TokenLog()
        self.client = None
        if enabled and os.getenv("GROQ_API_KEY"):
            from groq import Groq
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    @property
    def available(self) -> bool:
        return self.client is not None

    def chat(self, system: str, user: str, purpose: str, json_mode: bool = False):
        if not self.available:
            return None
        kwargs = dict(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            seed=config.LLM_SEED,
            max_completion_tokens=config.LLM_MAX_TOKENS,
        )
        if config.LLM_MODEL.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
            kwargs["include_reasoning"] = False
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # rate limit or transient error: back off and retry
                if attempt == 4:
                    print(f"[llm] giving up after retries ({purpose}): {exc}")
                    return None
                time.sleep(2 ** attempt * 2)

        usage = resp.usage
        self.log.add(purpose, usage.prompt_tokens, usage.completion_tokens)
        text = (resp.choices[0].message.content or "").strip()
        if json_mode:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        return text
