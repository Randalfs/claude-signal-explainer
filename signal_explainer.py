"""
signal_explainer.py — Two-phase Claude wrapper for signal-scoring systems.

Phase 1: real-time explainer. Take a scored signal dict, return a short
natural-language rationale suitable for an operator alert (Telegram, Slack,
email). Uses a fast model (Haiku), rate-limited, LRU-cached, and safe to
call from synchronous code that already runs an event loop elsewhere.

Phase 2: batch analyst. Take a dataset dict (any shape you want) plus an
optional operator question and return a longer analysis with a bigger model
(Sonnet). Designed for after-the-fact review of many signals at once, with
built-in truncation of oversized JSON blobs so the prompt stays within budget.

Deliberately calls the Anthropic REST API directly rather than the SDK.
Zero dependencies beyond aiohttp — drops into any Python 3.10+ project.

Environment:
    ANTHROPIC_API_KEY   required

Usage:
    from signal_explainer import explain_signal, analyze_dataset

    text = explain_signal({
        "id": "abc",
        "score": 82,
        "signals": ["momentum_break", "volume_spike"],
        "context": {"rsi": 71, "change_24h": 6.4},
    })

    report = analyze_dataset(
        dataset={"trades": [...], "weights": {...}},
        instructions="What signal combinations drive winning trades?",
    )
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
from typing import Any

import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Model selection.  Fast + cheap for real-time rationales, larger for batch.
FAST_MODEL = os.environ.get("SIGNAL_EXPLAINER_FAST_MODEL", "claude-haiku-4-5-20251001")
DEEP_MODEL = os.environ.get("SIGNAL_EXPLAINER_DEEP_MODEL", "claude-sonnet-4-6")

# Real-time budget.  Explainer calls are hot-path — if we blow through
# MAX_CALLS_PER_MINUTE we drop rather than block the operator alert.
MAX_CALLS_PER_MINUTE = int(os.environ.get("SIGNAL_EXPLAINER_RPM", "10"))
CACHE_TTL_SECONDS = int(os.environ.get("SIGNAL_EXPLAINER_CACHE_TTL", "300"))
EXPLAIN_TIMEOUT_S = 8
ANALYZE_TIMEOUT_S = 90

_call_timestamps: list[float] = []
_explanation_cache: dict[str, tuple[float, str]] = {}


# ---------------------------------------------------------------------------
# Phase 1 — real-time signal explainer
# ---------------------------------------------------------------------------
def _build_signal_prompt(signal: dict[str, Any]) -> str:
    """Prompt for Phase 1.

    Kept intentionally minimal.  The signal dict carries all context; the
    prompt just tells the model what to do with it.
    """
    return (
        "You are the explainer inside a signal-scoring system. A high-conviction "
        "signal just triggered. In 2-3 sentences, tell the operator WHY. Be "
        "specific about which fields are strongest and what they imply. Reference "
        "actual numbers from the signal. Do not hedge; the system already decided "
        "this is worth surfacing.\n\n"
        f"Signal:\n```json\n{json.dumps(signal, indent=2, default=str)}\n```\n\n"
        "Rules:\n"
        "- Max 3 sentences\n"
        "- Plain language, no jargon soup\n"
        "- Name the 1-2 strongest fields driving the score\n"
        "- End with the key risk if any input is borderline\n"
        "- Respond with ONLY the explanation text, no labels or formatting"
    )


async def explain_signal_async(signal: dict[str, Any]) -> str | None:
    """Async explainer.

    Returns a rationale string, or None if:
      - no API key set
      - rate-limit budget exhausted for the current minute
      - the API call fails or times out

    Callers should treat None as "we did not get an explanation for this
    signal" and fall through to sending the alert without one, never block on
    the LLM.
    """
    if not ANTHROPIC_API_KEY:
        return None

    now = time.time()
    global _call_timestamps
    _call_timestamps = [t for t in _call_timestamps if now - t < 60]
    if len(_call_timestamps) >= MAX_CALLS_PER_MINUTE:
        return None
    _call_timestamps.append(now)

    cache_key = str(signal.get("id") or signal.get("symbol") or "")
    if cache_key:
        cached = _explanation_cache.get(cache_key)
        if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
            return cached[1]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                json={
                    "model": FAST_MODEL,
                    "max_tokens": 200,
                    "messages": [
                        {"role": "user", "content": _build_signal_prompt(signal)}
                    ],
                },
                timeout=aiohttp.ClientTimeout(total=EXPLAIN_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                explanation = data["content"][0]["text"].strip()
                if cache_key:
                    _explanation_cache[cache_key] = (now, explanation)
                return explanation
    except Exception:
        return None


def explain_signal(signal: dict[str, Any]) -> str | None:
    """Sync wrapper.

    Safe to call from a synchronous code path even if the surrounding process
    is running an asyncio loop elsewhere.  We spin up a fresh loop in a worker
    thread rather than fighting whatever loop the host process owns.
    """
    def _run() -> str | None:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(explain_signal_async(signal))
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result(timeout=EXPLAIN_TIMEOUT_S + 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 2 — batch dataset analyst
# ---------------------------------------------------------------------------
DEFAULT_ANALYSIS_INSTRUCTIONS = (
    "Analyze this dataset. Report:\n"
    "1. Headline outcomes (aggregate metrics, distributions)\n"
    "2. Patterns that separate positive vs negative cases\n"
    "3. Parameters that look mis-calibrated relative to outcomes\n"
    "4. Blind spots the current setup might miss\n"
    "5. Specific, concrete changes to try next, with reasoning\n"
    "\n"
    "Be brutally specific. Use actual numbers from the data. No generic advice."
)


def _truncate_json_blob(obj: Any, budget: int) -> str:
    """Serialize `obj` and hard-cap the result at `budget` characters.

    Prevents oversize inputs from blowing past the model context window.
    Truncation is not aware of JSON structure — assume the model will handle
    a mid-object cut, and adjust the budget if you see garbled tail behaviour.
    """
    s = json.dumps(obj, indent=2, default=str)
    if len(s) <= budget:
        return s
    return s[:budget] + "\n... [truncated]"


def _build_analysis_prompt(
    dataset: dict[str, Any],
    instructions: str,
    section_budgets: dict[str, int] | None = None,
) -> str:
    """Prompt for Phase 2.

    `dataset` is a dict of named sections. Each section becomes a JSON blob
    in the prompt, capped at the corresponding per-section character budget.
    Absent budgets default to 4000 chars per section.
    """
    budgets = section_budgets or {}
    parts = ["You are a data analyst. Read the sections below carefully.\n"]
    for name, section in dataset.items():
        budget = budgets.get(name, 4000)
        parts.append(f"\n### {name.upper()}\n```json\n{_truncate_json_blob(section, budget)}\n```")
    parts.append(f"\n\nInstructions:\n{instructions}")
    return "".join(parts)


async def analyze_dataset_async(
    dataset: dict[str, Any],
    instructions: str = DEFAULT_ANALYSIS_INSTRUCTIONS,
    section_budgets: dict[str, int] | None = None,
    max_tokens: int = 3000,
) -> str | None:
    """Async batch analyst.  Returns the full analysis text or None on error."""
    if not ANTHROPIC_API_KEY:
        return None

    prompt = _build_analysis_prompt(dataset, instructions, section_budgets)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                json={
                    "model": DEEP_MODEL,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=ANALYZE_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data["content"][0]["text"].strip()
    except Exception:
        return None


def analyze_dataset(
    dataset: dict[str, Any],
    instructions: str = DEFAULT_ANALYSIS_INSTRUCTIONS,
    section_budgets: dict[str, int] | None = None,
    max_tokens: int = 3000,
) -> str | None:
    """Sync wrapper around `analyze_dataset_async`."""
    def _run() -> str | None:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                analyze_dataset_async(dataset, instructions, section_budgets, max_tokens)
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run).result(timeout=ANALYZE_TIMEOUT_S + 5)
