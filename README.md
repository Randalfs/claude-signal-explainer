# claude-signal-explainer

A tiny two-phase Claude wrapper for any signal-scoring system that wants an LLM in the loop without becoming an LLM system.

- **Phase 1** takes a scored signal and returns a short, operator-facing rationale in 2-3 sentences. Runs on a fast model, rate-limited, TTL-cached, and safe to call from synchronous code. Designed to be dropped into the hot path just before a Telegram / Slack / email alert fires. If the model call fails or the rate-limit budget is exhausted, it returns `None` instead of blocking, so the alert still ships without an explanation.
- **Phase 2** takes a dataset dict and an optional instruction and returns a longer analysis on a bigger model. Built for after-the-fact review of many signals at once. Handles oversized inputs by hard-capping each named section of the dataset at a configurable character budget, so you can hand it 10 MB of trade history without blowing the context window.

Zero dependencies beyond `aiohttp`. No SDK required — the module calls the Anthropic REST API directly, which keeps the surface small and the wire behaviour obvious.

## Why this exists

I built the original of this module for the LLM-brain layer of a live crypto futures scanner. Every time the scanner fires a high-conviction signal, Phase 1 writes a two-sentence rationale into the Telegram alert next to the numeric score, so the operator can decide whether to act without having to reconstruct the scanner's reasoning from raw fields. Phase 2 runs on demand, chews through the accumulated trade history and adaptive signal weights, and returns a written report on where the scanner is well-calibrated and where it is not.

The design constraints that shaped this repo:

- **Never block the hot path.** If Anthropic is slow, the operator alert must still ship on time. Every call has an explicit timeout and a rate-limit budget; if either trips, we return `None` and the caller keeps going.
- **Never leak state into the model.** Rate limiting is per-process, cache is in-memory, no writes to disk.
- **Be embeddable from sync code.** Bots and scanners tend to live in `while True:` loops that were never designed for `asyncio`. The public API is synchronous; the async loop is spawned in a worker thread so we do not have to negotiate with the host loop.
- **Handle oversized inputs.** Real datasets are messy. Phase 2 lets you specify a per-section character budget so a single blob of raw JSON cannot break the request.

## Install

Requires Python 3.10+.

```bash
pip install aiohttp
export ANTHROPIC_API_KEY=sk-ant-...
```

## Use it

```python
from signal_explainer import explain_signal, analyze_dataset

# Phase 1 — real-time explainer
rationale = explain_signal({
    "id": "signal-42",
    "score": 82,
    "signals": ["momentum_break", "volume_spike"],
    "context": {"rsi": 71, "change_24h": 6.4},
})
# -> "Score of 82 driven mainly by the volume spike (Z=3.1) on top of a fresh
#    momentum break. RSI at 71 is stretched but not extreme; the setup has room
#    to reverse if volume fades."

# Phase 2 — batch analyst
report = analyze_dataset(
    dataset={
        "trades":  [...],
        "weights": {...},
    },
    instructions="Which signal combos correlate with wins?",
    section_budgets={"trades": 8000, "weights": 500},
)
```

A runnable end-to-end example with fake data lives in `example.py`. If your `ANTHROPIC_API_KEY` is set, `python example.py` will print both a Phase 1 rationale and a Phase 2 report.

## Configuration

Everything is env-vars, with sensible defaults:

| Variable                          | Default                        | Purpose                                        |
| --------------------------------- | ------------------------------ | ---------------------------------------------- |
| `ANTHROPIC_API_KEY`               | *(required)*                   | Anthropic API key                              |
| `SIGNAL_EXPLAINER_FAST_MODEL`     | `claude-haiku-4-5-20251001`    | Model for Phase 1                              |
| `SIGNAL_EXPLAINER_DEEP_MODEL`     | `claude-sonnet-4-6`            | Model for Phase 2                              |
| `SIGNAL_EXPLAINER_RPM`            | `10`                           | Phase 1 requests per minute cap                |
| `SIGNAL_EXPLAINER_CACHE_TTL`      | `300`                          | Phase 1 cache lifetime (seconds)               |

## Design notes

- **Cache key.** Phase 1 uses `signal["id"]` if present, otherwise `signal["symbol"]`. Same key inside `CACHE_TTL_SECONDS` returns the cached rationale rather than making another call. If neither key is present, we skip the cache entirely.
- **Rate limit.** A rolling 60-second window of call timestamps. Cheap, no external dependency, resets on process restart. Good enough for a single-process scanner; if you fan out to workers you will want to move the budget into Redis.
- **Sync-from-async trick.** `explain_signal` spawns a fresh event loop inside a `ThreadPoolExecutor(max_workers=1)`. This is deliberately clunky: it lets you call the module from code that is neither asyncio-aware nor thread-safe, without asking the caller to know anything about either.
- **Truncation is dumb.** Phase 2's per-section budget is a plain character cap. If the JSON is cut mid-object, the model gets a torn tail. In practice this has been fine; if you see garbled analyses, raise the budget for the offending section.

## License

MIT — see `LICENSE`.
