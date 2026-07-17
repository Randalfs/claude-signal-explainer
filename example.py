"""
example.py — end-to-end demo of signal_explainer.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python example.py
"""
import json

from signal_explainer import explain_signal, analyze_dataset


# ---------------------------------------------------------------------------
# Phase 1 — one hot signal, one short rationale
# ---------------------------------------------------------------------------
hot_signal = {
    "id": "signal-2026-07-17-0001",
    "score": 82,
    "level": "high_conviction",
    "signals": ["momentum_break", "volume_spike", "reversion_setup"],
    "context": {
        "price": 1.234,
        "change_1h": 4.2,
        "change_24h": 11.6,
        "rsi_14": 74,
        "volume_zscore": 3.1,
    },
    "notes": "third breakout attempt in 48h",
}

print("=" * 60)
print("PHASE 1 — real-time explainer (fast model)")
print("=" * 60)
rationale = explain_signal(hot_signal)
print(rationale or "[no rationale — check ANTHROPIC_API_KEY and rate limit]")


# ---------------------------------------------------------------------------
# Phase 2 — batch analysis of a small history
# ---------------------------------------------------------------------------
history = {
    "trades": [
        {"id": 1, "outcome": "win",  "pnl_pct": 4.2, "signals": ["momentum_break", "volume_spike"]},
        {"id": 2, "outcome": "loss", "pnl_pct": -2.1, "signals": ["reversion_setup"]},
        {"id": 3, "outcome": "win",  "pnl_pct": 3.8, "signals": ["momentum_break", "volume_spike", "reversion_setup"]},
        {"id": 4, "outcome": "loss", "pnl_pct": -1.7, "signals": ["momentum_break"]},
        {"id": 5, "outcome": "win",  "pnl_pct": 6.1, "signals": ["volume_spike", "reversion_setup"]},
    ],
    "weights": {
        "momentum_break": 15,
        "volume_spike": 20,
        "reversion_setup": 10,
    },
}

print("\n" + "=" * 60)
print("PHASE 2 — batch analyst (deep model)")
print("=" * 60)
report = analyze_dataset(
    dataset=history,
    instructions="Which signal combinations correlate with wins? What weights should I try next?",
    section_budgets={"trades": 3000, "weights": 500},
    max_tokens=1200,
)
print(report or "[no report — check ANTHROPIC_API_KEY]")
