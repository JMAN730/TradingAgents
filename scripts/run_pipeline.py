"""Headless TradingAgents runner for the Robinhood agentic-trading bridge.

Runs the full graph for each ticker and writes one machine-readable decision
summary the executing agent can act on. LLM provider defaults to claude_code
(Claude Pro/Max subscription auth); market data needs ALPHA_VANTAGE_API_KEY
in the environment or repo .env.

Usage:
    python scripts/run_pipeline.py --tickers AAPL,MSFT,NVDA --slot open
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--slot", default="adhoc", help="Run label: open|midday|close|adhoc")
    parser.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (default: today)")
    parser.add_argument("--deep-model", default="sonnet")
    parser.add_argument("--quick-model", default="haiku")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    trade_date = args.date or datetime.date.today().isoformat()
    config = dict(DEFAULT_CONFIG)
    config["llm_provider"] = "claude_code"
    config["deep_think_llm"] = args.deep_model
    config["quick_think_llm"] = args.quick_model

    decisions = []
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        print(f"[run_pipeline] {ticker} {trade_date} ...", flush=True)
        try:
            graph = TradingAgentsGraph(config=config)
            _, signal = graph.propagate(ticker, trade_date)
            decisions.append({"ticker": ticker, "signal": signal, "error": None})
            print(f"[run_pipeline] {ticker}: {signal}", flush=True)
        except Exception as e:  # one bad ticker must not kill the batch
            decisions.append({"ticker": ticker, "signal": None, "error": str(e)})
            print(f"[run_pipeline] {ticker} FAILED: {e}", flush=True)

    out_dir = REPO_ROOT / "results" / "agentic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trade_date}-{args.slot}.json"
    out_path.write_text(
        json.dumps(
            {
                "date": trade_date,
                "slot": args.slot,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "decisions": decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[run_pipeline] wrote {out_path}", flush=True)
    return 0 if any(d["signal"] for d in decisions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
