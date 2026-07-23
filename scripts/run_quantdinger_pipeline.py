"""Headless TradingAgents runner that executes decisions through QuantDinger.

For each ticker this runs the full agent graph, then submits the resulting
signal to a QuantDinger deployment via its Agent Gateway. QuantDinger records
paper fills by default; live routing requires the operator to unlock it
server-side (see docs/quantdinger-bridge.md).

Environment:
    QUANTDINGER_BASE_URL     gateway base URL (default http://localhost:5000)
    QUANTDINGER_AGENT_TOKEN  agent token with scopes R+T (required unless --dry-run)

Usage:
    python scripts/run_quantdinger_pipeline.py --tickers AAPL,MSFT,NVDA --slot open
    python scripts/run_quantdinger_pipeline.py --tickers AAPL --dry-run
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
    parser.add_argument("--provider", default="claude_code", help="LLM provider")
    parser.add_argument("--deep-model", default="sonnet")
    parser.add_argument("--quick-model", default="haiku")
    parser.add_argument("--market", default="USStock", help="QuantDinger market code")
    parser.add_argument(
        "--notional", type=float, default=1000.0,
        help="Quote value of a full-conviction order (Overweight/Underweight trade half)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the graph and print signals without contacting QuantDinger",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.integrations.quantdinger import (
        QuantDingerClient,
        QuantDingerError,
        QuantDingerExecutor,
    )

    executor = None
    if not args.dry_run:
        executor = QuantDingerExecutor(
            client=QuantDingerClient.from_env(),
            market=args.market,
            order_notional=args.notional,
        )

    trade_date = args.date or datetime.date.today().isoformat()
    config = dict(DEFAULT_CONFIG)
    config["llm_provider"] = args.provider
    config["deep_think_llm"] = args.deep_model
    config["quick_think_llm"] = args.quick_model

    results = []
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        print(f"[quantdinger_pipeline] {ticker} {trade_date} ...", flush=True)
        entry: dict = {"ticker": ticker, "signal": None, "receipt": None, "error": None}
        try:
            graph = TradingAgentsGraph(config=config)
            _, signal = graph.propagate(ticker, trade_date)
            entry["signal"] = signal
            print(f"[quantdinger_pipeline] {ticker}: {signal}", flush=True)
        except Exception as e:  # one bad ticker must not kill the batch
            entry["error"] = f"graph: {e}"
            print(f"[quantdinger_pipeline] {ticker} GRAPH FAILED: {e}", flush=True)
            results.append(entry)
            continue
        if executor is not None:
            try:
                entry["receipt"] = executor.execute(
                    ticker, signal, trade_date=trade_date, slot=args.slot,
                )
                order = entry["receipt"].get("order") or {}
                print(
                    f"[quantdinger_pipeline] {ticker}: {entry['receipt']['action']} "
                    f"-> {order.get('status', 'no order')}"
                    f"{' (paper)' if order.get('paper') else ''}",
                    flush=True,
                )
            except QuantDingerError as e:
                entry["error"] = f"execution: {e}"
                print(f"[quantdinger_pipeline] {ticker} EXECUTION FAILED: {e}", flush=True)
        results.append(entry)

    out_dir = REPO_ROOT / "results" / "quantdinger"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trade_date}-{args.slot}.json"
    out_path.write_text(
        json.dumps(
            {
                "date": trade_date,
                "slot": args.slot,
                "market": args.market,
                "dry_run": bool(args.dry_run),
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[quantdinger_pipeline] wrote {out_path}", flush=True)
    return 0 if any(r["signal"] and not r["error"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
