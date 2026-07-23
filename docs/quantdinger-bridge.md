# QuantDinger Execution Bridge

TradingAgents decides; [QuantDinger](https://github.com/jman730/QuantDinger)
executes. This bridge submits the graph's five-tier portfolio decision
(Buy / Overweight / Hold / Underweight / Sell) to a QuantDinger deployment
through its Agent Gateway (`/api/agent/v1`), replacing ad-hoc execution
glue with an audited, paper-by-default trading surface.

## How it maps

| Signal      | Order side | Size                    |
|-------------|------------|-------------------------|
| Buy         | buy        | full `--notional`       |
| Overweight  | buy        | half `--notional`       |
| Hold        | —          | no order                |
| Underweight | sell       | half `--notional`       |
| Sell        | sell       | full `--notional`       |

Quantity is computed from the gateway's latest price
(`GET /api/agent/v1/price`). Each order carries an idempotency key derived
from `(date, slot, ticker, side)`, so re-running a slot replays the recorded
order instead of doubling the position.

## Safety model

Execution risk lives entirely on the QuantDinger side, behind three
independent gates:

1. The agent token must have scope `T` (trading).
2. The token must have `paper_only=false` — set deliberately by the operator.
3. The backend must run with `AGENT_LIVE_TRADING_ENABLED=true`.

Until **all three** hold, every order is recorded as a simulated paper fill
(`qd_agent_paper_orders`) — the full round trip works without touching
exchange credentials. TradingAgents never sees broker keys; credential
management (`C` scope) is admin-only in QuantDinger.

## Setup

1. In QuantDinger, issue an agent token with scopes `R` (reads) and `T`
   (trading), leaving `paper_only` on.
2. Configure the environment (repo `.env` works):

   ```bash
   QUANTDINGER_BASE_URL=http://localhost:5000
   QUANTDINGER_AGENT_TOKEN=<token>
   ```

3. Run the pipeline:

   ```bash
   # decide + submit paper orders
   python scripts/run_quantdinger_pipeline.py --tickers AAPL,MSFT,NVDA --slot open

   # decide only, no gateway calls
   python scripts/run_quantdinger_pipeline.py --tickers AAPL --dry-run
   ```

Results (signals + order receipts) are written to
`results/quantdinger/<date>-<slot>.json`.

## Programmatic use

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.integrations.quantdinger import QuantDingerClient, QuantDingerExecutor

graph = TradingAgentsGraph()
_, signal = graph.propagate("NVDA", "2026-07-23")

executor = QuantDingerExecutor(
    client=QuantDingerClient.from_env(),
    market="USStock",
    order_notional=1000.0,
)
receipt = executor.execute("NVDA", signal, trade_date="2026-07-23", slot="adhoc")
```

`QuantDingerClient.kill_switch()` cancels all of the token's open paper
orders.

## Going the other way

QuantDinger strategies can also consult TradingAgents in-process through the
`TradingAgentsDecisionClient` shipped in the QuantDinger repo
(`docs/agent/TRADINGAGENTS_INTEGRATION.md` there) — useful when QuantDinger
should own sizing/risk and TradingAgents only supplies direction.
