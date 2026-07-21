# Robinhood Agentic Trading bridge (supervised shape)

Wires TradingAgents' decision output into Robinhood's Agentic Trading via the
Claude Code CLI + Robinhood MCP server. No repo code involved: the pipeline
produces the decision, a Claude Code session holding the Robinhood MCP
connection executes it.

## One-time setup

1. Robinhood: primary individual account in good standing. The MCP OAuth flow
   auto-creates a dedicated **Agentic account** — the only account the agent
   can trade. Fund it separately with only what the agent may lose.
2. Register the MCP server (already done for this project):

   ```sh
   claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
   ```

3. Authenticate: inside a Claude Code session in this repo, run `/mcp`, pick
   `robinhood-trading`, complete the browser/app OAuth.

## Per-trade flow

1. Run the pipeline (provider `claude_code` = subscription-billed LLM calls):

   ```sh
   python -m cli.main
   ```

   Reports and the final decision land under the configured `results_dir`
   (default `./results/<ticker>/<date>/`).

2. In a Claude Code session with the MCP connected, ask e.g.:

   > Read the latest final decision under results/ and execute it in my
   > Robinhood Agentic account: BUY -> market order sized per the trader
   > proposal (or your judgment from buying power), SELL -> close the
   > position, HOLD -> do nothing. Report the order confirmation.

## Automated schedule (current setup)

Windows Task Scheduler fires three weekday runs (machine must be awake):

| Task | Time (ET) | Prompt | What it does |
|---|---|---|---|
| `TradingAgents\OpenRun` | 09:47 | `scripts/prompts/open_run.md` | Pipeline on AAPL, MSFT, NVDA → execute signals |
| `TradingAgents\MiddayRun` | 12:33 | `scripts/prompts/midday_run.md` | Web-research one new candidate → pipeline → execute (≤25% buying power) |
| `TradingAgents\CloseRun` | 15:28 | `scripts/prompts/close_run.md` | Position review; pipeline only on holdings with adverse news |

Each task runs `claude -p` headless (`scripts/tasks/*.cmd`) with
`--permission-mode bypassPermissions`; logs to `results/agentic/cron_*.log`
and orders to `results/agentic/trade_log.md`. Headless pipeline entry:
`scripts/run_pipeline.py` (quick=haiku, deep=sonnet by default).

Manage: `schtasks /Query /TN "TradingAgents\OpenRun"`, `/Change /DISABLE`,
`/Delete`. Prompts are plain markdown — edit to retune rules.

## Risk posture

Current owner preference: **full auto** — Robinhood-side manual review is OFF
and no per-order caps are configured. The dedicated Agentic account balance is
the only blast-radius limit. To tighten later:

- Robinhood app -> Agentic Trading settings -> require manual review per order.
- Add a hard cap to the execution prompt ("never exceed $X per order").
- Robinhood's own disclosure: agents can err; all agent trades are on you.
