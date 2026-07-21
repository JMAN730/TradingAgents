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

## Risk posture

Current owner preference: **full auto** — Robinhood-side manual review is OFF
and no per-order caps are configured. The dedicated Agentic account balance is
the only blast-radius limit. To tighten later:

- Robinhood app -> Agentic Trading settings -> require manual review per order.
- Add a hard cap to the execution prompt ("never exceed $X per order").
- Robinhood's own disclosure: agents can err; all agent trades are on you.
