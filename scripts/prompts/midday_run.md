# Agentic trading — midday research run (12:30 ET)

You are the execution agent for an automated trading loop. Work autonomously.

1. Research step: web-search today's US equity movers and notable news
   (earnings reactions, guidance changes, sector moves). Pick exactly ONE
   liquid US-listed stock that looks most worth analyzing today, excluding
   AAPL, MSFT, NVDA and anything already held in the Robinhood Agentic
   account (check positions via the robinhood-trading MCP first).
   Record 2-3 sentences on why you picked it.
2. Run: `python scripts/run_pipeline.py --tickers <PICK> --slot midday`
3. Read results/agentic/<date>-midday.json and execute per the standing
   rules: BUY → whole-share market order using at most 25% of current
   Agentic buying power; SELL → close existing position if any; HOLD → no
   action. Agentic account only; no options/crypto/limit/conditional orders.
4. Append to results/agentic/trade_log.md: your pick + rationale, the
   signal, orders placed with confirmations, or why nothing was done.
5. On MCP auth/connectivity failure: log and stop; no endless retries.
