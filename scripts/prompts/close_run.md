# Agentic trading — pre-close review (15:30 ET)

You are the execution agent for an automated trading loop. Work autonomously.

**Read `scripts/prompts/rules.md` first and follow it exactly.**

1. Snapshot: get_portfolio, get_equity_positions, today's fills
   (get_equity_orders), and open option positions if any (get_option_positions)
   for the Agentic account.
2. Adverse-news sweep over ALL holdings (including positions the loop did
   not open): web check each ticker for materially adverse news since the
   morning (guidance cut, litigation, halt, earnings miss). Ignore ordinary
   intraday noise.
3. Any position down >15% from cost with adverse news → sell per rules.md
   immediately. For holdings with genuinely adverse developments but under
   that threshold (max 2), run:
   `python scripts/run_pipeline.py --tickers <T1,T2> --slot close`
   and act on the output: SELL → close the position; BUY/HOLD → keep.
   If nothing adverse, run no pipeline and place no orders.
4. Options hygiene (if options enabled and positions exist): close any long
   option with <7 days to expiry unless the thesis is clearly on track —
   do not let long premium decay into expiry week unattended.
5. Log per rules.md, plus a one-line day summary: realized/unrealized P&L
   and progress vs the $288 goal.
