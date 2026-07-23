# Agentic trading — midday research run (12:30 ET)

You are the execution agent for an automated trading loop. Work autonomously.

**Read `scripts/prompts/rules.md` first and follow it exactly.**

1. Snapshot: get_portfolio + get_equity_positions for the Agentic account.
   Check option_level per rules.md (fresh get_accounts call).
2. Candidate hunt: run_scan on "Undervalued hunt"
   (scan_id 80b034c8-a198-40de-801c-3dc046c5098b). Cross-check against
   today's news via web search (earnings reactions, sector moves). Rank the
   TWO best validated candidates not already held and not analyzed by
   today's open run. Record 2-3 sentences per pick on why (and why it's
   not a value trap).
3. Run: `python scripts/run_pipeline.py --tickers <PICK1> --slot midday`
   (top pick only — foreground, wait for it).
4. Read results/agentic/<date>-midday.json and execute per rules.md, with
   this slot's tighter cap: at most 25% of current buying power on the BUY
   (shares or a single-leg call if options enabled).
   SELL → close existing position if any; HOLD → no action.
5. Fallback: if PICK1's signal was NOT a BUY, run the pipeline once more on
   PICK2 (same command, foreground) and execute its decision under the same
   25% cap. Max 2 pipeline runs per midday slot — never a third.
6. Log per rules.md (note both picks and, on fallback, both signals) and
   update pnl_report.md.
