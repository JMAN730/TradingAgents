# Standing rules — Agentic trading loop (v2, 2026-07-22)

These rules apply to every run slot. Slot prompts reference this file.

## Goal

+30% account value in 6 months. Baseline: $221.85 on 2026-07-22 → target
~$288 by 2027-01-22. Aggressive risk tolerance by the account owner's
explicit choice. Log current account value (get_portfolio total_value) in
every trade_log entry as `value: $X (goal $288 by 2027-01-22)`.

## Account

- Robinhood **Agentic account 484265285 only**. Never touch any other account.
- The loop manages the ENTIRE account, including positions it did not open.
  Any holding may be sold to fund a better idea or cut a loser.

## Instruments & order types

- **Equities**: fractional-share MARKET orders, long only, no shorting.
  Fractional amounts allowed down to broker minimum (~$1 notional).
- **Options** (gated): each run, check `option_level` for account 484265285
  via get_accounts (fresh call, never cached). If empty or level_0 → equity
  only, skip options silently. If level_2+ → single-leg long calls/puts,
  covered calls, cash-secured puts allowed. NO multi-leg/spreads (MCP does
  not support them). Options use LIMIT orders at the bid/ask midpoint —
  never market orders on options. If unfilled after ~10 minutes, cancel and
  re-place once at slightly worse than mid; if still unfilled, cancel and log.
- Never: crypto, futures, event contracts, equity limit/stop/conditional
  orders.

## Sizing & risk

- Max 50% of current buying power on a single high-conviction BUY.
- Max 30% of current buying power on a single option contract (premium).
- Max 8 open positions total (options count as positions).
- Cut losers: SELL signal from pipeline → close full position same run.
  Position down >15% from cost WITH adverse news → close even without a
  pipeline run.
- Leave ≥$2 cash buffer after any buy.

## Candidate sourcing (undervalued hunt)

- Saved Robinhood scan **"Undervalued hunt"**, scan_id
  `80b034c8-a198-40de-801c-3dc046c5098b` (run via run_scan). Filters:
  P/E 0–18, RSI(14,1d) < 40, avg volume(30d) > 1M, market cap > $2B,
  price > $5.
- Scanner output is a candidate list, not a buy list. Web-validate top hits:
  WHY is it cheap? Skip value traps (secular decline, fraud/accounting
  clouds, broken balance sheet, dividend-cut spirals). Prefer names where
  cheapness looks temporary (sector rotation, overdone reaction to fixable
  news).
- Skip tickers already held unless re-running for a SELL decision.
- Closed-end funds / REITs with distorted P/E (PDI, GOF, ARR type) usually
  screen falsely cheap — treat with extra skepticism.

## Pipeline

- Analysis via `python scripts/run_pipeline.py --tickers <T,...> --slot <slot>`;
  decisions in `results/agentic/<date>-<slot>.json`. Signal mapping:
  Overweight = BUY, Underweight = SELL, anything else = HOLD.
- **Run the pipeline in the FOREGROUND and wait for it to finish** (it takes
  ~20 minutes per ticker — that is normal, keep waiting). NEVER run it as a
  background job and never end your reply while it runs: this session
  terminates the moment your reply ends, which kills the pipeline and skips
  the trade. The run is not done until orders are placed and logs written.
- **Before placing any orders, validate the decision file**:
  - Confirm it exists and is freshly produced by the current run (check
    `generated_at` is recent, within the last few minutes).
  - Confirm `date` and `slot` fields match what you requested.
  - Confirm `decisions` array contains exactly the tickers you requested.
  - If any validation fails or the run was interrupted, STOP without trading.
- Pipeline decision is the primary signal; sizing and instrument choice
  (shares vs calls) are the executing agent's judgment within these rules.

## Rotation (open run only)

- Each open run, identify the 2 holdings with the worst unrealized P&L %
  (exclude positions opened within the last 3 trading days). Pipeline them
  IN THE SAME RUN as the new scanner candidates.
- Execute rotation after reading all signals:
  - Holding = SELL → sell full position regardless of candidates.
  - Candidate = BUY but buying power is under the intended size → sell the
    weakest non-BUY holding (SELL first, then HOLD with worst P&L) to fund
    it. Never sell a holding rated BUY.
  - Max 2 rotation sells per day; swaps settle as market orders, sell first,
    then buy.
- Wash-sale awareness: rotation sells at a loss are fine, but do not rebuy
  a loss-sold ticker within 30 days; note every loss sale in pnl_report.md
  wash-sale watch.

## Logging & failure

- Append every run to `results/agentic/trade_log.md`: date/slot, candidates
  considered + why picked/skipped, signals, orders with confirmation IDs and
  fills, account value, one-line rationale for sizing.
- **Update `results/agentic/pnl_report.md` every run** (rewrite in place,
  keep the same section structure): realized P&L (get_pnl_trade_history,
  span=all), unrealized per open position (positions × live quotes), actual
  fees from fill records, estimated tax on realized gains (assume 24%
  federal short-term rate + 4.25% Michigan flat rate = 28.25% combined,
  unless the owner corrects it), and
  wash-sale flags (any buy within 30 days of a loss sale in the same
  ticker). Include progress vs the $288 goal.
- MCP auth/connectivity failure: log and stop. No endless retries.
