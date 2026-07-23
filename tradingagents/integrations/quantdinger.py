"""QuantDinger execution bridge.

Routes TradingAgents portfolio decisions into a QuantDinger deployment through
its Agent Gateway (``/api/agent/v1``). QuantDinger owns execution: orders are
recorded as paper fills unless the operator has explicitly unlocked live
trading server-side (token scope ``T`` + ``paper_only=false`` on the token +
``AGENT_LIVE_TRADING_ENABLED=true`` in the backend environment). This module
never handles broker or exchange credentials.

Signal semantics: the five-tier portfolio rating maps to a side and a
conviction weight. ``Hold`` produces no order; ``Overweight``/``Underweight``
trade at half the notional of ``Buy``/``Sell``.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import requests

AGENT_API_PREFIX = "/api/agent/v1"

# rating -> (order side or None, fraction of the configured order notional)
SIGNAL_ORDERS: dict[str, tuple[str | None, float]] = {
    "buy": ("buy", 1.0),
    "overweight": ("buy", 0.5),
    "hold": (None, 0.0),
    "underweight": ("sell", 0.5),
    "sell": ("sell", 1.0),
}


class QuantDingerError(RuntimeError):
    """Raised when the QuantDinger Agent Gateway rejects or fails a request."""


class QuantDingerClient:
    """Minimal authenticated client for the QuantDinger Agent Gateway.

    Every response uses the agent envelope ``{"code": 0, "message", "data"}``;
    a non-zero ``code`` or HTTP error surfaces as :class:`QuantDingerError`.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 30.0,
        session: Any | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not token:
            raise ValueError("token is required (issue one under QuantDinger's Agent Tokens)")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"Authorization": f"Bearer {token}"}

    @classmethod
    def from_env(cls, **kwargs: Any) -> QuantDingerClient:
        """Build a client from QUANTDINGER_BASE_URL / QUANTDINGER_AGENT_TOKEN."""
        base_url = os.getenv("QUANTDINGER_BASE_URL", "http://localhost:5000")
        token = os.getenv("QUANTDINGER_AGENT_TOKEN", "")
        if not token:
            raise QuantDingerError(
                "QUANTDINGER_AGENT_TOKEN is not set; create an agent token in "
                "QuantDinger (Settings -> Agent Tokens) with scopes R+T"
            )
        return cls(base_url, token, **kwargs)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        headers = dict(self._headers)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{AGENT_API_PREFIX}{path}",
                params=params,
                json=json_body,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise QuantDingerError(f"request to {path} failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise QuantDingerError(
                f"{path} returned non-JSON response (HTTP {response.status_code})"
            ) from exc
        if response.status_code >= 400 or payload.get("code"):
            raise QuantDingerError(
                f"{path} failed (HTTP {response.status_code}, code {payload.get('code')}): "
                f"{payload.get('message')}"
            )
        return payload.get("data")

    def get_price(self, symbol: str, market: str = "USStock") -> float | None:
        data = self._request("GET", "/price", params={"market": market, "symbol": symbol})
        price = (data or {}).get("price")
        return float(price) if price is not None else None

    def place_order(
        self,
        *,
        market: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "market": market,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": order_type,
        }
        if limit_price is not None:
            body["limit_price"] = limit_price
        return self._request(
            "POST",
            "/quick-trade/orders",
            json_body=body,
            idempotency_key=idempotency_key or f"ta-{uuid.uuid4().hex}",
        )

    def kill_switch(self) -> dict:
        """Cancel all of this token's open paper orders."""
        return self._request("POST", "/quick-trade/kill-switch", json_body={})


@dataclass
class QuantDingerExecutor:
    """Turns five-tier TradingAgents signals into sized QuantDinger orders.

    ``order_notional`` is the quote-currency value of a full-conviction (Buy or
    Sell) order; Overweight/Underweight trade half of it. Quantity is derived
    from the gateway's latest price, so the executor needs a token with scope
    ``R`` (price reads) in addition to ``T`` (trading).
    """

    client: QuantDingerClient
    market: str = "USStock"
    order_notional: float = 1000.0
    qty_precision: int = 6

    def plan_order(self, ticker: str, signal: str) -> dict | None:
        """Return an order plan for a signal, or None when no trade is implied."""
        normalized = str(signal or "").strip().lower()
        if normalized not in SIGNAL_ORDERS:
            raise QuantDingerError(f"unrecognized signal for {ticker}: {signal!r}")
        side, weight = SIGNAL_ORDERS[normalized]
        if side is None:
            return None
        price = self.client.get_price(ticker, market=self.market)
        if not price or price <= 0:
            raise QuantDingerError(f"no price available for {ticker} in {self.market}")
        qty = round(self.order_notional * weight / price, self.qty_precision)
        if qty <= 0:
            raise QuantDingerError(
                f"computed qty for {ticker} is zero (notional {self.order_notional}, price {price})"
            )
        return {
            "market": self.market,
            "symbol": ticker,
            "side": side,
            "qty": qty,
            "reference_price": price,
        }

    def execute(
        self,
        ticker: str,
        signal: str,
        *,
        trade_date: str = "",
        slot: str = "adhoc",
    ) -> dict:
        """Execute one decision; returns a receipt describing what happened.

        The idempotency key is derived from (date, slot, ticker, side) so
        re-running the same pipeline slot replays the recorded order instead of
        doubling the position.
        """
        plan = self.plan_order(ticker, signal)
        if plan is None:
            return {"ticker": ticker, "signal": signal, "action": "none", "order": None}
        idempotency_key = f"ta-{trade_date}-{slot}-{ticker}-{plan['side']}"
        order = self.client.place_order(
            market=plan["market"],
            symbol=plan["symbol"],
            side=plan["side"],
            qty=plan["qty"],
            idempotency_key=idempotency_key,
        )
        return {
            "ticker": ticker,
            "signal": signal,
            "action": plan["side"],
            "planned_qty": plan["qty"],
            "reference_price": plan["reference_price"],
            "order": order,
        }
