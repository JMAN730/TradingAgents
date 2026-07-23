"""Unit tests for the QuantDinger execution bridge (no network)."""

import pytest

from tradingagents.integrations.quantdinger import (
    SIGNAL_ORDERS,
    QuantDingerClient,
    QuantDingerError,
    QuantDingerExecutor,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Records requests and replays canned responses keyed by (method, path)."""

    def __init__(self):
        self.responses = {}
        self.calls = []

    def stub(self, method, path_suffix, payload, status_code=200):
        self.responses[(method, path_suffix)] = FakeResponse(payload, status_code)

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append(
            {"method": method, "url": url, "params": params, "json": json, "headers": headers}
        )
        for (m, suffix), response in self.responses.items():
            if m == method and url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected request: {method} {url}")


@pytest.fixture()
def session():
    return FakeSession()


@pytest.fixture()
def client(session):
    return QuantDingerClient("http://qd.local:5000", "tok123", session=session)


def _ok(data):
    return {"code": 0, "message": "ok", "data": data}


def test_client_requires_token():
    with pytest.raises(ValueError):
        QuantDingerClient("http://qd.local:5000", "")


def test_bearer_auth_header(client, session):
    session.stub("GET", "/price", _ok({"price": 10.0}))
    client.get_price("AAPL")
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok123"
    assert session.calls[0]["url"] == "http://qd.local:5000/api/agent/v1/price"


def test_get_price_none_when_missing(client, session):
    session.stub("GET", "/price", _ok({"price": None}))
    assert client.get_price("AAPL") is None


def test_error_envelope_raises(client, session):
    session.stub("GET", "/price", {"code": 403, "message": "Market not allowed"}, status_code=403)
    with pytest.raises(QuantDingerError, match="Market not allowed"):
        client.get_price("AAPL")


def test_place_order_sends_idempotency_key(client, session):
    session.stub("POST", "/quick-trade/orders", _ok({"status": "filled", "paper": True}))
    client.place_order(
        market="USStock", symbol="AAPL", side="buy", qty=2.5, idempotency_key="ta-x",
    )
    call = session.calls[0]
    assert call["headers"]["Idempotency-Key"] == "ta-x"
    assert call["json"] == {
        "market": "USStock", "symbol": "AAPL", "side": "buy",
        "qty": 2.5, "order_type": "market",
    }


def test_signal_map_covers_all_five_tiers():
    assert set(SIGNAL_ORDERS) == {"buy", "overweight", "hold", "underweight", "sell"}


@pytest.mark.parametrize(
    ("signal", "side", "qty"),
    [
        ("Buy", "buy", 10.0),          # 1000 / 100
        ("Overweight", "buy", 5.0),    # half notional
        ("Underweight", "sell", 5.0),
        ("Sell", "sell", 10.0),
    ],
)
def test_plan_order_sizing(client, session, signal, side, qty):
    session.stub("GET", "/price", _ok({"price": 100.0}))
    executor = QuantDingerExecutor(client=client, order_notional=1000.0)
    plan = executor.plan_order("AAPL", signal)
    assert plan["side"] == side
    assert plan["qty"] == qty
    assert plan["market"] == "USStock"


def test_hold_produces_no_order(client, session):
    executor = QuantDingerExecutor(client=client)
    assert executor.plan_order("AAPL", "Hold") is None
    receipt = executor.execute("AAPL", "Hold", trade_date="2026-07-23", slot="open")
    assert receipt["action"] == "none"
    assert session.calls == []  # no price fetch, no order


def test_unknown_signal_raises(client):
    executor = QuantDingerExecutor(client=client)
    with pytest.raises(QuantDingerError, match="unrecognized signal"):
        executor.plan_order("AAPL", "Moon")


def test_missing_price_raises(client, session):
    session.stub("GET", "/price", _ok({"price": None}))
    executor = QuantDingerExecutor(client=client)
    with pytest.raises(QuantDingerError, match="no price"):
        executor.plan_order("AAPL", "Buy")


def test_execute_uses_slot_scoped_idempotency_key(client, session):
    session.stub("GET", "/price", _ok({"price": 200.0}))
    session.stub("POST", "/quick-trade/orders", _ok({"status": "filled", "paper": True}))
    executor = QuantDingerExecutor(client=client, order_notional=500.0)
    receipt = executor.execute("NVDA", "Buy", trade_date="2026-07-23", slot="open")
    order_call = session.calls[-1]
    assert order_call["headers"]["Idempotency-Key"] == "ta-2026-07-23-open-NVDA-buy"
    assert receipt["planned_qty"] == 2.5
    assert receipt["order"]["paper"] is True
