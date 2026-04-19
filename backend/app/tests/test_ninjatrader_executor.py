from __future__ import annotations

from datetime import UTC, datetime

import httpx

from app.core.config import Settings
from app.models.domain import ActionTag, MarketSnapshot, SessionConfig, StreamSession, TradeIntent, TradeSide
from app.services.execution.ninjatrader import NinjaTraderBridgeClient, NinjaTraderExecutor


class _FakeBridgeClient:
    def __init__(
        self,
        *,
        response: httpx.Response | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.payloads: list[dict[str, object]] = []

    async def post_command(self, payload: dict[str, object]) -> httpx.Response:
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeAsyncClient:
    def __init__(self, *, response: httpx.Response, capture: dict[str, object], **kwargs: object) -> None:
        self._response = response
        self._capture = capture
        self._capture["init"] = kwargs

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        self._capture["method"] = method
        self._capture["url"] = url
        self._capture["kwargs"] = kwargs
        return self._response

    async def aclose(self) -> None:
        return None


def _sample_intent(session_id: str) -> TradeIntent:
    return TradeIntent(
        session_id=session_id,
        tag=ActionTag.enter_short,
        side=TradeSide.short,
        entry_price=21243.75,
        stop_price=21260.0,
        evidence_text="In short there for a small piece. Short versus 60s.",
        confidence=0.95,
        created_at=datetime.now(UTC),
    )


async def test_ninjatrader_executor_posts_command() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:18080/api/stream-copier/commands")
    response = httpx.Response(status_code=200, json={"message": "accepted"}, request=request)
    bridge_client = _FakeBridgeClient(response=response)
    settings = Settings(
        ninjatrader_account="Playback101",
        ninjatrader_symbol="MNQ 03-26",
        ninjatrader_time_in_force="Day",
    )
    executor = NinjaTraderExecutor(settings, bridge_client=bridge_client)  # type: ignore[arg-type]
    session = StreamSession(
        config=SessionConfig(
            broker_account_override="Sim101",
            broker_symbol_override="MNQ 06-26",
        ),
        market=MarketSnapshot(symbol="NQ", last_price=21243.75),
    )
    intent = _sample_intent(session.id)

    result = await executor.execute(session, intent)

    assert result.approved is True
    assert len(bridge_client.payloads) == 1
    payload = bridge_client.payloads[0]
    assert payload["account"] == "Sim101"
    assert payload["symbol"] == "MNQ 06-26"
    assert payload["action"] == ActionTag.enter_short.value
    assert payload["side"] == TradeSide.short.value
    assert payload["time_in_force"] == "Day"
    assert payload["stop_price"] == 21260.0
    assert payload["target_price"] is None


async def test_ninjatrader_executor_reports_transport_failure() -> None:
    settings = Settings()
    executor = NinjaTraderExecutor(
        settings,
        bridge_client=_FakeBridgeClient(error=RuntimeError("bridge offline")),  # type: ignore[arg-type]
    )
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))
    intent = _sample_intent(session.id)

    result = await executor.execute(session, intent)

    assert result.approved is False
    assert "bridge offline" in result.message


async def test_ninjatrader_executor_rejects_when_bridge_payload_ok_false() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:18080/api/stream-copier/commands")
    response = httpx.Response(
        status_code=200,
        json={"ok": False, "code": "instrument_not_found", "message": "Instrument not found: NQ"},
        request=request,
    )
    executor = NinjaTraderExecutor(
        Settings(),
        bridge_client=_FakeBridgeClient(response=response),  # type: ignore[arg-type]
    )
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))
    intent = _sample_intent(session.id)

    result = await executor.execute(session, intent)

    assert result.approved is False
    assert "NinjaTrader rejected" in result.message
    assert "Instrument not found: NQ" in result.message


async def test_ninjatrader_executor_rejects_http_400() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:18080/api/stream-copier/commands")
    response = httpx.Response(
        status_code=400,
        json={"ok": False, "code": "instrument_not_found", "message": "Instrument not found: NQ"},
        request=request,
    )
    executor = NinjaTraderExecutor(
        Settings(),
        bridge_client=_FakeBridgeClient(response=response),  # type: ignore[arg-type]
    )
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))
    intent = _sample_intent(session.id)

    result = await executor.execute(session, intent)

    assert result.approved is False
    assert "NinjaTrader rejected" in result.message
    assert "HTTP 400" in result.message


async def test_bridge_client_fetch_state_forwards_query_params(monkeypatch) -> None:
    settings = Settings(
        ninjatrader_bridge_url="http://127.0.0.1:18080",
        ninjatrader_bridge_token="token123",
    )
    capture: dict[str, object] = {}
    request = httpx.Request("GET", "http://127.0.0.1:18080/api/stream-copier/state")
    response = httpx.Response(
        status_code=200,
        json={"ok": True, "code": "state", "account": "Playback101"},
        request=request,
    )

    monkeypatch.setattr(
        "app.services.execution.ninjatrader.httpx.AsyncClient",
        lambda **kwargs: _FakeAsyncClient(response=response, capture=capture, **kwargs),
    )

    client = NinjaTraderBridgeClient(settings)
    try:
        result = await client.fetch_state(
            account="Playback101",
            symbol="MNQ 03-26",
        )
    finally:
        await client.close()

    assert result["ok"] is True
    assert capture["method"] == "GET"
    assert capture["url"] == "http://127.0.0.1:18080/api/stream-copier/state"
    assert capture["kwargs"] == {
        "params": {
            "account": "Playback101",
            "symbol": "MNQ 03-26",
        }
    }
    assert capture["init"] == {
        "timeout": 2.5,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        },
    }
