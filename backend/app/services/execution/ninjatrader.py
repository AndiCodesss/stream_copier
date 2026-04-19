from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.core.config import Settings
from app.models.domain import ExecutionResult, StreamSession, TradeIntent


class NinjaTraderBridgeClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=_bridge_timeout_seconds(settings),
            headers=_bridge_headers(settings),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def post_command(self, payload: dict[str, Any]) -> httpx.Response:
        return await self._request(
            "POST",
            "/api/stream-copier/commands",
            error_context="command request",
            json=payload,
        )

    async def fetch_state(
        self,
        *,
        account: str | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if account:
            params["account"] = account
        if symbol:
            params["symbol"] = symbol

        response = await self._request(
            "GET",
            "/api/stream-copier/state",
            error_context="state request",
            params=params,
            require_success_status=True,
        )
        return _decode_state_payload(response)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        error_context: str,
        require_success_status: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for base_url in _bridge_base_urls(self._settings):
            endpoint = f"{base_url}{path}"
            try:
                response = await self._client.request(method, endpoint, **kwargs)
                if require_success_status:
                    response.raise_for_status()
                return response
            except Exception as error:
                last_error = error

        raise RuntimeError(f"NinjaTrader bridge {error_context} failed: {last_error}") from last_error


class NinjaTraderExecutor:
    """Routes trade commands to a local NinjaTrader bridge HTTP service.

    The bridge is expected to expose POST /api/stream-copier/commands and return
    JSON (or plain text) with order acceptance details.
    """

    def __init__(self, settings: Settings, *, bridge_client: NinjaTraderBridgeClient) -> None:
        self._settings = settings
        self._bridge_client = bridge_client

    async def execute(self, session: StreamSession, intent: TradeIntent) -> ExecutionResult:
        payload = self._build_payload(session=session, intent=intent)
        try:
            response = await self._bridge_client.post_command(payload)
        except Exception as error:
            return ExecutionResult(
                session_id=session.id,
                action=intent.tag,
                approved=False,
                message=str(error),
                market_price=session.market.last_price,
            )

        bridge_message = self._extract_message(response)
        bridge_accepted = self._is_accepted_response(response)
        if not bridge_accepted:
            status_note = f" (HTTP {response.status_code})" if response.status_code >= 400 else ""
            return ExecutionResult(
                session_id=session.id,
                action=intent.tag,
                approved=False,
                message=f"NinjaTrader rejected: {bridge_message}{status_note}",
                market_price=session.market.last_price,
                position=session.position.model_copy(deep=True) if session.position else None,
            )

        return ExecutionResult(
            session_id=session.id,
            action=intent.tag,
            approved=True,
            message=f"NinjaTrader accepted: {bridge_message}",
            market_price=session.market.last_price,
            position=session.position.model_copy(deep=True) if session.position else None,
        )

    def _build_payload(self, *, session: StreamSession, intent: TradeIntent) -> dict[str, Any]:
        configured_account = (session.config.broker_account_override or self._settings.ninjatrader_account or "").strip()
        configured_symbol = (session.config.broker_symbol_override or self._settings.ninjatrader_symbol or "").strip()
        resolved_symbol = configured_symbol or intent.symbol or session.market.symbol
        return {
            "intent_id": intent.id,
            "session_id": session.id,
            "account": configured_account or None,
            "symbol": resolved_symbol,
            "action": _as_value(intent.tag),
            "side": _as_value(intent.side),
            "quantity_hint": intent.quantity_hint,
            "default_contract_size": session.config.default_contract_size,
            "time_in_force": self._settings.ninjatrader_time_in_force,
            "entry_price": intent.entry_price,
            "stop_price": intent.stop_price,
            "target_price": intent.target_price,
            "market_price": session.market.last_price,
            "evidence_text": intent.evidence_text,
            "sent_at": datetime.now(UTC).isoformat(),
        }

    def _extract_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            text = response.text.strip()
            return text[:200] if text else "ok"

        if isinstance(payload, dict):
            for key in ("message", "status", "result", "orderId", "order_id"):
                value = payload.get(key)
                if value:
                    return str(value)
        return "ok"

    def _is_accepted_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False

        try:
            payload = response.json()
        except Exception:
            return True

        if isinstance(payload, dict) and "ok" in payload:
            return bool(payload.get("ok"))

        return True


def _decode_state_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        text = response.text.strip()
        return {"ok": True, "code": "ok", "message": text[:200] if text else "ok"}

    if isinstance(payload, dict):
        return payload

    return {"ok": True, "code": "ok", "message": str(payload)}


def _as_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _bridge_timeout_seconds(settings: Settings) -> float:
    return max(0.25, settings.ninjatrader_bridge_timeout_ms / 1000)


def _bridge_headers(settings: Settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.ninjatrader_bridge_token:
        headers["Authorization"] = f"Bearer {settings.ninjatrader_bridge_token}"
    return headers


def _bridge_base_urls(settings: Settings) -> list[str]:
    primary = settings.ninjatrader_bridge_url.rstrip("/")
    urls = [primary]
    fallback = _wsl_windows_host_url(primary)
    if fallback and fallback not in urls:
        urls.append(fallback)
    return urls


def _wsl_windows_host_url(url: str) -> str | None:
    if not os.environ.get("WSL_DISTRO_NAME"):
        return None

    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return None

    windows_host_ip = _resolve_wsl_windows_host_ip()
    if not windows_host_ip:
        return None

    port = parsed.port
    netloc = f"{windows_host_ip}:{port}" if port is not None else windows_host_ip
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", "")).rstrip("/")


def _resolve_wsl_windows_host_ip() -> str | None:
    resolv_conf = Path("/etc/resolv.conf")
    if not resolv_conf.exists():
        return None

    try:
        for line in resolv_conf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("nameserver "):
                continue
            candidate = line.split(maxsplit=1)[1].strip()
            if candidate:
                return candidate
    except Exception:
        return None

    return None
