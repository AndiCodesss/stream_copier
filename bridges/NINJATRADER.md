# NinjaTrader Bridge AddOn

This folder contains a NinjaTrader 8 AddOn that exposes:

- `GET /api/stream-copier/health`
- `GET /api/stream-copier/state?account=...&symbol=...`
- `POST /api/stream-copier/commands`

It is designed to pair with this backend's `EXECUTION_BACKEND=ninjatrader` mode.

## Install

1. Copy [StreamCopierBridgeAddOn.cs](/mnt/c/Users/Andreas Oberdörfer/Downloads/stream_copier/bridges/ninjatrader/StreamCopierBridgeAddOn.cs) to:
`Documents\NinjaTrader 8\bin\Custom\AddOns\StreamCopierBridgeAddOn.cs`
2. Open NinjaTrader 8.
3. Open `New -> NinjaScript Editor`.
4. Press `Compile`.
5. Restart NinjaTrader once after successful compile.

## Configure AddOn

Edit constants at the top of `StreamCopierBridgeAddOn.cs`:

- `ListenerPrefix` (default: `http://127.0.0.1:18080/`)
- `RequiredBearerToken` (must match backend `NINJATRADER_BRIDGE_TOKEN`)
- `DefaultAccountName` (optional fallback account if command payload does not provide one)

## Configure backend

In `backend/.env`:

```bash
EXECUTION_BACKEND=ninjatrader
LIVE_EXECUTION_ENABLED=true
NINJATRADER_BRIDGE_URL=http://127.0.0.1:18080
NINJATRADER_BRIDGE_TOKEN=your_shared_secret
NINJATRADER_ACCOUNT=Sim101
NINJATRADER_TIME_IN_FORCE=Day
NINJATRADER_SHADOW_SYNC=true
```

## URL ACL (if listener fails to start)

Run in elevated Command Prompt:

```bat
netsh http add urlacl url=http://127.0.0.1:18080/ user=%USERNAME%
```

To remove later:

```bat
netsh http delete urlacl url=http://127.0.0.1:18080/
```

### WSL backend + CUDA setup

If backend runs inside WSL and NinjaTrader runs on Windows:

1. Use wildcard listener in AddOn (`ListenerPrefix = "http://*:18080/"`).
2. Run these in elevated Command Prompt (Windows):

```bat
netsh http add urlacl url=http://*:18080/ user=%USERNAME%
netsh advfirewall firewall add rule name="NinjaTrader Bridge 18080" dir=in action=allow protocol=TCP localport=18080
```

3. In WSL, get Windows host IP:

```bash
ip route | awk '/default/ {print $3}'
```

4. Set backend env:

```bash
NINJATRADER_BRIDGE_URL=http://<windows-host-ip>:18080
```

## Smoke test

Health:

```bash
curl -s http://127.0.0.1:18080/api/stream-copier/health
```

State (entry/position/market snapshot + PnL):

```bash
curl -s "http://127.0.0.1:18080/api/stream-copier/state?account=Playback101&symbol=MNQ%2003-26"
```

Command:

```bash
curl -s -X POST http://127.0.0.1:18080/api/stream-copier/commands \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_shared_secret" \
  -d '{
    "intent_id": "demo123",
    "session_id": "demo-session",
    "account": "Sim101",
    "symbol": "NQ 06-26",
    "action": "ENTER_SHORT",
    "side": "SHORT",
    "default_contract_size": 1,
    "time_in_force": "Day",
    "stop_price": 21260.0
  }'
```

## Notes

- Use exact NinjaTrader instruments (for example: `NQ 06-26`), not generic `NQ`.
- Start in `Sim101` first.
- The AddOn submits entries/adds/trims/flatten and manages protective stop/target orders when stop/target is provided.
