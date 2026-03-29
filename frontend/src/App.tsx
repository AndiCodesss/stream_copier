import { useEffect, useMemo, useRef, useState } from "react";

import {
  audioSocketUrl,
  createSession,
  deleteSession,
  eventSocketUrl,
  getBrokerState,
  listSessions,
  sendManualSegment,
  sendManualTrade,
  updateSessionConfig,
} from "./api";
import type {
  BrokerState,
  SessionPatch,
  StreamSession,
  TimelineEvent,
  TradeIntent,
  TranscriptionMetrics,
  WsMessage,
} from "./types";

const AUDIO_CHUNK_TARGET_MS = 30;
const AUDIO_RECONNECT_BASE_MS = 750;
const AUDIO_RECONNECT_MAX_MS = 3_000;
const BROKER_POLL_INTERVAL_ACTIVE_MS = 500;
const BROKER_POLL_INTERVAL_POSITION_MS = 750;
const BROKER_POLL_INTERVAL_IDLE_MS = 2_000;
const TRANSCRIPTION_MODELS = [
  { value: "distil-small.en", label: "distil-small.en" },
  { value: "distil-large-v3", label: "distil-large-v3" },
  { value: "small.en", label: "small.en" },
];

function App() {
  const [sessions, setSessions] = useState<StreamSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [manualText, setManualText] = useState("I'm long here at 46, stop under 36, first trim at 58");
  const [captureStatus, setCaptureStatus] = useState("idle");
  const [captureStatusDetail, setCaptureStatusDetail] = useState("Ready to capture tab audio.");
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [updatingConfigKey, setUpdatingConfigKey] = useState<string | null>(null);
  const [brokerAccount, setBrokerAccount] = useState("");
  const [brokerSymbol, setBrokerSymbol] = useState("");
  const [brokerState, setBrokerState] = useState<BrokerState | null>(null);
  const [brokerStateError, setBrokerStateError] = useState<string | null>(null);
  const [isInjectingTranscript, setIsInjectingTranscript] = useState(false);
  const [injectTranscriptError, setInjectTranscriptError] = useState<string | null>(null);
  const [manualContractSize, setManualContractSize] = useState(3);
  const [isSubmittingManualTrade, setIsSubmittingManualTrade] = useState(false);
  const [manualTradeError, setManualTradeError] = useState<string | null>(null);
  const [brokerRefreshNonce, setBrokerRefreshNonce] = useState(0);

  const eventSocketRef = useRef<WebSocket | null>(null);
  const audioSocketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const muteNodeRef = useRef<GainNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const stopRequestedRef = useRef(false);
  const captureSessionIdRef = useRef<string | null>(null);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? null,
    [activeSessionId, sessions],
  );
  const latencyState = useMemo(() => getLatencyState(activeSession), [activeSession]);
  const hasLiveBrokerState = brokerState?.ok === true;
  const hasLiveBrokerPosition = useMemo(() => {
    if (brokerState?.ok !== true) {
      return false;
    }
    if (typeof brokerState.has_position === "boolean") {
      return brokerState.has_position;
    }
    return brokerState.market_position === "LONG" || brokerState.market_position === "SHORT";
  }, [brokerState]);
  const brokerPollIntervalMs = useMemo(() => {
    if (captureStatus === "capturing" || captureStatus === "connecting" || captureStatus === "reconnecting") {
      return BROKER_POLL_INTERVAL_ACTIVE_MS;
    }
    if (hasLiveBrokerPosition) {
      return BROKER_POLL_INTERVAL_POSITION_MS;
    }
    return BROKER_POLL_INTERVAL_IDLE_MS;
  }, [captureStatus, hasLiveBrokerPosition]);

  useEffect(() => {
    void listSessions().then((result) => {
      setSessions(result);
      if (result[0]) {
        setActiveSessionId(result[0].id);
      }
    });
  }, []);

  useEffect(() => {
    if (sessions.length === 0) {
      if (activeSessionId) {
        setActiveSessionId("");
      }
      return;
    }

    if (!sessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(sessions[0].id);
    }
  }, [activeSessionId, sessions]);

  useEffect(() => {
    if (!activeSessionId) {
      return undefined;
    }

    eventSocketRef.current?.close();
    const socket = new WebSocket(eventSocketUrl(activeSessionId));
    eventSocketRef.current = socket;

    socket.onmessage = (rawEvent) => {
      const msg = JSON.parse(rawEvent.data) as WsMessage;
      if (msg.type === "snapshot") {
        setSessions((current) => upsertSession(current, msg.session));
      } else {
        setSessions((current) => {
          const session = current.find((s) => s.id === activeSessionId);
          if (!session) return current;
          return upsertSession(current, applyPatch(session, msg.event, msg.patch, msg.append_event));
        });
      }
    };

    return () => {
      socket.close();
    };
  }, [activeSessionId]);

  useEffect(() => {
    if (!activeSession) {
      setBrokerAccount("");
      setBrokerSymbol("");
      return;
    }

    setBrokerAccount(activeSession.config.broker_account_override ?? "");
    setBrokerSymbol(activeSession.config.broker_symbol_override ?? "");
  }, [
    activeSession?.id,
    activeSession?.config.broker_account_override,
    activeSession?.config.broker_symbol_override,
  ]);

  useEffect(() => {
    if (!activeSessionId || !activeSession) {
      return undefined;
    }

    const nextAccount = brokerAccount.trim() || null;
    const nextSymbol = brokerSymbol.trim() || null;
    const currentAccount = activeSession.config.broker_account_override ?? null;
    const currentSymbol = activeSession.config.broker_symbol_override ?? null;
    if (nextAccount === currentAccount && nextSymbol === currentSymbol) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      void handleUpdateSessionConfig(
        {
          broker_account_override: nextAccount,
          broker_symbol_override: nextSymbol,
        },
        "broker-routing",
      );
    }, 300);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [activeSession, activeSessionId, brokerAccount, brokerSymbol]);

  useEffect(() => {
    if (!activeSessionId) {
      setBrokerState(null);
      setBrokerStateError(null);
      return undefined;
    }

    let closed = false;
    let timerId: number | undefined;

    const poll = async (): Promise<void> => {
      try {
        const state = await getBrokerState(activeSessionId, {
          account: brokerAccount,
          symbol: brokerSymbol,
        });
        if (closed) {
          return;
        }
        setBrokerState(state);
        if (state.ok) {
          setBrokerStateError(null);
        } else {
          setBrokerStateError(state.message || "Broker telemetry unavailable.");
        }
      } catch (error) {
        if (closed) {
          return;
        }
        setBrokerStateError(error instanceof Error ? error.message : String(error));
      } finally {
        if (!closed) {
          timerId = window.setTimeout(() => {
            void poll();
          }, brokerPollIntervalMs);
        }
      }
    };

    void poll();

    return () => {
      closed = true;
      if (timerId !== undefined) {
        window.clearTimeout(timerId);
      }
    };
  }, [activeSessionId, brokerAccount, brokerPollIntervalMs, brokerRefreshNonce, brokerSymbol]);

  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
    };
  }, []);

  function clearReconnectTimer(): void {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }

  function requestBrokerRefresh(): void {
    setBrokerRefreshNonce((current) => current + 1);
  }

  function hasLiveAudioTrack(): boolean {
    return (mediaStreamRef.current?.getAudioTracks() ?? []).some((track) => track.readyState === "live");
  }

  function describeAudioClose(event: CloseEvent): string {
    if (event.reason) {
      return event.reason;
    }
    if (event.code === 1006) {
      return "Audio connection dropped. Backend may have restarted.";
    }
    if (event.code === 1011) {
      return "Backend audio ingest failed.";
    }
    if (event.code === 4404) {
      return "Capture session no longer exists on the backend.";
    }
    if (event.code === 1000) {
      return "Audio connection closed.";
    }
    return `Audio connection closed (code ${event.code}).`;
  }

  function resetSocketOnly(): void {
    audioSocketRef.current = null;
  }

  function scheduleAudioReconnect(reason: string): void {
    clearReconnectTimer();
    if (stopRequestedRef.current || !hasLiveAudioTrack() || !audioContextRef.current || !workletNodeRef.current) {
      setCaptureStatus("stopped");
      setCaptureStatusDetail(reason);
      return;
    }

    reconnectAttemptRef.current += 1;
    const delayMs = Math.min(AUDIO_RECONNECT_MAX_MS, AUDIO_RECONNECT_BASE_MS * reconnectAttemptRef.current);
    setCaptureStatus("reconnecting");
    setCaptureStatusDetail(`${reason} Reconnecting in ${(delayMs / 1000).toFixed(1)}s...`);
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      const sessionId = captureSessionIdRef.current;
      if (!sessionId || stopRequestedRef.current || !hasLiveAudioTrack()) {
        setCaptureStatus("stopped");
        setCaptureStatusDetail(reason);
        return;
      }
      void connectAudioSocket(sessionId, { reconnecting: true });
    }, delayMs);
  }

  async function teardownCapture(reason: string): Promise<void> {
    stopRequestedRef.current = true;
    clearReconnectTimer();

    const socket = audioSocketRef.current;
    resetSocketOnly();
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      socket.close(1000, "Capture stopped");
    }

    workletNodeRef.current?.disconnect();
    if (workletNodeRef.current) {
      workletNodeRef.current.port.onmessage = null;
    }
    workletNodeRef.current = null;
    sourceNodeRef.current?.disconnect();
    sourceNodeRef.current = null;
    muteNodeRef.current?.disconnect();
    muteNodeRef.current = null;

    const stream = mediaStreamRef.current;
    mediaStreamRef.current = null;
    stream?.getTracks().forEach((track) => {
      track.onended = null;
      track.stop();
    });

    try {
      await audioContextRef.current?.close();
    } catch {
      // Ignore close failures when the context is already shutting down.
    }
    audioContextRef.current = null;
    captureSessionIdRef.current = null;
    reconnectAttemptRef.current = 0;
    setCaptureStatus("stopped");
    setCaptureStatusDetail(reason);
  }

  async function connectAudioSocket(sessionId: string, options?: { reconnecting?: boolean }): Promise<void> {
    const audioContext = audioContextRef.current;
    if (!audioContext) {
      setCaptureStatus("stopped");
      setCaptureStatusDetail("Audio context is unavailable.");
      return;
    }

    const existingSocket = audioSocketRef.current;
    if (existingSocket && (existingSocket.readyState === WebSocket.OPEN || existingSocket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setCaptureStatus(options?.reconnecting ? "reconnecting" : "connecting");
    setCaptureStatusDetail(options?.reconnecting ? "Reconnecting audio stream..." : "Connecting audio stream...");

    const socket = new WebSocket(audioSocketUrl(sessionId));
    audioSocketRef.current = socket;

    socket.onopen = () => {
      if (audioSocketRef.current !== socket || stopRequestedRef.current) {
        socket.close(1000, "Capture stopped");
        return;
      }
      reconnectAttemptRef.current = 0;
      socket.send(
        JSON.stringify({
          type: "audio_config",
          sample_rate: audioContext.sampleRate,
        }),
      );
      setCaptureStatus("capturing");
      setCaptureStatusDetail("Capturing tab audio.");
    };

    socket.onerror = () => {
      if (!stopRequestedRef.current) {
        setCaptureStatusDetail("Audio connection error. Waiting to reconnect...");
      }
    };

    socket.onclose = (event) => {
      if (audioSocketRef.current === socket) {
        resetSocketOnly();
      }
      if (stopRequestedRef.current) {
        return;
      }
      scheduleAudioReconnect(describeAudioClose(event));
    };
  }

  async function handleCreateSession(): Promise<void> {
    setIsCreatingSession(true);
    try {
      if (captureStatus === "capturing" || captureStatus === "connecting" || captureStatus === "reconnecting") {
        await handleStopCapture();
      }
      const session = await createSession();
      setSessions((current) => upsertSession(current, session, { prepend: true }));
      setActiveSessionId(session.id);
    } finally {
      setIsCreatingSession(false);
    }
  }

  async function handleDeleteSession(sessionId: string): Promise<void> {
    setDeletingSessionId(sessionId);
    try {
      if (
        sessionId === activeSessionId
        && (captureStatus === "capturing" || captureStatus === "connecting" || captureStatus === "reconnecting")
      ) {
        await handleStopCapture();
      }
      await deleteSession(sessionId);
      setSessions((current) => current.filter((session) => session.id !== sessionId));
    } finally {
      setDeletingSessionId(null);
    }
  }

  async function handleSelectSession(sessionId: string): Promise<void> {
    if (sessionId === activeSessionId) {
      return;
    }
    if (captureStatus === "capturing" || captureStatus === "connecting" || captureStatus === "reconnecting") {
      await handleStopCapture();
    }
    setActiveSessionId(sessionId);
  }

  async function handleSendManual(): Promise<void> {
    if (!activeSessionId) {
      return;
    }
    if (!manualText.trim()) {
      setInjectTranscriptError("Transcript text is empty.");
      return;
    }

    setIsInjectingTranscript(true);
    setInjectTranscriptError(null);
    try {
      const session = await sendManualSegment(activeSessionId, manualText);
      setSessions((current) => upsertSession(current, session));
      requestBrokerRefresh();
    } catch (error) {
      setInjectTranscriptError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsInjectingTranscript(false);
    }
  }

  async function handleManualTrade(action: "BUY" | "SELL" | "CLOSE"): Promise<void> {
    if (!activeSessionId) {
      return;
    }
    const size = Math.max(1, Math.min(10, Math.floor(manualContractSize || 0)));
    if (!Number.isFinite(size) || size < 1) {
      setManualTradeError("Contract size must be between 1 and 10.");
      return;
    }

    setIsSubmittingManualTrade(true);
    setManualTradeError(null);
    try {
      const session = await sendManualTrade(activeSessionId, action, size, {
        account: brokerAccount,
        symbol: brokerSymbol,
      });
      setSessions((current) => upsertSession(current, session));
      requestBrokerRefresh();
    } catch (error) {
      setManualTradeError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSubmittingManualTrade(false);
    }
  }

  async function handleUpdateSessionConfig(
    patch: {
      enable_partial_intent_detection?: boolean;
      enable_ai_fallback?: boolean;
      enable_early_preview_entries?: boolean;
      transcription_model?: string;
      broker_account_override?: string | null;
      broker_symbol_override?: string | null;
    },
    reason: string,
  ): Promise<void> {
    if (!activeSessionId) {
      return;
    }
    setUpdatingConfigKey(reason);
    try {
      const session = await updateSessionConfig(activeSessionId, patch);
      setSessions((current) => upsertSession(current, session));
      requestBrokerRefresh();
    } finally {
      setUpdatingConfigKey(null);
    }
  }

  async function handleStartCapture(): Promise<void> {
    if (!activeSessionId) {
      return;
    }
    stopRequestedRef.current = false;
    captureSessionIdRef.current = activeSessionId;
    reconnectAttemptRef.current = 0;
    clearReconnectTimer();
    setCaptureStatus("connecting");
    setCaptureStatusDetail("Waiting for tab audio share...");

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: true,
      });

      // We only need tab audio for transcription. Stop display video track
      // immediately to avoid unnecessary capture/render overhead.
      stream.getVideoTracks().forEach((track) => track.stop());

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        stream.getTracks().forEach((track) => track.stop());
        setCaptureStatus("stopped");
        setCaptureStatusDetail("No tab audio track was shared.");
        captureSessionIdRef.current = null;
        return;
      }

      mediaStreamRef.current = stream;
      for (const track of audioTracks) {
        track.onended = () => {
          if (stopRequestedRef.current) {
            return;
          }
          void teardownCapture("Tab audio sharing ended.");
        };
      }

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      await audioContext.audioWorklet.addModule("/audio-processor.js");

      const sourceNode = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = sourceNode;
      const frameThreshold = Math.max(2048, Math.floor(audioContext.sampleRate * (AUDIO_CHUNK_TARGET_MS / 1000)));
      const workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor", {
        processorOptions: { targetFrameCount: frameThreshold },
      });
      workletNodeRef.current = workletNode;

      const muteNode = audioContext.createGain();
      muteNode.gain.value = 0;
      muteNodeRef.current = muteNode;

      workletNode.port.onmessage = (event) => {
        const socket = audioSocketRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN) {
          return;
        }
        socket.send(event.data.samples as ArrayBuffer);
      };

      sourceNode.connect(workletNode);
      workletNode.connect(muteNode);
      muteNode.connect(audioContext.destination);

      await connectAudioSocket(activeSessionId);
    } catch (error) {
      await teardownCapture(error instanceof Error ? error.message : "Capture failed to start.");
    }
  }

  async function handleStopCapture(reason = "Capture stopped manually."): Promise<void> {
    await teardownCapture(reason);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="panel-header">
          <h1>Stream Copier</h1>
          <button disabled={isCreatingSession} onClick={() => void handleCreateSession()}>
            {isCreatingSession ? "Creating..." : "New Session"}
          </button>
        </div>
        <div className="session-list">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={session.id === activeSessionId ? "session-row active" : "session-row"}
            >
              <button className={session.id === activeSessionId ? "session-card active" : "session-card"} onClick={() => void handleSelectSession(session.id)}>
                <span>{session.config.source_name}</span>
                <span>{session.market.symbol}</span>
              </button>
              <button
                className="session-delete"
                disabled={deletingSessionId === session.id}
                onClick={() => void handleDeleteSession(session.id)}
                title="Delete session"
              >
                {deletingSessionId === session.id ? "..." : "Delete"}
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="content">
        <section className="hero">
          <div>
            <p className="eyebrow">Operator Console</p>
            <h2>Live audio, transcript events, and NinjaTrader execution.</h2>
          </div>
          <div className="capture-controls">
            <button disabled={isCreatingSession} onClick={() => void handleCreateSession()}>
              {isCreatingSession ? "Creating..." : "New Session"}
            </button>
            <button
              disabled={!activeSessionId || captureStatus === "capturing" || captureStatus === "connecting" || captureStatus === "reconnecting"}
              onClick={() => void handleStartCapture()}
            >
              Share YouTube Tab
            </button>
            <button
              disabled={captureStatus !== "capturing" && captureStatus !== "connecting" && captureStatus !== "reconnecting"}
              onClick={() => void handleStopCapture()}
            >
              Stop Capture
            </button>
            <span className={`status-pill status-${captureStatus}`}>{captureStatus}</span>
          </div>
          <p className="capture-status-detail muted">{captureStatusDetail}</p>
        </section>

        {activeSession ? (
          <div className="grid">
            <section className="panel">
              <div className="panel-header">
                <h3>Session State</h3>
                <span>{activeSession.id.slice(0, 8)}</span>
              </div>
              <div className="stats">
                <div>
                  <label>Market</label>
                  <strong>
                    {formatMaybePrice(
                      hasLiveBrokerState
                        ? brokerState?.last_price
                        : (activeSession.market.last_price ?? undefined),
                    )}
                  </strong>
                </div>
                <div>
                  <label>Position</label>
                  <strong>
                    {hasLiveBrokerState
                      ? (brokerState?.has_position
                        ? `${brokerState.market_position ?? "UNKNOWN"} ${brokerState.quantity ?? 0}`
                        : "Flat")
                      : (activeSession.position ? `${activeSession.position.side} ${activeSession.position.quantity}` : "Flat")}
                  </strong>
                </div>
                <div>
                  <label>Last Intent</label>
                  <strong>{activeSession.last_intent?.tag ?? "None"}</strong>
                </div>
                <div>
                  <label>Realized PnL</label>
                  <strong>
                    {formatMaybeCurrency(
                      hasLiveBrokerState
                        ? brokerState?.account_realized_pnl
                        : activeSession.realized_pnl,
                      hasLiveBrokerState ? brokerState?.account_currency : "USD",
                    )}
                  </strong>
                </div>
              </div>

              <div className="broker-card">
                <div className="panel-header">
                  <h3>Broker Telemetry</h3>
                  <span>{brokerState?.timestamp_utc ? new Date(brokerState.timestamp_utc).toLocaleTimeString() : "Waiting"}</span>
                </div>
                <div className="broker-controls">
                  <input
                    value={brokerAccount}
                    onChange={(event) => setBrokerAccount(event.target.value)}
                    placeholder="Account override (optional)"
                  />
                  <input
                    value={brokerSymbol}
                    onChange={(event) => setBrokerSymbol(event.target.value)}
                    placeholder="Instrument override (e.g. MNQ 03-26)"
                  />
                </div>
                {brokerStateError ? <p className="muted broker-error">{brokerStateError}</p> : null}
                {brokerState ? (
                  <div className="broker-stats">
                    <div>
                      <label>Account / Symbol</label>
                      <strong>{`${brokerState.account ?? "-"} / ${brokerState.symbol ?? "-"}`}</strong>
                    </div>
                    <div>
                      <label>Live Last</label>
                      <strong>{formatMaybePrice(brokerState.last_price)}</strong>
                    </div>
                    <div>
                      <label>Bid / Ask</label>
                      <strong>{`${formatMaybePrice(brokerState.bid_price)} / ${formatMaybePrice(brokerState.ask_price)}`}</strong>
                    </div>
                    <div>
                      <label>Position</label>
                      <strong>
                        {brokerState.market_position
                          ? `${brokerState.market_position} ${brokerState.quantity ?? 0}`
                          : "Unknown"}
                      </strong>
                    </div>
                    <div>
                      <label>Avg Entry</label>
                      <strong>{formatMaybePrice(brokerState.average_price)}</strong>
                    </div>
                    <div>
                      <label>Total PnL</label>
                      <strong>{formatMaybeCurrency(brokerState.account_total_pnl, brokerState.account_currency)}</strong>
                    </div>
                  </div>
                ) : (
                  <p className="muted">No broker state yet.</p>
                )}
              </div>

              <div className="manual-trade-controls">
                <label className="field">
                  <span>Contract Size</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    step={1}
                    value={manualContractSize}
                    onChange={(event) => {
                      const parsed = Number.parseInt(event.target.value, 10);
                      if (Number.isNaN(parsed)) {
                        setManualContractSize(1);
                        return;
                      }
                      setManualContractSize(Math.max(1, Math.min(10, parsed)));
                    }}
                  />
                </label>
                <div className="manual-trade-buttons">
                  <button
                    disabled={!activeSessionId || isSubmittingManualTrade}
                    onClick={() => void handleManualTrade("BUY")}
                  >
                    Buy
                  </button>
                  <button
                    disabled={!activeSessionId || isSubmittingManualTrade}
                    onClick={() => void handleManualTrade("SELL")}
                  >
                    Sell
                  </button>
                  <button
                    disabled={!activeSessionId || isSubmittingManualTrade}
                    onClick={() => void handleManualTrade("CLOSE")}
                  >
                    Close
                  </button>
                </div>
                {manualTradeError ? <p className="muted broker-error">{manualTradeError}</p> : null}
              </div>

              <p className="muted">Market snapshot is sourced from NinjaTrader live telemetry.</p>

              <div className="config-grid">
                <label className="field">
                  <span>ASR Model</span>
                  <select
                    value={activeSession.config.transcription_model}
                    disabled={updatingConfigKey === "model"}
                    onChange={(event) =>
                      void handleUpdateSessionConfig({ transcription_model: event.target.value }, "model")
                    }
                  >
                    {TRANSCRIPTION_MODELS.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="toggle-field">
                  <span>Partial Intent Detection</span>
                  <input
                    type="checkbox"
                    checked={activeSession.config.enable_partial_intent_detection}
                    disabled={updatingConfigKey === "partial-intent"}
                    onChange={(event) =>
                      void handleUpdateSessionConfig(
                        { enable_partial_intent_detection: event.target.checked },
                        "partial-intent",
                      )
                    }
                  />
                </label>
                <label className="toggle-field">
                  <span>Gemini Entry Confirm</span>
                  <input
                    type="checkbox"
                    checked={activeSession.config.enable_ai_fallback}
                    disabled={updatingConfigKey === "gemini-confirm"}
                    onChange={(event) =>
                      void handleUpdateSessionConfig(
                        { enable_ai_fallback: event.target.checked },
                        "gemini-confirm",
                      )
                    }
                  />
                </label>
                <label className="toggle-field">
                  <span>Early Preview Entry</span>
                  <input
                    type="checkbox"
                    checked={activeSession.config.enable_early_preview_entries}
                    disabled={updatingConfigKey === "early-preview"}
                    onChange={(event) =>
                      void handleUpdateSessionConfig(
                        { enable_early_preview_entries: event.target.checked },
                        "early-preview",
                      )
                    }
                  />
                </label>
              </div>

              <textarea
                value={manualText}
                onChange={(event) => setManualText(event.target.value)}
                rows={5}
                placeholder="Paste or type transcript text here"
              />
              <button
                disabled={!activeSessionId || isInjectingTranscript || !manualText.trim()}
                onClick={() => void handleSendManual()}
              >
                {isInjectingTranscript ? "Injecting..." : "Inject Transcript"}
              </button>
              {injectTranscriptError ? <p className="muted broker-error">{injectTranscriptError}</p> : null}
            </section>

            <section className="panel">
              <div className="panel-header">
                <h3>Open Position</h3>
              </div>
              {hasLiveBrokerState ? (
                brokerState?.has_position ? (
                  <div className="position-card">
                    <p>{brokerState.market_position ?? "UNKNOWN"}</p>
                    <p>{brokerState.quantity ?? 0} contract(s)</p>
                    <p>Avg {formatMaybePrice(brokerState.average_price)}</p>
                    <p>Stop {formatMaybePrice(brokerState.stop_price)}</p>
                    <p>Target {formatMaybePrice(brokerState.target_price)}</p>
                    <p>Unrealized {formatMaybeCurrency(brokerState.position_unrealized_pnl, brokerState.account_currency)}</p>
                  </div>
                ) : (
                  <p className="muted">No open position.</p>
                )
              ) : activeSession.position ? (
                <div className="position-card">
                  <p>{activeSession.position.side}</p>
                  <p>{activeSession.position.quantity} contract(s)</p>
                  <p>Avg {activeSession.position.average_price.toFixed(2)}</p>
                  <p>Stop {activeSession.position.stop_price?.toFixed(2) ?? "None"}</p>
                  <p>Target {activeSession.position.target_price?.toFixed(2) ?? "None"}</p>
                  <p>Realized ${activeSession.position.realized_pnl.toFixed(2)}</p>
                </div>
              ) : (
                <p className="muted">No open position.</p>
              )}

              <div className="panel-header">
                <h3>Recent Transcript</h3>
                <span>{captureStatus === "capturing" || captureStatus === "reconnecting" ? "Live" : "Idle"}</span>
              </div>
              <div className="latency-card">
                <div className="latency-header">
                  <strong>STT Latency</strong>
                  <span>{latencyState.label}</span>
                </div>
                {latencyState.metrics ? (
                  <div className="latency-grid">
                    <div>
                      <label>Total</label>
                      <strong>{formatLatency(latencyState.metrics.total_latency_ms)}</strong>
                    </div>
                    <div>
                      <label>Speech Window</label>
                      <strong>{formatLatency(latencyState.metrics.speech_capture_ms)}</strong>
                    </div>
                    <div>
                      <label>Processing</label>
                      <strong>{formatLatency(latencyState.metrics.processing_ms)}</strong>
                    </div>
                    <div>
                      <label>Voice</label>
                      <strong>{formatLatency(latencyState.metrics.voice_duration_ms)}</strong>
                    </div>
                  </div>
                ) : (
                  <p className="muted latency-empty">No transcription timing yet.</p>
                )}
              </div>
              <div className="candidate-card">
                <div className="candidate-header">
                  <strong>Live Preview Candidate</strong>
                  <span>{activeSession.config.enable_partial_intent_detection ? "Preview Parser On" : "Off"}</span>
                </div>
                <p className="muted candidate-note">Preview candidates come from partial speech only. Orders wait for the final transcript, confirmation, and risk checks.</p>
                {activeSession.latest_candidate_intent ? (
                  <div className="candidate-body">
                    <div className="candidate-pill-row">
                      <span className="candidate-pill">{activeSession.latest_candidate_intent.tag}</span>
                      <span>{Math.round(activeSession.latest_candidate_intent.confidence * 100)}% confidence</span>
                    </div>
                    <p>{activeSession.latest_candidate_intent.evidence_text}</p>
                    <div className="candidate-metrics">
                      {formatIntentLevels(activeSession.latest_candidate_intent)}
                    </div>
                  </div>
                ) : (
                  <p className="muted candidate-empty">
                    {activeSession.config.enable_partial_intent_detection
                      ? "No preview candidate in the live speech yet."
                      : "Partial intent detection is disabled for this session."}
                  </p>
                )}
              </div>
              <div className="transcript-list">
                {captureStatus === "capturing" || captureStatus === "reconnecting" ? (
                  <div className={activeSession.latest_partial_text ? "transcript-partial transcript-partial-live" : "transcript-partial transcript-partial-idle"}>
                    <span className="transcript-label">Live Preview</span>
                    <p>{activeSession.latest_partial_text || "Listening for speech..."}</p>
                    {activeSession.latest_partial_metrics ? (
                      <div className="transcript-metrics">
                        <span>{formatLatency(activeSession.latest_partial_metrics.total_latency_ms)} total</span>
                        <span>{formatLatency(activeSession.latest_partial_metrics.processing_ms)} processing</span>
                      </div>
                    ) : null}
                  </div>
                ) : null}
                {activeSession.transcripts
                  .slice()
                  .reverse()
                  .slice(0, 10)
                  .map((segment) => (
                    <article key={segment.id} className="transcript-item">
                      <span>{segment.source}</span>
                      <p>{segment.text}</p>
                      {segment.metrics ? (
                        <div className="transcript-metrics">
                          <span>{formatLatency(segment.metrics.total_latency_ms)} total</span>
                          <span>{formatLatency(segment.metrics.processing_ms)} processing</span>
                        </div>
                      ) : null}
                    </article>
                  ))}
              </div>
            </section>

            <section className="panel panel-wide">
              <div className="panel-header">
                <h3>Event Timeline</h3>
              </div>
              <div className="event-list">
                {activeSession.events
                  .slice()
                  .reverse()
                  .slice(0, 50)
                  .map((event) => (
                    <article key={event.id} className={`event-row event-${event.type.toLowerCase()}`}>
                      <div>
                        <strong>{event.title}</strong>
                        <p>{event.message}</p>
                      </div>
                      <time>{new Date(event.created_at).toLocaleTimeString()}</time>
                    </article>
                  ))}
              </div>
            </section>
          </div>
        ) : (
          <section className="panel empty-state">
            <p>Create a session to begin ingesting transcript or audio.</p>
          </section>
        )}
      </main>
    </div>
  );
}

function getLatencyState(session: StreamSession | null): { label: string; metrics: TranscriptionMetrics | null } {
  if (!session) {
    return { label: "No session", metrics: null };
  }
  if (session.latest_partial_metrics) {
    return { label: "Live Preview", metrics: session.latest_partial_metrics };
  }
  if (session.latest_final_metrics) {
    return { label: "Last Final", metrics: session.latest_final_metrics };
  }
  return { label: "Waiting", metrics: null };
}

function upsertSession(
  sessions: StreamSession[],
  session: StreamSession,
  options?: { prepend?: boolean },
): StreamSession[] {
  const existingIndex = sessions.findIndex((item) => item.id === session.id);
  if (existingIndex === -1) {
    return options?.prepend ? [session, ...sessions] : [...sessions, session];
  }

  const next = sessions.slice();
  next[existingIndex] = session;
  if (options?.prepend && existingIndex > 0) {
    next.splice(existingIndex, 1);
    next.unshift(session);
  }
  return next;
}

function applyPatch(
  session: StreamSession,
  event: TimelineEvent,
  patch: SessionPatch,
  appendEvent: boolean,
): StreamSession {
  const next = { ...session };

  if (appendEvent) {
    const events = [...session.events, event];
    next.events = events.length > 500 ? events.slice(-500) : events;
  }

  if ("new_transcript" in patch && patch.new_transcript !== undefined) {
    const transcripts = [...session.transcripts, patch.new_transcript];
    next.transcripts = transcripts.length > 250 ? transcripts.slice(-250) : transcripts;
  }

  if ("latest_partial_text" in patch) {
    next.latest_partial_text = patch.latest_partial_text ?? "";
  }
  if ("latest_partial_metrics" in patch) {
    next.latest_partial_metrics = patch.latest_partial_metrics ?? null;
  }
  if ("latest_final_metrics" in patch) {
    next.latest_final_metrics = patch.latest_final_metrics ?? null;
  }
  if ("latest_candidate_intent" in patch) {
    next.latest_candidate_intent = patch.latest_candidate_intent ?? null;
  }
  if ("last_intent" in patch) {
    next.last_intent = patch.last_intent ?? null;
  }
  if ("market" in patch) {
    next.market = patch.market ?? session.market;
  }
  if ("position" in patch) {
    next.position = patch.position ?? null;
  }
  if ("realized_pnl" in patch && patch.realized_pnl !== undefined) {
    next.realized_pnl = patch.realized_pnl;
  }

  return next;
}

function formatLatency(valueMs: number): string {
  if (valueMs >= 1000) {
    return `${(valueMs / 1000).toFixed(2)}s`;
  }
  return `${valueMs} ms`;
}

function formatIntentLevels(intent: TradeIntent): string {
  const levels = [
    intent.entry_price !== null ? `Entry ${intent.entry_price.toFixed(2)}` : null,
    intent.stop_price !== null ? `Stop ${intent.stop_price.toFixed(2)}` : null,
    intent.target_price !== null ? `Target ${intent.target_price.toFixed(2)}` : null,
    intent.side ? intent.side : null,
  ].filter(Boolean);
  return levels.join("  •  ") || "No price levels extracted yet.";
}

function formatMaybePrice(value: number | undefined): string {
  if (value === undefined) {
    return "-";
  }
  return value.toFixed(2);
}

function formatMaybeCurrency(value: number | undefined, currency: string | undefined): string {
  if (value === undefined) {
    return "-";
  }
  const prefix = currency === "UsDollar" || currency === "USD" || !currency ? "$" : "";
  return `${prefix}${value.toFixed(2)}`;
}

export default App;
