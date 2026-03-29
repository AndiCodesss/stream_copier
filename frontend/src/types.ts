export type ActionTag =
  | "NO_ACTION"
  | "SETUP_LONG"
  | "SETUP_SHORT"
  | "ENTER_LONG"
  | "ENTER_SHORT"
  | "ADD"
  | "TRIM"
  | "EXIT_ALL"
  | "MOVE_STOP"
  | "MOVE_TO_BREAKEVEN"
  | "TARGET"
  | "CANCEL_SETUP"
  | "COMMENTARY";

export interface PositionState {
  side: "LONG" | "SHORT";
  quantity: number;
  average_price: number;
  stop_price: number | null;
  target_price: number | null;
  opened_at: string;
  realized_pnl: number;
}

export interface TranscriptSegment {
  id: string;
  session_id: string;
  text: string;
  status: "partial" | "final";
  source: string;
  confidence: number;
  metrics: TranscriptionMetrics | null;
  received_at: string;
}

export interface TranscriptionMetrics {
  total_latency_ms: number;
  speech_capture_ms: number;
  processing_ms: number;
  audio_duration_ms: number;
  voice_duration_ms: number;
}

export interface TimelineEvent {
  id: string;
  session_id: string;
  type: "INFO" | "WARNING" | "TRANSCRIPT" | "INTENT" | "RISK" | "EXECUTION" | "MARKET" | "SYSTEM";
  title: string;
  message: string;
  data: Record<string, unknown>;
  created_at: string;
}

export interface TradeIntent {
  id: string;
  session_id: string;
  tag: ActionTag;
  symbol: string;
  side: "LONG" | "SHORT" | null;
  entry_price: number | null;
  stop_price: number | null;
  target_price: number | null;
  quantity_hint: string | null;
  confidence: number;
  evidence_text: string;
  source_segment_id: string | null;
  source_received_at: string | null;
  source_latency_ms: number;
  guard_reason: string | null;
  stale_after_ms: number;
  created_at: string;
}

export interface MarketSnapshot {
  symbol: string;
  last_price: number | null;
  bid_price: number | null;
  ask_price: number | null;
  received_at: string | null;
}

export interface StreamSession {
  id: string;
  created_at: string;
  config: {
    source_name: string;
    symbol: string;
    execution_mode: "auto" | "review";
    enable_audio_capture: boolean;
    enable_ai_fallback: boolean;
    enable_partial_intent_detection: boolean;
    enable_early_preview_entries: boolean;
    auto_execute: boolean;
    default_contract_size: number;
    transcription_model: string;
    broker_account_override: string | null;
    broker_symbol_override: string | null;
  };
  market: MarketSnapshot;
  position: PositionState | null;
  realized_pnl: number;
  transcripts: TranscriptSegment[];
  events: TimelineEvent[];
  latest_candidate_intent: TradeIntent | null;
  last_intent: TradeIntent | null;
  latest_partial_text: string;
  latest_partial_metrics: TranscriptionMetrics | null;
  latest_final_metrics: TranscriptionMetrics | null;
}

export interface BrokerState {
  ok: boolean;
  code: string;
  message: string;
  timestamp_utc?: string;
  account?: string;
  account_currency?: string;
  symbol?: string;
  market_position?: "LONG" | "SHORT" | "FLAT";
  quantity?: number;
  average_price?: number;
  stop_price?: number;
  target_price?: number;
  last_price?: number;
  bid_price?: number;
  ask_price?: number;
  position_unrealized_pnl?: number;
  account_realized_pnl?: number;
  account_unrealized_pnl?: number;
  account_total_pnl?: number;
  has_position?: boolean;
}

/**
 * Fields that changed in a single event. A field being absent means "no change".
 * A field present as null means "set this field to null".
 * Matches backend SessionPatch serialized with exclude_unset=True.
 */
export interface SessionPatch {
  latest_partial_text?: string | null;
  latest_partial_metrics?: TranscriptionMetrics | null;
  latest_final_metrics?: TranscriptionMetrics | null;
  latest_candidate_intent?: TradeIntent | null;
  last_intent?: TradeIntent | null;
  market?: MarketSnapshot | null;
  position?: PositionState | null;
  realized_pnl?: number;
  new_transcript?: TranscriptSegment;
}

/** Messages received from the /ws/sessions/{id}/events WebSocket. */
export type WsMessage =
  | { type: "snapshot"; session: StreamSession }
  | { type: "event"; event: TimelineEvent; patch: SessionPatch; append_event: boolean };
