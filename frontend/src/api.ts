import type { BrokerState, StreamSession } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:2712/api";
const WS_BASE = (import.meta.env.VITE_WS_BASE_URL as string | undefined) ?? "ws://localhost:2712/ws";
const DEFAULT_STREAM_SYMBOL = (import.meta.env.VITE_DEFAULT_SYMBOL as string | undefined)?.trim() || "MNQ 03-26";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string; message?: string };
      if (payload.detail) {
        detail = payload.detail;
      } else if (payload.message) {
        detail = payload.message;
      }
    } catch {
      const textBody = await response.text();
      if (textBody.trim()) {
        detail = textBody.trim();
      }
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export async function listSessions(): Promise<StreamSession[]> {
  const response = await fetch(`${API_BASE}/sessions`);
  return parseResponse<StreamSession[]>(response);
}

export async function createSession(): Promise<StreamSession> {
  const response = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config: {
        source_name: "Flow Zone Trader",
        symbol: DEFAULT_STREAM_SYMBOL,
        execution_mode: "auto",
        enable_audio_capture: true,
        enable_ai_fallback: true,
        enable_partial_intent_detection: true,
        enable_early_preview_entries: false,
        auto_execute: true,
        default_contract_size: 3,
      },
    }),
  });
  return parseResponse<StreamSession>(response);
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: "DELETE",
  });
  await parseResponse<void>(response);
}

export async function updateSessionConfig(
  sessionId: string,
  patch: {
    enable_partial_intent_detection?: boolean;
    enable_ai_fallback?: boolean;
    enable_early_preview_entries?: boolean;
    transcription_model?: string;
    broker_account_override?: string | null;
    broker_symbol_override?: string | null;
  },
): Promise<StreamSession> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/config`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return parseResponse<StreamSession>(response);
}

export async function sendManualSegment(sessionId: string, text: string): Promise<StreamSession> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/segments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      status: "final",
      source: "manual_ui",
      confidence: 1.0,
    }),
  });
  return parseResponse<StreamSession>(response);
}

export async function sendManualTrade(
  sessionId: string,
  action: "BUY" | "SELL" | "CLOSE",
  contractSize: number,
  options?: { account?: string; symbol?: string },
): Promise<StreamSession> {
  const account = options?.account?.trim();
  const symbol = options?.symbol?.trim();
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/manual-trade`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      contract_size: contractSize,
      account: account || null,
      symbol: symbol || null,
    }),
  });
  return parseResponse<StreamSession>(response);
}

export function eventSocketUrl(sessionId: string): string {
  return `${WS_BASE}/sessions/${sessionId}/events`;
}

export function audioSocketUrl(sessionId: string): string {
  return `${WS_BASE}/sessions/${sessionId}/audio`;
}

export async function getBrokerState(
  sessionId: string,
  options?: { account?: string; symbol?: string },
): Promise<BrokerState> {
  const params = new URLSearchParams();
  if (options?.account?.trim()) {
    params.set("account", options.account.trim());
  }
  if (options?.symbol?.trim()) {
    params.set("symbol", options.symbol.trim());
  }

  const suffix = params.toString();
  const response = await fetch(
    `${API_BASE}/sessions/${sessionId}/broker-state${suffix ? `?${suffix}` : ""}`,
  );
  return parseResponse<BrokerState>(response);
}
