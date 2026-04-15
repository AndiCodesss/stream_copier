# How the Code Works

A detailed walkthrough of every part of Stream Copier, from audio capture to trade execution.

---

## The Big Picture

Stream Copier is a pipeline. Audio goes in one end, trades come out the other. Each stage transforms the data a little further:

```
Audio from browser
  → Resample
  → Detect speech
  → Transcribe with Whisper
  → Fix transcription errors
  → Detect trading intent
  → Check risk rules
  → Send order to NinjaTrader
  → Show everything on the dashboard
```

Every stage is its own module. They're loosely connected through the session manager, which acts as the central coordinator.

---

## Stage 1: Audio Capture

**File:** `frontend/src/App.tsx`

When you click "Share YouTube Tab" in the dashboard, the browser uses the Web Audio API to capture whatever audio is playing in that tab. The app creates an AudioWorklet (a low-level audio processor that runs on a separate thread) to grab chunks of raw PCM audio at 48 kHz.

These chunks get sent over a WebSocket connection to the backend every ~30ms. The WebSocket stays open for the entire session. If the connection drops, the frontend automatically reconnects with linear backoff (750ms per attempt, capped at 3 seconds).

The frontend also handles:
- Creating, listing, and deleting sessions
- Polling the broker for position and market data (every 500ms while capturing, 2s when idle)
- Displaying the live transcript, detected intents, risk decisions, and executions
- Manual trade buttons (BUY/SELL/CLOSE) and manual transcript injection

**File:** `frontend/src/api.ts`

A thin HTTP/WebSocket client. Functions like `createSession()`, `listSessions()`, `sendManualTrade()`, and URL builders for the two WebSocket endpoints (audio and events).

---

## Stage 2: Audio Resampling

**File:** `backend/app/core/audio.py`

Whisper needs 16 kHz audio, but the browser sends 48 kHz. This file has one function: `resample_pcm16_mono()`.

How it works:
1. Convert the raw bytes into a NumPy array of 16-bit integers
2. Cast to float32 for arithmetic (values stay in the PCM-16 range, not normalized to [-1, 1])
3. Calculate where each target sample falls in the source (e.g., target sample 0 maps to source sample 0, target sample 1 maps to source sample 3, etc.)
4. Use linear interpolation between the two nearest source samples to compute each target sample
5. Clip to [-32768, 32767] and convert back to 16-bit integers and return as bytes

If the source and target rates are the same, it just returns the original data untouched.

---

## Stage 3: Voice Activity Detection

**File:** `backend/app/services/transcription/segmenter.py`

Raw audio is continuous — there's music, silence, crosstalk, and actual trading calls all mixed together. The segmenter's job is to figure out when someone is actually talking and extract just those parts.

There are two backends:

**WebRTC VAD** (default): Google's voice activity detection library. It processes audio in small frames (20-30ms each) and classifies each frame as "speech" or "silence". The segmenter maintains a sliding window and starts recording when enough consecutive frames are speech. It keeps a 120ms preroll buffer so it doesn't clip the beginning of words.

**Energy-based** (fallback): Simpler approach. Computes the RMS (root mean square) energy of each frame. If the energy is above a threshold, it's speech. Less accurate but doesn't need the WebRTC library.

A segment is finalized when:
- Silence has lasted long enough (`silence_duration_ms`, default 300ms), OR
- The segment hit the maximum length (`max_duration_ms`)

Too-short segments (below `min_duration_ms`) get discarded — they're usually noise.

The output is a `ReadyAudioSegment`: the raw audio bytes plus metadata like total duration, voice duration, and timestamps.

---

## Stage 4: Whisper Transcription

**File:** `backend/app/services/transcription/local_whisper.py`

This is the core transcription engine. It uses faster-whisper (a CTranslate2 optimized version of OpenAI's Whisper) with a dual-model approach:

**Preview model** (`distil-small.en`, beam size 1): Fast but rough. Runs periodically on whatever audio has accumulated so far, giving you something to show on the dashboard within a second or two. Think of it as a "draft" transcription.

**Final model** (`distil-large-v3`, beam size 2): Slower but much more accurate. Runs once the segmenter says a speech segment is complete. Replaces the preview with the "real" transcription.

The transcription process:
1. Convert PCM16 bytes to float32 NumPy array, normalized to [-1, 1]
2. Feed to faster-whisper with configured beam size, temperature, and repetition penalty
3. Collect all decoded segments and their per-token log probabilities
4. Compute overall confidence from the log probs
5. Check for hallucinations — Whisper sometimes generates repetitive or made-up text when the audio is noisy. The code detects this by checking:
   - Compression ratio (highly repetitive text compresses well)
   - Average log probability (low confidence = likely hallucination)
6. If a hallucination is detected, retry with different parameters

**File:** `backend/app/services/transcription/streaming_preview.py`

Manages the assembly of preview transcripts. Since previews run on incomplete audio, each new preview might partially overlap with the previous one. The assembler tracks committed (stable) words and new (unstable) words, finding overlaps between consecutive previews and only committing words that have been stable across multiple runs.

---

## Stage 5: Text Normalization

**File:** `backend/app/services/interpretation/transcript_normalizer.py`

Whisper makes predictable mistakes with trading jargon. This module lowercases the text, normalizes curly apostrophes, strips commas from numbers, then runs 14 regex replacements:

| Whisper says | Corrected to |
|-------------|-------------|
| "v w a p" or "view up" or "vw up" | "vwap" |
| "break even" / "broke even" | "breakeven" |
| "got my ad on" | "got my add on" |
| "m n q" | "mnq" |
| "n q" | "nq" |
| "m e s" | "mes" |
| "e s" | "es" |
| "paying myself peace" | "paying myself piece" |
| "peace on/here/there/at/versus" | "piece on/here/..." |

The "peace" to "piece" correction only fires in specific trading contexts (after "paying myself", "paid myself", or before words like "on", "here", "versus") — it won't touch unrelated uses. All outputs are lowercase. There are 17 total correction operations (3 inline + 14 regex). They're intentionally conservative — only patterns that are unambiguous in a trading context.

---

## Stage 6: Intent Detection

This is the most complex part of the system. It takes clean transcript text and decides: is this a trading action, and if so, what kind?

### Layer 1: Pattern Matching

**File:** `backend/app/services/interpretation/action_language.py`

The foundation. Contains 195 regex patterns organized by action type:

**Entry patterns** (23 long + 25 short + 32 side-neutral = 80 total): Match phrases like "I'm long", "going short", "buying here", "got my entry". Side-neutral patterns ("small piece on here", "I'm in this") use position context to determine the side. Each returns a `PhraseSignal` with the action tag and side.

**Exit patterns** (31): "I'm out", "flatten", "close it out", "stopped out", "knocked out", "done with this".

**Trim patterns** (23): "paying myself", "taking profit", "taking some off", "covering a piece", "peeling off".

**Stop patterns** (9): "move stop to 600", "stop at breakeven", "trailing this".

**Breakeven patterns** (7): "stop now breakeven", "move stop to breakeven", "stops into the money".

**Setup patterns** (6 long + 11 short = 17): "looking for a long", "watching for short setup", "interested in shorts". These are lower confidence — the trader is thinking about a trade but hasn't committed.

There is no dedicated "add" list — adding is determined contextually: if a side-neutral entry pattern matches while the trader already has a position on that side, it becomes an ADD.

The module also has filters to reject false positives:
- `is_historical_trade_context()` (12 patterns): Catches past-tense language ("yesterday I went long", "earlier I was short")
- `is_hypothetical_trade_context()` (16 patterns): Catches conditionals ("if it breaks out", "I would go long")

### Layer 2: Cross-Segment Stitching

**Inside:** `backend/app/services/interpretation/rule_engine.py`

Whisper sometimes splits a single phrase across two segments. The trader says "I'm long at 600" but Whisper produces:
- Segment 1: "I'm"
- Segment 2: "long at 600"

Neither segment alone triggers a pattern match. The rule engine keeps a buffer of recent transcript fragments with timestamps. When a new segment arrives, it looks back at the last few fragments and tries concatenating them. If the combined text matches a pattern, it fires.

### Layer 3: Local Classifier (Optional)

**File:** `backend/app/services/interpretation/local_classifier.py`

A ModernBERT transformer model with a trained classification head. When enabled, it acts as a confidence gate on the rule engine's detections.

How it works:
1. Takes the transcript text plus context (symbol, current position, market price, recent history) and formats it as a multi-line string
2. Tokenizes with the model's tokenizer (max 256 tokens)
3. Runs through the frozen ModernBERT encoder to get embeddings
4. Mean-pools the embeddings across tokens
5. Passes through a small trained head (LayerNorm → Linear) to get logits
6. Softmax to get probabilities for each of 7 labels: NO_ACTION, ENTER_LONG, ENTER_SHORT, TRIM, EXIT_ALL, MOVE_STOP, MOVE_TO_BREAKEVEN
7. Compares probabilities against per-label thresholds (calibrated during training)

If the classifier says the probability of the detected action is below its threshold, the rule engine's detection gets blocked as a likely false positive.

**File:** `backend/app/services/interpretation/intent_context.py`

Formats the multi-line context string that the classifier expects. Includes symbol, position state, recent text, and current text, each clipped to 48 words max to avoid token overflow.

### Layer 4: Gemini Fallback (Optional)

**File:** `backend/app/services/interpretation/gemini_fallback.py`

For cases where the rule engine and classifier disagree, or confidence is borderline, the system can send the text to Google's Gemini 2.0 Flash API.

Two modes:
- **Interpret**: Send a transcript segment and get back a complete TradeIntent (action, side, prices, confidence)
- **Confirm**: Send a proposed intent and ask "is this a real live trade action?" Gets back confirmed/rejected with a reason

The prompt engineering is important here. It explicitly tells Gemini to reject:
- Historical mentions ("yesterday I went long")
- Teaching/educational context ("you would want to go long when...")
- Hypothetical scenarios ("if price breaks this level")
- Second-person language ("you should go short")

### The Orchestrator

**File:** `backend/app/services/interpretation/rule_engine.py`

The `RuleBasedTradeInterpreter` class (2,600+ lines) ties all four layers together. It also manages:

**Candidate windows**: When a potential entry signal is detected but not yet confirmed, it opens a time-limited window (configurable, default ~6 seconds). During this window, it accumulates more transcript fragments for additional context. If the signal strengthens (more confirming text arrives), it fires. If the window expires without confirmation, the candidate is dropped.

**Flow state**: Tracks per-session state including recent fragments, active candidate windows, and confirmation status. This is how it handles cross-segment stitching and multi-sentence context.

**Price extraction**: When an entry is detected, the system tries to extract numeric values from the text for stop and target prices. It understands different formats: "stop at 600", "target twenty four thousand six hundred", and handles quarter-point notation common in futures.

**Probability assessment**: Each candidate gets a probability score from multiple sources (rule match strength, classifier probability, seed phrase detection). The highest-scoring source wins.

### Supporting Modules

**File:** `backend/app/services/interpretation/candidate_detector.py`

Assigns a probability to each potential trade candidate by combining signals from the classifier, explicit pattern matches, setup patterns, and seed phrases. Returns the strongest signal.

**File:** `backend/app/services/interpretation/embedding_gate.py`

An optional pre-filter. Uses a small embedding model (BAAI/bge-small-en-v1.5) to compute semantic similarity between the transcript and 36 canonical trading phrases. If the similarity is below a threshold (default 0.40), the segment is considered not trade-relevant and skipped entirely. This saves processing time on clearly irrelevant speech (market commentary, jokes, etc.).

---

## Stage 7: Risk Engine

**File:** `backend/app/services/execution/risk.py`

Every detected intent passes through the risk engine before any order is placed. It's a series of checks:

**For all intents:**
- Is it too old? (signal age vs `stale_intent_ms`)
- Is the confidence above minimum? (`min_confidence`, default 0.74)

**For entries specifically** (checked in this order):
- Is the effective signal age (intent age + source latency) within `max_entry_signal_age_ms`?
- Are there any guard reasons flagged by the rule engine? (context-based blockers like recent exit/management cues)
- Is there a market price available?
- Is there already a position? (no double entries unless it's an ADD)
- Does the entry have a stop price? (no-stop entries get blocked)
- Is the entry price within `max_entry_distance_points` of the current market? (chasing protection)
- Would the position size exceed `max_contract_size`?

**For management actions** (trims, stops, exits):
- Is there actually a position to manage?

The output is a `RiskDecision`: approved or rejected, with a human-readable reason.

---

## Stage 8: Order Execution

**File:** `backend/app/services/execution/ninjatrader.py`

If the risk engine approves, this module sends the order to NinjaTrader.

The `NinjaTraderExecutor` builds a JSON payload with 15 fields:
```json
{
  "intent_id": "abc-123",
  "session_id": "sess-456",
  "account": "Sim101",
  "symbol": "MNQ 03-26",
  "action": "ENTER_LONG",
  "side": "LONG",
  "quantity_hint": 2,
  "default_contract_size": 3,
  "time_in_force": "Day",
  "entry_price": 600.0,
  "stop_price": 580.0,
  "target_price": 620.0,
  "market_price": 600.0,
  "evidence_text": "I'm long versus 580",
  "sent_at": "2025-01-15T14:30:00Z"
}
```

If `force_wide_brackets` is enabled, it overrides the stop and target with fixed distances from the entry price (configurable, default 120 points stop and 240 points target).

The payload is POSTed to `http://127.0.0.1:18080/api/stream-copier/commands`. If that fails (common in WSL environments where the broker is on Windows), it tries alternative IPs.

**File:** `bridges/StreamCopierBridgeAddOn.cs`

The other side of the bridge. A C# AddOn running inside NinjaTrader 8 that:

1. Listens on port 18080 for HTTP requests
2. Validates the bearer token
3. Translates commands into NinjaTrader API calls:
   - `ENTER_LONG`/`ENTER_SHORT`: Submit entry order, then attach stop/target protection orders after the fill
   - `ADD`: Add to existing position
   - `TRIM`: Partial exit
   - `EXIT_ALL`: Flatten everything
   - `MOVE_STOP`: Modify the stop order price
4. Exposes a `GET /state` endpoint returning current position, entry price, PnL, and market snapshot

The bridge handles threading carefully — NinjaTrader requires all order operations to happen on its dispatcher thread, so the HTTP handler marshals commands across threads with timeouts.

---

## The Glue: Session Manager

**File:** `backend/app/services/session_manager.py`

The orchestrator that ties everything together. When audio arrives:

1. Resamples 48 kHz → 16 kHz
2. Pushes to the transcriber
3. Transcriber calls back with transcript segments
4. Segments go through normalization → intent detection → risk → execution
5. Every event gets published to the EventHub
6. Session state gets saved (debounced, 1 second)

It also handles:
- Lazy-loading the transcriber (models download on first use)
- Caching broker state (0.5s TTL to avoid hammering NinjaTrader)
- Manual trades (bypass the whole pipeline, go straight to execution)
- Preview entry confirmation (short window to confirm or reject borderline entries)

---

## The Event System

**File:** `backend/app/services/event_hub.py`

A simple in-process pub/sub. When anything happens (new transcript, intent detected, trade executed), the session manager publishes a message. The WebSocket handler subscribes when a client connects and forwards events to the browser.

Uses asyncio queues — one per connected client. Thread-safe with an asyncio lock.

---

## Persistence

**File:** `backend/app/services/storage/session_store.py`

Saves full session snapshots as JSON files, one per session (`{session_id}.json`). On startup, loads all existing sessions back into memory.

**File:** `backend/app/services/storage/event_store.py`

Append-only event log. Each event is one JSON line in a JSONL file (`{session_id}.jsonl`). Used for audit trails and debugging — you can replay exactly what happened during a session.

---

## Data Models

**File:** `backend/app/models/domain.py`

Pydantic models that define the shape of all data in the system:

| Model | What it represents |
|-------|-------------------|
| `ActionTag` | What kind of trade action (ENTER_LONG, TRIM, EXIT_ALL, MOVE_STOP, etc.) |
| `TradeSide` | LONG or SHORT |
| `TranscriptSegment` | A piece of transcribed text with confidence and timing |
| `TradeIntent` | A detected trading action with prices, confidence, and evidence |
| `RiskDecision` | Approved or rejected, with reason |
| `ExecutionResult` | What happened when the order was sent |
| `MarketSnapshot` | Current price, bid, ask |
| `PositionState` | Open position details (side, size, entry price, stop, target) |
| `StreamSession` | Everything about a session — config, transcripts, events, position, market |
| `SessionConfig` | User settings — symbol, execution mode, feature flags |
| `TimelineEvent` | An audit log entry (INFO, WARNING, TRANSCRIPT, INTENT, RISK, EXECUTION) |
| `SessionPatch` | Delta update for WebSocket (only send what changed) |

---

## Configuration

**File:** `backend/app/core/config.py`

A Pydantic Settings class that reads 79 environment variables from `backend/.env`. Groups:

**Transcription**: Which Whisper models to use, CPU vs CUDA, beam sizes, preview intervals, warmup settings.

**Speech/VAD**: Which voice detection backend (WebRTC or energy), aggressiveness (0-3), frame sizes, silence duration, min/max segment lengths.

**Interpretation**: Rule engine mode, classifier model name, probability thresholds (min, block, recovery), candidate window timing, Gemini API key.

**Risk**: Confidence floors, max entry distance, signal age limits, position size caps.

**Execution**: NinjaTrader bridge URL, account name, default symbol, bracket sizes, contract limits.

Has a model validator that normalizes legacy config names and clamps values to valid ranges.

---

## API Layer

**File:** `backend/app/api/routes.py`

FastAPI routes. 9 REST endpoints and 2 WebSocket endpoints.

The WebSocket event handler is careful about race conditions: it subscribes to the event hub *before* reading the current session state. This ensures no events are missed between reading the state and starting to listen.

The audio WebSocket accepts both binary frames (raw PCM) and JSON messages (for changing the sample rate mid-stream).

**File:** `backend/app/main.py`

Creates the FastAPI app, adds CORS middleware, includes the router, and attaches the WebSocket handlers. Uses a lifespan context manager for clean shutdown.

---

## Offline Tools (Training and Benchmarking)

These don't run during live trading. They're for building and evaluating the intent classifier.

### Training Pipeline

**File:** `backend/app/services/interpretation/train_local_classifier.py`

Trains the ModernBERT classification head:
1. Load labeled examples from JSONL files
2. Deduplicate by prompt + label
3. Split by transcript (so no transcript appears in both train and test)
4. Rebalance classes (limit NO_ACTION examples to a ratio of positives)
5. Encode all examples through the frozen ModernBERT encoder
6. Train the linear head with class-weighted loss and early stopping
7. Calibrate per-label probability thresholds using F1 optimization on validation set
8. Save the head weights and metadata

### Data Collection

**File:** `backend/app/services/transcription/youtube_captions.py` — Downloads YouTube captions for a channel's VODs. Not Whisper transcription — these are YouTube's own captions.

**File:** `backend/app/services/interpretation/ai_transcript_annotator.py` — Sends transcripts to Gemini and asks it to label every trading action.

**File:** `backend/app/services/interpretation/build_reviewed_ai_corpus.py` — Higher-quality labeling with automatic review passes.

**File:** `backend/app/services/interpretation/build_reviewed_execution_dataset.py` — Merges multiple reviewed corpora into a single training JSONL file.

### Benchmarking

**File:** `backend/app/services/interpretation/benchmark_models.py`

Compares five model families for trade-intent classification:

| Model | How it works |
|-------|-------------|
| TF-IDF + Logistic Regression | Bag-of-words features (20k, 1-2 grams) → linear classifier |
| TF-IDF + SVM | Same features → support vector machine with calibrated probabilities |
| TF-IDF + MLP | Same features → 2-layer neural net (256 → 128 neurons) |
| DistilBERT (frozen) | Pretrained transformer encodes text → trained linear head |
| ModernBERT (frozen) | Same approach, newer model |

Supports two evaluation modes:
- **Single split**: Fixed train/test split (5 test transcripts)
- **K-fold cross-validation**: Transcripts grouped into k folds, rotated so every transcript is tested exactly once

Outputs JSON with per-model metrics (accuracy, F1, precision, recall), confusion matrices, and per-fold statistics.
