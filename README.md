# Stream Copier

Listens to a live YouTube trading stream, understands what the trader is saying, and automatically places trades in NinjaTrader. Everything runs locally on your machine.

## What It Does

1. **Captures audio** from a browser tab via the React dashboard
2. **Transcribes speech** in real time using Whisper (two models: a fast one for previews, an accurate one for finals)
3. **Fixes transcription errors** common in trading lingo ("peace" becomes "piece", "v w a p" becomes "vwap")
4. **Detects trading intent** using 100+ pattern rules, with optional ML and AI confirmation layers
5. **Validates trades** through a risk engine (checks for stale signals, bad confidence, oversized positions, etc.)
6. **Executes orders** in NinjaTrader 8 via a local HTTP bridge

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.12+, FastAPI, Pydantic v2 |
| Speech-to-text | faster-whisper (CTranslate2), CPU or CUDA |
| Voice detection | webrtcvad |
| Intent classifier | ModernBERT via HuggingFace (optional) |
| AI fallback | Google Gemini 2.0 Flash (optional) |
| Frontend | React 18, TypeScript, Vite |
| Trade execution | C# NinjaTrader 8 AddOn |

## Getting Started

```bash
chmod +x start.sh
./start.sh
```

That's it. The script sets up a virtualenv, installs everything, detects your GPU, and starts both the backend (port 2712) and frontend (port 4300). Whisper models download automatically on first run.

### Manual Setup

```bash
# Backend
cd backend
cp .env.example .env
pip install -e .[dev]
uvicorn app.main:app --reload --host 0.0.0.0 --port 2712

# Frontend
cd frontend
npm install
npm run dev
```

### Connecting NinjaTrader

Add these to `backend/.env`:

```
NINJATRADER_BRIDGE_URL=http://127.0.0.1:18080
NINJATRADER_BRIDGE_TOKEN=<shared-secret>
NINJATRADER_ACCOUNT=Sim101
```

The C# AddOn is in `bridges/` with its own setup guide.

## How to Use

1. Open the dashboard and create a session.
2. Make sure NinjaTrader is running with the bridge AddOn loaded.
3. Click **Share YouTube Tab** and pick the trading stream.
4. Watch it work — transcripts, detected intents, risk decisions, and order executions all show up live.

To test without placing real orders, set `execution_mode=review`.

## How Intent Detection Works

The system uses four layers to figure out what the trader wants to do:

1. **Rule engine** — 100+ regex patterns catch phrases like "I'm long", "trim half", "stop at 600"
2. **Cross-segment stitching** — reassembles phrases that Whisper split across segments ("I'm" + "long")
3. **Local classifier** (optional) — a ModernBERT model adds confidence scores to filter false positives
4. **Gemini fallback** (optional) — cloud AI confirms ambiguous cases

## Project Structure

```
backend/
  app/
    api/                REST + WebSocket endpoints
    core/               Config and audio processing
    models/             Data models
    services/
      transcription/    Whisper, VAD, streaming preview
      interpretation/   Rule engine, classifier, normalizer, benchmarking
      execution/        NinjaTrader client, risk engine
      storage/          Session and event persistence
    tests/              21 test modules
frontend/               React dashboard
bridges/    C# NinjaTrader 8 AddOn
transcripts/            Archived stream transcripts for training
```

## Configuration

All settings are in `backend/.env` (copy from `.env.example`). Main groups:

| Group | What it controls |
|-------|-----------------|
| Transcription | Which Whisper models to use, CPU vs GPU |
| VAD | Voice detection sensitivity and timing |
| Interpretation | Rule engine mode, classifier thresholds |
| Gemini | Cloud AI fallback (off by default) |
| Risk | Confidence limits, max position size, signal age |
| Execution | NinjaTrader connection, default symbol, contract size |

## API

### REST (prefix: `/api`)

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Health check |
| GET | `/sessions` | List all sessions |
| POST | `/sessions` | Create a new session |
| GET | `/sessions/{id}` | Get session state |
| DELETE | `/sessions/{id}` | Delete a session |
| PATCH | `/sessions/{id}/config` | Update session config |
| POST | `/sessions/{id}/segments` | Inject transcript text |
| POST | `/sessions/{id}/manual-trade` | Manual BUY/SELL/CLOSE |
| GET | `/sessions/{id}/broker-state` | Current position and PnL |

### WebSocket

| Path | What it does |
|------|--------------|
| `/ws/sessions/{id}/events` | Live updates (snapshot on connect, then deltas) |
| `/ws/sessions/{id}/audio` | Send audio from the browser (48 kHz PCM) |

## Tests

```bash
cd backend
pytest app/tests
```

## Training Data Pipeline

Tools for building classifier training data from archived YouTube streams:

1. **Download captions** — grab YouTube captions for a channel
2. **AI annotation** — have Gemini label trading actions in transcripts
3. **Reviewed corpus** — higher-quality labels with automatic review
4. **Build dataset** — merge everything into a single training file (JSONL)

Run from `backend/`:

```bash
# 1. Download
python -m app.services.transcription.youtube_captions \
  "https://www.youtube.com/@Channel/streams" --out-dir ../transcripts/channel

# 2. Annotate
python -m app.services.interpretation.ai_transcript_annotator \
  ../transcripts --symbol "MNQ 03-26" --market-price 24600 \
  --jsonl-out data/interpretation/ai_intent_examples.jsonl

# 3. Review
python -m app.services.interpretation.build_reviewed_ai_corpus \
  ../transcripts --backend gemini_cli --model gemini-2.5-pro \
  --output-dir data/interpretation/full_ai_corpus

# 4. Merge
python -m app.services.interpretation.build_reviewed_execution_dataset \
  data/interpretation/full_ai_corpus \
  --jsonl-out data/interpretation/reviewed_execution_intent_examples.jsonl
```

## Model Benchmarking

Compare ML models for trade-intent classification:

```bash
python -m app.services.interpretation.benchmark_models \
  --models tfidf_logreg tfidf_svm tfidf_mlp distilbert modernbert \
  --cv 5 --output data/bench_models/results.json
```

Tests five approaches: three classical (TF-IDF with Logistic Regression, SVM, or MLP) and two transformer-based (frozen DistilBERT or ModernBERT encoder with a trained head). Uses transcript-level fold splitting so no transcript leaks across train/test.
