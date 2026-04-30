# Gemma Local Chatbot

Production-grade, local-first AI chatbot with runtime model switching and hardware-aware inference routing.

## Highlights

- Local-only execution (no external APIs)
- Runtime model switching: `gemma-2b`, `gemma-e2b`, `gemma-e4b`
- Hardware auto-detection (Apple Silicon, Intel CPU, CUDA)
- Backend auto-selection:
  - Apple Silicon + Metal: `mlx-lm`
  - CUDA / CPU fallback: `llama-cpp-python`
- Streaming chat over SSE (`text/event-stream`)
- Strict security controls (validation, injection guard, rate limits, secure headers)
- Full React admin dashboard for runtime metrics

## Project Structure

```text
gemma-chatbot/
├── backend/
│   ├── config.py
│   ├── errors.py
│   ├── hardware.py
│   ├── main.py
│   ├── metrics.py
│   ├── model_manager.py
│   ├── quantization.py
│   ├── rate_limiter.py
│   ├── requirements.txt
│   ├── schemas.py
│   ├── skills.py
│   └── validators.py
├── frontend/
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── api/
│       ├── components/
│       ├── pages/
│       ├── stores/
│       └── __tests__/
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_classes.py
├── .env
├── .env.example
└── README.md
```

## Environment

Copy `.env.example` to `.env` and adjust values:

```env
MODEL_PATH=backend/models
DEFAULT_MODEL=gemma-2b
MAX_TOKENS=512
REQUEST_BODY_LIMIT_BYTES=65536
RATE_LIMIT_PER_MINUTE=30
SKIP_MODEL_LOAD=false
```

## Backend Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

## API Endpoints

- `POST /api/chat` (SSE stream)
- `GET /api/health`
- `GET /api/admin`
- `GET /api/skills`
- `GET /api/models`

### Chat Request Example

```json
{
  "messages": [{ "role": "user", "content": "Hello" }],
  "skill_id": "chat",
  "model_id": "gemma-2b",
  "stream": true
}
```

## Hardware Routing Logic

- Apple Silicon + Metal
  - Uses `MLXQuantization`
  - `INT4` when RAM >= 16 GB
  - `INT8` when RAM is 8-15 GB
- CUDA available (non-Apple path)
  - Uses `LlamaCppQuantization` with GPU offload
- Intel / CPU only
  - Uses `LlamaCppQuantization` (`Q4_K_M`)

## Security Controls

- Pydantic v2 strict request validation
- Message size: max 4096 chars
- History size: max 20 messages
- Input sanitization (control chars, null bytes, direction overrides)
- Prompt injection detection and rejection (`HTTP 400`)
- Rate limiting: 30 requests/min/IP (`HTTP 429` + `Retry-After`)
- Request body limit: 64 KB (`HTTP 413`)
- SSE token escaping (`html.escape`)
- CORS restricted to local Vite origins
- Hardened response headers
- No stack traces leaked to clients

## Testing

Backend:

```bash
PYTHONPATH=. uv run pytest tests/ -v --asyncio-mode=auto
```

Frontend:

```bash
cd frontend
npx vitest run
```

Build frontend:

```bash
cd frontend
npm run build
```

## Notes

- Place GGUF files under `backend/models/` for llama.cpp runtime.
- `SKIP_MODEL_LOAD=true` can be used in constrained environments to boot without loading model weights.
