# Gemma Local Chatbot

A production-quality, local-first AI chatbot with:

- **Backend:** FastAPI + Pydantic v2 + class-based OOP architecture
- **Model runtime:** Gemma 4 2B via `mlx-lm` (Apple Silicon) or `llama-cpp-python` (Intel)
- **Frontend:** React 18 + TypeScript + Tailwind CSS v3 + Zustand
- **Streaming:** Server-Sent Events (SSE) token streaming
- **Testing:** pytest + pytest-asyncio + httpx (backend), Vitest + RTL (frontend)
- **Auth:** None (local-only)

## Project Structure

```text
gemma-chatbot/
├── backend/
│   ├── main.py
│   ├── hardware.py
│   ├── quantization.py
│   ├── model_manager.py
│   ├── skills.py
│   ├── validators.py
│   ├── rate_limiter.py
│   ├── metrics.py
│   ├── schemas.py
│   ├── errors.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Chat.tsx
│   │   │   └── Admin.tsx
│   │   ├── components/
│   │   │   ├── MessageList.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── TypingIndicator.tsx
│   │   │   ├── SkillToggle.tsx
│   │   │   ├── AdminCard.tsx
│   │   │   ├── SkillUsageBar.tsx
│   │   │   ├── Toast.tsx
│   │   │   └── ErrorBoundary.tsx
│   │   ├── stores/
│   │   │   ├── chatStore.ts
│   │   │   └── adminStore.ts
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   └── types.ts
│   │   └── __tests__/
│   │       ├── Chat.test.tsx
│   │       └── Admin.test.tsx
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── tsconfig.json
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_classes.py
└── README.md
```

## Backend Setup

1. Create and activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Run API server:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Model Runtime Notes

- **Apple Silicon:** `MLXQuantization` is selected automatically.
  - `>= 16 GB RAM` -> `INT4`
  - `8-15 GB RAM` -> `INT8`
- **Intel / non-Metal:** `LlamaCppQuantization` with `Q4_K_M` GGUF path `models/gemma-4-2b-it.Q4_K_M.gguf`

Set `GEMMA_SKIP_MODEL_LOAD=1` for local CI/test runs without loading weights.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend dev server runs on `http://127.0.0.1:5173`.

## API Endpoints

- `POST /api/chat` (SSE stream)
- `GET /api/health`
- `GET /api/admin`
- `GET /api/skills`

## Security Controls Implemented

- Strict Pydantic v2 request validation
- Message sanitization (null bytes, control chars, Unicode direction overrides)
- Prompt injection pattern blocking (`HTTP 400`)
- In-memory IP rate limiting (30 req/min, sliding window)
- Request body size limit (64 KB)
- Restrictive CORS for local frontend origins
- SSE output token escaping
- Safe error envelopes with `request_id`
- Security headers on every response

## Running Tests

### Backend

```bash
pytest tests/ -v --asyncio-mode=auto
```

### Frontend

```bash
cd frontend
npx vitest run
```

## Local Usage

1. Start backend (`uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`).
2. Start frontend (`npm run dev`).
3. Open `http://127.0.0.1:5173/chat`.
4. Open `http://127.0.0.1:5173/admin` for runtime metrics.
