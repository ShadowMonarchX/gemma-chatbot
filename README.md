# Gemma Chatbot (Local-First, Mac-Optimized)

Production-oriented local chatbot stack for Gemma 4 (2B) on macOS, with three layers:
- `backend/` FastAPI AI backend + REST API
- `frontend/` React + Vite + Tailwind UI
- `tests/` async pytest suite with mocked model backend

## Features

- Automatic Mac hardware detection at boot (`Apple Silicon` vs `Intel`, RAM, cores, Metal)
- Automatic quantization strategy selection:
  - Apple Silicon + Metal + `>= 16 GB` RAM -> `INT4-mlx`
  - Apple Silicon + Metal + `>= 8 GB` RAM -> `INT8-mlx`
  - Intel / no Metal -> `Q4_K_M-gguf` via `llama.cpp`
- Model loads once at startup and stays resident in RAM
- `/chat` supports multi-turn history and SSE token streaming
- `/health` and `/admin` provide runtime + performance telemetry
- Skill system shared across backend/frontend (`skills.py` and `skills.ts`)
- No authentication anywhere (local-first dev setup)

## Project Layout

```text
gemma-chatbot/
├── backend/
│   ├── __init__.py
│   ├── hardware.py
│   ├── main.py
│   ├── requirements.txt
│   └── skills.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   └── src/
│       ├── Admin.tsx
│       ├── App.tsx
│       ├── Chat.tsx
│       ├── index.css
│       ├── main.tsx
│       └── skills.ts
├── tests/
│   ├── conftest.py
│   └── test_backend.py
├── run.sh
└── README.md
```

## Install

From the `gemma-chatbot/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Run Backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Run Frontend

```bash
cd frontend
npm run dev
```

## One-command local run

```bash
./run.sh
```

This starts:
- Backend at `http://127.0.0.1:8000`
- Frontend at `http://127.0.0.1:5173`

## Run Tests

```bash
pytest tests/
```

## API Endpoints

- `POST /chat`
  - Body: `{"messages": [...], "skill": "chat"|"code", "stream": true|false}`
  - Returns SSE stream with `event: token` and `event: done`
  - Includes `x-response-ms` response header

- `GET /health`
  - Runtime health + model + hardware + throughput summary

- `GET /admin`
  - `/health` fields plus total requests, errors, avg response, skill usage

## Notes for Model Files

- MLX backends use model IDs from the Hugging Face ecosystem.
- GGUF backend expects `gemma-4-2b-it.Q4_K_M.gguf` in one of:
  - `./`
  - `./models/`
  - `./backend/models/`
- You can override via env vars:
  - `MLX_INT4_MODEL_ID`
  - `MLX_INT8_MODEL_ID`
  - `GGUF_MODEL_PATH`
