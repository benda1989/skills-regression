# Regression Agent

A modular, secure, testable pure-backend regression analysis service powered by LLM.

## Features

- **7 regression models**: Linear, Ridge, Lasso, Logistic, Poisson, Negative Binomial, RDD
- **LLM-driven workflow**: understand → select model → fit → interpret
- **Secure API**: Bearer token auth, file size limits, CORS restriction
- **Caching**: MD5-based LLM response cache (FIFO, max 100 entries)
- **Tested**: 42 unit + integration tests

## Quick Start

```bash
# 1. Copy and fill in env vars
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the API server
uvicorn regression_agent.api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key for LLM provider |
| `OPENAI_BASE_URL` | No | OpenAI default | Base URL (for DeepSeek, Qwen, etc.) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model name |
| `OPENAI_TEMPERATURE` | No | `0.1` | Sampling temperature |
| `OPENAI_TIMEOUT` | No | `60.0` | Request timeout (seconds) |
| `OPENAI_MAX_RETRIES` | No | `3` | Max retry attempts on connection errors |
| `API_SECRET_KEY` | No* | — | Bearer token for protected endpoints |
| `ENV` | No | `development` | `development` enables `/docs`; `production` disables it |
| `CORS_ALLOWED_ORIGINS` | No | localhost ports | Comma-separated allowed origins |

*Leave `API_SECRET_KEY` empty to disable auth (development only).

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check |
| `POST` | `/api/analyze` | No | Upload file + run regression analysis |
| `GET` | `/api/history` | Bearer | List past analyses |
| `GET` | `/api/history/{id}` | Bearer | Get analysis detail |
| `DELETE` | `/api/history/{id}` | Bearer | Delete a record |
| `GET` | `/api/export/{id}` | Bearer | Export analysis as Word (.docx) |

### POST /api/analyze

```
Content-Type: multipart/form-data

file:        CSV or Excel file (max 10 MB)
instruction: Natural language instruction (optional)
             e.g. "分析收入对消费的影响，使用岭回归"
```

Response:
```json
{
  "id": "uuid",
  "data_understanding": {...},
  "model_selection": {...},
  "model_results": {...},
  "interpretation": {...},
  "analysis_context": {"target": "...", "features": [...]},
  "warnings": []
}
```

## Supported LLM Providers

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# DeepSeek
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# Alibaba Qwen (DashScope)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo
```

## Docker

```bash
docker-compose up -d
```

The service starts on port 8000. Data persists in a Docker volume (`regression_data`).

## Project Structure

```
skills-regression/
├── core/
│   ├── config.py          # LLMConfig (reads from env)
│   ├── llm.py             # LLM calls + MD5 cache
│   ├── data_loader.py     # CSV/Excel loading
│   ├── prompts.py         # LLM prompt constants
│   ├── agent.py           # Orchestration: understand→select→fit→interpret
│   └── models/
│       ├── linear.py      # Linear / Ridge / Lasso
│       ├── logistic.py    # Logistic regression
│       ├── count.py       # Poisson / Negative Binomial
│       └── rdd.py         # Regression Discontinuity Design
├── api/
│   ├── main.py            # FastAPI app entry point
│   ├── auth.py            # Bearer token middleware
│   ├── database.py        # SQLite persistence
│   ├── schemas.py         # Pydantic request/response models
│   └── routes/
│       ├── analyze.py     # POST /api/analyze
│       ├── history.py     # GET/DELETE /api/history
│       └── export.py      # GET /api/export
```
