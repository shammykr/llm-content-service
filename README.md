# LLM Content Service

A content processing service that uses LLM APIs for **summarization**, **translation** and
**content recommendations**. It exposes FastAPI endpoints for real-time requests and for
batch/streaming ingestion. LLM responses are cached in Redis, so repeated queries come
back fast. Ingested documents go through a small NLP preprocessing step before they are
queued for processing.

**Stack:** Python 3.12, FastAPI, any OpenAI-compatible LLM API (via the `openai` SDK), Redis, Docker

## Features

- **Content endpoints:** `POST /summarize`, `POST /translate` and `POST /recommend`. Each one calls an OpenAI-compatible chat completions API.
- **Redis response caching:** the cache key is `sha256(task + model + normalized input)`. Identical requests are served from Redis without calling the LLM. Responses carry `X-Cache: HIT|MISS` and `X-Cache-Lookup-Ms` headers.
- **Batch and streaming ingestion:**
  - `POST /ingest` takes a list of documents.
  - `POST /ingest/stream` takes an NDJSON body, and documents are queued in chunks while the upload is still arriving.
  - Each document is normalized, tokenized and language-detected, then pushed to a Redis queue.
  - Async workers summarize queued documents. `GET /ingest/{id}` returns a document's status and result.
- **Prompt optimization:**
  - *Prompt caching:* system prompts are static per task. They are built once, memoized and reused byte-for-byte on every call. All per-request parameters go in the user message, so the system prefix is stable and providers with automatic prefix caching can reuse it.
  - *Token batching:* small summarize/translate requests (≤ `SMALL_REQUEST_TOKENS`) that arrive within a short window (`BATCH_WINDOW_MS`) are merged into **one** LLM call. The model returns a JSON array with one result per item. If that reply can't be parsed, each item falls back to its own call.
  - *Async handling:* everything runs on asyncio: the async OpenAI client, `redis.asyncio` and concurrent ingest workers. Upstream concurrency is capped with a semaphore.

## Architecture

```
            ┌─────────────────────────── FastAPI (app/main.py) ───────────────────────────┐
 client ──► │ /summarize  /translate  /recommend          /ingest  /ingest/stream         │
            │        │                                         │                          │
            │        ▼                                         ▼                          │
            │  ContentService ──► ResponseCache ◄──┐    preprocess (normalize,            │
            │  (app/service.py)   (Redis GET/SET)  │     tokenize, langdetect)            │
            │        │ miss                        │           │                          │
            │        ▼                             │           ▼                          │
            │  TokenBatcher ── merges small ──►  LLMClient     Redis list ingest:queue    │
            │  (app/batching.py) requests       (AsyncOpenAI,        │                    │
            │                                    cached system       ▼                    │
            │                                    prompts)       ingest workers ──► ContentService
            └────────────────────────────────────────┼────────────────────────────────────┘
                                                     ▼
                                  OpenAI-compatible API  (or bench/mock_llm.py)
```

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, routes, lifespan wiring |
| `app/service.py` | Cache-first task logic (summarize / translate / recommend) |
| `app/cache.py` | Redis response cache, hashed keys |
| `app/llm.py` | Async OpenAI-compatible client with concurrency limit |
| `app/prompts.py` | Static, memoized system prompts and prompt builders |
| `app/batching.py` | Token batcher that merges small requests into one LLM call |
| `app/preprocess.py` | Normalization, tokenization, sentence split, language detection |
| `app/ingest.py` | Redis-backed ingestion queue and async workers |
| `bench/mock_llm.py` | Instant-response OpenAI-compatible mock backend |
| `bench/load_test.py` | Load generator (throughput and latency at increasing concurrency) |

## Running

### Set the API key

The service works with any OpenAI-compatible endpoint. Set these environment variables, or put them in a `.env` file next to `docker-compose.yml` (see `.env.example`):

```bash
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini          # optional
export LLM_BASE_URL=                  # optional, e.g. a vLLM/Ollama/Together endpoint ending in /v1
```

### Docker Compose (API + Redis)

```bash
docker compose up --build
curl -X POST localhost:8000/summarize -H 'content-type: application/json' \
     -d '{"text": "FastAPI is a modern, fast web framework for building APIs with Python.", "max_sentences": 1}'
```

Interactive API docs are served at http://localhost:8000/docs.

### Local

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
redis-server &                                  # or: docker compose up redis
.venv/bin/uvicorn app.main:app --port 8000
```

### API examples

```bash
# Translation
curl -X POST localhost:8000/translate -H 'content-type: application/json' \
     -d '{"text": "Hello, how are you?", "target_language": "French"}'

# Recommendations: rank candidate items by relevance to a source document
curl -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{
  "content": "An introduction to caching strategies with Redis",
  "candidates": [{"id": "a1", "title": "Redis eviction policies"},
                 {"id": "a2", "title": "Sourdough baking basics"},
                 {"id": "a3", "title": "CDN cache invalidation"}],
  "top_k": 2}'

# Batch ingestion
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
     -d '{"documents": [{"id": "doc-1", "text": "Le chat dort sur la table. Il fait beau."}]}'

# Streaming ingestion (NDJSON, one document per line)
printf '{"id":"doc-2","text":"First streamed doc."}\n{"text":"Second streamed doc."}\n' |
  curl -X POST localhost:8000/ingest/stream -H 'content-type: application/x-ndjson' --data-binary @-

# Ingestion status and result
curl localhost:8000/ingest/doc-1
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CACHE_TTL_SECONDS` | `86400` | Response cache TTL |
| `BATCH_WINDOW_MS` | `10` | How long the batcher waits to collect requests |
| `BATCH_MAX_ITEMS` / `BATCH_MAX_TOKENS` | `8` / `2000` | When a batch is flushed early |
| `SMALL_REQUEST_TOKENS` | `300` | Requests at or below this size are batched |
| `LLM_MAX_CONCURRENCY` | `64` | Max in-flight upstream LLM calls |
| `INGEST_WORKERS` | `4` | Async ingestion workers |

## Benchmark

`bench/load_test.py` sends `POST /summarize` requests at increasing concurrency, running each level for 10 s. It reports sustained req/s and p50/p99 latency. The service runs against `bench/mock_llm.py`, an OpenAI-compatible backend that answers instantly, so the numbers measure the service itself rather than an LLM provider.

- **cached**: 50 texts are warmed, then every request is a Redis cache hit.
- **uncached**: every request has unique text, so it misses the cache and goes through token batching to the mock LLM.

```bash
LLM_BASE_URL=http://mock-llm:9000/v1 docker compose --profile bench up --build -d
pip install -r bench/requirements.txt
python bench/load_test.py --url http://localhost:8000 --levels 1,10,50,100,200 --duration 10
```

## Results

Measured on an Apple M5 (10 cores, 16 GB), with a single uvicorn worker. The load generator ran on the same machine. 0 errors at every level.

**Docker Compose stack (API + Redis + mock LLM in containers):**

| Concurrency | Cached req/s | Cached p99 (ms) | Uncached req/s | Uncached p99 (ms) |
|---:|---:|---:|---:|---:|
| 1   | 2,489 | 0.63  | 56    | 21.1 |
| 10  | 7,130 | 2.15  | 3,338 | 9.6  |
| 50  | 7,000 | 28.8  | 2,894 | 51.8 |
| 100 | 6,345 | 40.9  | 3,182 | 72.3 |
| 200 | 5,524 | 89.2  | 3,313 | 86.7 |

**Native (uvicorn + local Redis, no Docker networking):**

| Concurrency | Cached req/s | Cached p99 (ms) | Uncached req/s | Uncached p99 (ms) |
|---:|---:|---:|---:|---:|
| 1   | 3,811  | 0.68 | 60    | 19.1 |
| 10  | 11,744 | 2.40 | 4,677 | 4.1  |
| 50  | 10,805 | 17.3 | 4,859 | 24.2 |
| 100 | 10,081 | 27.2 | 4,513 | 39.3 |
| 200 | 9,939  | 39.2 | 4,135 | 79.2 |

What the numbers show:

- **Throughput:** with Redis caching the service sustains **5.5k–10k req/s at 200 concurrent clients**, far above the 200 req/s target.
- **Cache lookup latency:** the server-side Redis lookup (`X-Cache-Lookup-Ms`) has a p50/p99 of **0.06/0.11 ms** (Docker) and **0.10/0.29 ms** (native) at concurrency 1. At high concurrency the measured time grows to a few ms. That extra time is mostly event-loop queueing inside the saturated process, not Redis itself.
- **Token batching:** in the uncached runs, about 128k LLM-bound requests were served by about 16.7k LLM calls, **7.7 items per call** on average.
- **Batching cost at low load:** uncached throughput at concurrency 1 is about 56–60 req/s. Each lone request waits out the 10 ms batching window. That trades a little single-request latency for far fewer upstream calls under load. Set `BATCH_WINDOW_MS=0` to remove the wait.
- **Real providers:** with a real LLM, uncached throughput will be bounded by the provider's latency and rate limits. Cached throughput is unaffected.
