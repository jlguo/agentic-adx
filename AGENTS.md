# AGENTS.md — agentic-adx

> Implementation complete (Phase 1–4). See `draft/product-idea.md` for the full architecture spec.

## Project

GPR (Generative Pre-trained Recommendation) ADX — an agentic ad exchange that replaces traditional recall→ranking pipelines with a single LLM-based unified scoring model.

## Architecture (5-layer)

| Layer | Role | Status |
|---|---|---|
| 1. Access Gateway | Traffic ingress, rate limiting, OpenRTB 2.5 protocol parsing | ✅ |
| 2. ADX Trading (hot path) | Budget/frequency filtering, vector recall, GPR invocation (cached via Redis), second-price auction, **A/B traffic routing** | ✅ |
| 3. GPR AI Sorting | Hybrid: llama.cpp server (agent LLM) + PyTorch CPU batch scorer (Redis-cached scoring) | ✅ |
| 4. Data Loop | Kafka→sample cleaning→LoRA fine-tune trigger→vector index update | ✅ |
| 5. AI Agent Control (side-path) | Creative generation + bidding optimization agents — **NEVER in the RTB hot path** | ✅ |

## File Structure

```
adx/                    # Go ADX core (Layer 1-2)
├── cmd/server/         # Main binary (Gin HTTP + gRPC + Prometheus)
├── internal/
│   ├── ab/             # A/B experiment manager (CRC32 hash routing)
│   ├── auction/        # Second-price auction with eCPM
│   ├── baseline/       # DeepFM-style control group scorer
│   ├── event/          # Kafka async event producer (Sarama)
│   ├── fallback/       # LR fallback scorer (GPR timeout)
│   ├── filter/         # Redis budget/frequency/blacklist filter
│   ├── gpr/            # GPR inference HTTP client (vLLM)
│   ├── model/          # OpenRTB 2.5 data models
│   ├── pipeline/       # RTB pipeline: recall→score→auction→event
│   └── vector/         # Qdrant semantic vector recall client
│
gpr/                    # GPR model (Layer 3)
├── model/              # Qwen2-7B backbone + CTR/CVR/eCPM heads
├── embedding/          # Sentence-transformer ad embedding (384-dim)
├── train/              # LoRA pretraining (Criteo/Avazu + TSV samples)
├── serve/              # vLLM serving configuration
│   └── cpu_scorer.py   # PyTorch CPU batch scorer (→ Redis cache)

data/
└── flink/              # Data loop (Layer 4)
    ├── sample_cleaner.py    # Kafka consumer: impression+click+conversion→TSV
    ├── training_trigger.py  # TSV accumulation → LoRA fine-tune trigger
    └── vector_updater.py    # MySQL poll → re-embed → Qdrant upsert

agents/                 # AI Agents (Layer 5)
├── mab.py              # Epsilon-Greedy MAB (6 bid-multiplier arms)
├── compliance.py       # Two-layer ad compliance checker
├── creative_agent.py   # LangChain creative generation agent
└── bidding_agent.py    # LangChain ROAS optimization agent

sim/                    # SSP traffic simulator (Go)
deploy/                 # Docker Compose + infra configs
├── docker-compose.yml  # 14 services (all infra + ADX + agents)
├── grafana/
│   └── provisioning/   # Dashboards (8 panels) + datasources (Prometheus + Loki)
├── loki/               # Loki log aggregation config
├── promtail/           # Log shipper (container logs → Loki)
├── nginx/              # Reverse proxy + rate limiting
├── prometheus/         # Metrics scrape config (adx:9091 every 5s)
└── schema/             # MySQL init, ClickHouse init, seed data
demo/                   # Live demo scripts
├── run.sh              # Orchestrator (--dry-run / --traffic / full demo)
└── demo_flow.md        # 6-part presenter script (15-20 min)
draft/                  # Architecture spec (product-idea.md)
```

## Tech Stack

| Component | Choice |
|---|---|
| ADX core | Go + Gin + gRPC + Sarama (Kafka) + OpenTelemetry (Jaeger) |
| GPR inference | vLLM + Qwen2-7B or Llama3-8B (INT4/FP8) |
| GPR training | PyTorch + PEFT-LoRA (+ Criteo/Avazu pre-training) |
| Vector recall | Qdrant (semantic retrieval, not RAG) |
| Cache / rules | Redis 7.2 |
| Database | MySQL 8.0 |
| Message queue | Kafka 7.6 (KRaft mode, single-node) |
| Stream processing | Python Kafka consumers (MVP; Flink role fulfilled) |
| Analytics | ClickHouse 24.3 |
| Observability | Prometheus + Grafana + Loki + Jaeger (full stack) |
| Agents framework | LangChain |
| Deployment | Docker Compose (3 nodes, 14 services) |

## Hard Constraints

- **Agent side-path rule**: Creative and bidding agents operate on DB/cache only. Never in the synchronous RTB request path.
- **P99 latency target**: ADX end-to-end <100ms, GPR inference 30–60ms.
- **MVP scope**: No distributed HA, no multi-DC. Three-node Docker Compose deploy (business + data + GPU).
- **No token decoding**: GPR is a discriminative model with structured output heads, not text generation.
- **All open-source stack**: No commercial dependencies. Privately deployable.

## Key Design Decisions

- **Second-price auction (SPD)** with reserve/floor pricing
- **A/B framework** is a first-class component: traffic splitting, experiment isolation, automated metrics. CRC32 hash-based deterministic routing; control=DeepFM, treatment=GPR.
- **Terminology**: Qdrant is "semantic vector recall" (语义向量召回), not RAG
- **Timeout fallback**: GPR timeout → traditional LR/DNN scoring (no service degradation)
- **Stream processing**: Python Kafka consumers replace Apache Flink for MVP simplicity; same role (sample join, batch trigger, vector update)
- **OpenTelemetry tracing**: 5 span types (bid_request, vector_recall, gpr_score, baseline_score, auction) exported to Jaeger — every stage in the hot path is traceable
- **Structured logging**: Promtail ships ADX container logs to Loki; Grafana queries both Prometheus metrics and Loki logs in unified dashboards
- **Hybrid GPR scoring**: llama.cpp server (Qwen2-1.5B) serves agent LLM via OpenAI-compatible API; `cpu_scorer.py` batch-scores all active creatives every 30s via PyTorch CPU → Redis cache; ADX pipeline reads cached scores in O(1) Redis lookup (NO GPU required for demo)

## Test Status

| Suite | Tests | Result |
|---|---|---|
| Go (11 packages) | unit + integration | All pass ✅ |
| Python (124 tests) | unit + mock integration | 124 passed, 2 skipped ✅ |

## Demo

```bash
# Dry-run (no services needed — explains full architecture)
./demo/run.sh --dry-run

# Full live demo (requires Docker)
./demo/run.sh

# Traffic simulator against running ADX
./demo/run.sh --traffic http://localhost:8080/bid 20 30
```

See `demo/demo_flow.md` for the 6-part presenter script (15-20 min walkthrough).

## Service Port Map

| Host Port | Container | Internal Port | Purpose |
|-----------|-----------|---------------|---------|
| 8080 | nginx | 80 | RTB entry — proxies to `adx:8080` |
| 8081 | adx | 8080 | ADX HTTP direct (debug) |
| 9090 | prometheus | 9090 | Prometheus UI |
| 9091 | adx | 9091 | Prometheus metrics (`/metrics`) |
| 9092 | kafka | 9092 | Kafka PLAINTEXT |
| 3000 | grafana | 3000 | Grafana dashboards |
| 3100 | loki | 3100 | Loki log API |
| 6379 | redis | 6379 | Redis |
| 3306 | mysql | 3306 | MySQL |
| 8123 | clickhouse | 8123 | ClickHouse HTTP |
| 6333 | qdrant | 6333 | Qdrant REST API |
| 16686 | jaeger | 16686 | Jaeger UI |

**Internal-only** (Docker network, no host port): llama:8080 (llama.cpp API), qdrant:6334 (gRPC), jaeger:14268 (thrift collector), kafka:9093 (controller).

**Removed from host**: adx gRPC `:9090` (placeholder, no registered services), llama `:8080` (conflicted with nginx, internal-only).

## Known Fixes Applied

- **Kafka KRaft**: Requires `CLUSTER_ID` env var (base64 UUID) — added to docker-compose.
- **llama.cpp**: `libgomp.so.1` runtime dep — added `libgomp1` to `Dockerfile.llama` apt install to survive `autoremove`.
- **bidding-agent**: Changed from `python agents/bidding_agent.py` to `python -m agents.bidding_agent` — relative imports require module execution.
- **bidding-agent loop**: Wrapped in `sh -c "while true; do ...; sleep 3600; done"` — prevents Docker `restart: unless-stopped` from infinite-restarting the one-shot script when no campaigns exist. Matches original `--interval 3600` intent.
- **jaeger**: Added `user: root` — non-root user can't write to `/badger/` Docker volume.
- **qdrant healthcheck**: Replaced `curl` (not in image) with `wget`.
- **grafana/prometheus/loki healthchecks**: Added `-O /dev/null` to `wget` calls — default working dir is non-writable.
- **Dockerfile.adx**: Updated builder image `golang:1.22-alpine` → `golang:1.25-alpine` to match `go.mod`'s `go 1.25.8`.
- **llama healthcheck**: Changed from `/health` (404) to `/v1/models` (200) — llama.cpp exposes no `/health` endpoint.
- **ClickHouse auth**: Set `CLICKHOUSE_PASSWORD=clickhouse`, updated `CLICKHOUSE_URL` to `http://default:clickhouse@clickhouse:8123` — server requires authentication.
- **ClickHouse schema**: Created `adx_analytics.ad_metrics` table and mounted `clickhouse_init.sql` into initdb dir for fresh starts.
- **gpr-scorer MySQL query**: Removed `bid_price` from `SELECT` — column is in `campaigns`, not `creatives`, and was never used (eCPM is computed by the neural head).
- **Port conflicts**: Removed host port mappings for `llama:8080` (nginx collision) and `adx:9090` (gRPC placeholder, no registered services).

## Running

```bash
# Unit tests
cd adx && go test ./...
python -m pytest agents/ data/flink/ data/test_ab_report.py

# Full stack (requires Docker)
cd deploy && docker-compose up -d
```
