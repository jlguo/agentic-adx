# GPR ADX — Agentic Ad Exchange

> Generative Pre-trained Recommendation (GPR) — a unified LLM-based ad exchange that replaces traditional recall→ranking pipelines with a single model.

## Architecture

```mermaid
sequenceDiagram
    autonumber
    participant SSP as 🖥 SSP Simulator
    participant NG as 🌐 Nginx Gateway
    participant ADX as ⚙️ Go ADX Core
    participant RF as 📊 Redis Filter
    participant QD as 🔍 Qdrant Vector
    participant AB as 🧪 AB Manager
    participant RC as 💾 Redis ScoreCache
    participant LLM as 🧠 llama.cpp<br/>(CPU or GPU)
    participant FB as ⚠️ Fallback
    participant BL as 📈 Baseline DeepFM
    participant AU as 🏷️ Auction
    participant KF as 📨 Kafka

    SSP->>NG: POST /bid (OpenRTB 2.5)
    NG->>NG: Rate limit check
    NG->>ADX: Forward bid request
    ADX->>RF: Redis GET (budget/freq/blacklist)
    RF-->>ADX: Filter result
    ADX->>QD: Search(user_vector, topK=100)
    QD-->>ADX: Top-K ad candidates
    ADX->>AB: CRC32 hash → control/treatment?

    alt Variant = control
        ADX->>BL: Score(userCtx, adFeatures)
        BL-->>ADX: CTR/CVR/eCPM predictions
    else Variant = treatment
        ADX->>RC: HGETALL gpr_score:<ad_id>
        alt Cache HIT (O(1), ~1ms)
            RC-->>ADX: Pre-computed GPU scores
        else Cache MISS → LLM fallback
            ADX->>LLM: POST /v1/chat/completions
            alt LLM OK (~445ms GPU, higher CPU)
                LLM-->>ADX: JSON scores (CTR/CVR/eCPM)
            else LLM unreachable/error → hardcoded
                ADX->>FB: fallbackScores()
                FB-->>ADX: CTR=0.02, CVR=0.01, eCPM=bid×20
            end
        end
    end

    ADX->>AU: RunWithECPM(bids, scores)
    AU-->>ADX: Winning bid
    ADX->>KF: Produce(impression event) [async]
    ADX-->>NG: BidResponse
    NG-->>SSP: BidResponse

    Note over RC,FB: 3-tier fallback: cache → LLM → hardcoded<br/>Rate-limited logging (every 100th fallback)
```

### Background Services

```mermaid
sequenceDiagram
    autonumber
    participant SCR as cpu_scorer.py (GPU)
    participant MYS as MySQL
    participant RED as Redis
    participant KF as Kafka
    participant CLN as Sample Cleaner
    participant TRG as Training Trigger
    participant VUP as Vector Updater
    participant QD as Qdrant

    Note over SCR,RED: Every 30 seconds
    SCR->>MYS: SELECT active creatives
    MYS-->>SCR: Creative list
    SCR->>SCR: Tokenize → Qwen2-1.5B CUDA forward
    SCR->>RED: HSET gpr_score:<ad_id> {ctr,cvr,ecpm}

    rect rgb(240, 248, 255)
        Note over KF,QD: Async data loop
        ADX-)KF: Impressions / Clicks / Conversions
        KF->>CLN: Consume + join samples
        CLN->>TRG: Accumulate TSV (threshold: 500)
        TRG->>TRG: Trigger LoRA fine-tune
        VUP->>MYS: Poll for updated creatives
        VUP->>QD: Re-embed + upsert vectors
    end
```

### System Architecture

```mermaid
flowchart LR
    subgraph Hot["RTB Hot Path (<100ms P99)"]
        SS["SSP Simulator"] --> NG["Nginx Gateway"]
        NG --> ADX["Go ADX Core"]
        ADX --> FLT["Redis Filter<br/>Budget/Freq/Blacklist"]
        ADX --> VEC["Qdrant Vector Recall<br/>Semantic Top-K"]
        VEC --> GPR["GPR Inference<br/>CTR/CVR/eCPM"]
        GPR --> ADX
        ADX --> AU["Second-Price Auction"]
        AU --> RES["Bid Response"]
    end

    subgraph Cache["GPR Score Cache (O(1) Redis lookup)"]
        SCR["PyTorch Scorer<br/>GPU/CPU auto-detect<br/>Batch every 30s"]
    end
    SCR --> CACHE["Redis gpr_score:*"]

    subgraph Loop["Data Loop"]
        ADX --> KAFKA["Kafka Events"]
        KAFKA --> CLN["Sample Cleaner"]
        CLN --> TRG["Training Trigger"]
        TRG --> TRAIN["LoRA Fine-tune"]
        CLN --> VUP["Vector Updater"]
        VUP --> VEC
    end

    subgraph Agents["AI Agents (side-path, non-blocking)"]
        BID["Bidding Agent<br/>ROAS Optimization"]
        CRT["Creative Agent<br/>Material Generation"]
    end
    BID --> MYSQL[("MySQL<br/>Campaigns")]
    CRT --> MYSQL
    BID --> CK[("ClickHouse<br/>Analytics")]

    subgraph Obs["Observability"]
        ADX --> PRM["Prometheus"]
        ADX --> JGR["Jaeger Tracing"]
        PRM --> GRF["Grafana"]
        ADX --> LOG["Loki Logs"]
        LOG --> GRF
    end
```

## Five-Layer Design

| Layer | Role | Stack |
|---|---|---|
| **1. Access Gateway** | Traffic ingress, rate limiting, OpenRTB 2.5 parsing | Nginx + Gin |
| **2. ADX Trading** | Budget/frequency filtering, vector recall, GPR invocation, auction, A/B routing | Go + Redis + Qdrant |
| **3. GPR AI Sorting** | Hybrid: llama.cpp server (agent LLM) + PyTorch batch scorer (Redis-cached, GPU/CPU auto-detect) | Qwen2-1.5B + CUDA |
| **4. Data Loop** | Kafka → sample cleaning → LoRA fine-tune trigger → vector index update | Python + Kafka |
| **5. AI Agent Control** | Creative generation + bidding optimization — never in the RTB hot path | LangChain |

## Quick Start

```bash
# Full stack (14 services, requires Docker)
cd deploy && docker compose up -d

# Run traffic simulator
cd sim && go run ssp_sim.go -qps 10 -duration 30 http://localhost:8081/bid

# Unit tests
cd adx && go test ./...
python3 -m pytest agents/ data/ -q
```

## Service Ports

| Port | Service | Purpose |
|---|---|---|
| 8080 | Nginx | RTB entry |
| 8081 | ADX | HTTP direct (debug) |
| 9090 | Prometheus | Metrics UI |
| 9091 | ADX | `/metrics` endpoint |
| 3000 | Grafana | Dashboards |
| 3100 | Loki | Log aggregation |
| 6379 | Redis | Cache + filters |
| 3306 | MySQL | Campaign data |
| 8123 | ClickHouse | Analytics |
| 6333 | Qdrant | Vector search |
| 16686 | Jaeger | Distributed tracing |

## GPU Deployment

See [AGENTS.md](AGENTS.md#gpu-deployment) for full GPU host setup instructions.

```bash
# Quick start on GPU host (RTX 4090, 24GB, CUDA 12.6):
redis-server --daemonize yes

nohup python -m llama_cpp.server \
  --model /data/models/qwen2-1.5b-gguf/qwen2-1_5b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8000 --n_ctx 2048 --n_gpu_layers -1 &

nohup python gpr/serve/cpu_scorer.py \
  --model-path /data/models/qwen2-1.5b-hf --device cuda \
  --redis-addr localhost:6379 &

# ADX connects via:
export GPR_ADDR=<gpu-host>:8000
export GPR_REDIS_ADDR=<gpu-host>:6379
```

### Verified Performance (RTX 4090)

| Metric | Value |
|--------|-------|
| VRAM | 1.8 GB |
| Scoring latency (3 ads) | P50: 445ms |
| Latency variance | ±3ms |

## Key Design Decisions

- **No token decoding**: GPR is a discriminative model with structured output heads (CTR/CVR/eCPM), not text generation
- **Second-price auction** with reserve/floor pricing
- **A/B framework**: CRC32 hash-based deterministic routing; control=DeepFM, treatment=GPR
- **3-tier GPR fallback**: ① Redis cache (O(1), ~1ms) → ② llama.cpp server (POST /v1/chat/completions, CPU or GPU via `--n_gpu_layers`) → ③ hardcoded fallback (CTR=0.02, CVR=0.01) when llama.cpp is unreachable. Rate-limited logging prevents log spam.
- **Hybrid scoring**: llama.cpp server for agent LLM; batch scorer pre-computes all creatives every 30s via PyTorch CUDA → Redis cache → O(1) pipeline lookup
- **GPU acceleration**: Supports remote GPU host (RTX 4090 verified); CPU fallback via `DEVICE=auto`
- **All open-source**: no commercial dependencies, privately deployable
