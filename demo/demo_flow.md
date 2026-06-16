# GPR ADX — Demo Flow & Presenter Script

> 15-20 minute live demo walkthrough for technical audience.

## Setup (before audience arrives)

```bash
# Terminal 1: Full stack
cd deploy && docker compose up -d

# Terminal 2: Seed Qdrant + wait for GPR scorer to populate Redis cache
python deploy/schema/seed_ads.py
sleep 60  # give cpu_scorer.py time to batch-score all ads

# Terminal 3: Monitor
watch -n 2 'curl -s localhost:9091/metrics | grep adx_'
```

---

## Part 1 — Architecture Overview (3 min)

**Slide / Terminal: `demo/run.sh --dry-run`**

### What We Built

> "GPR ADX is an agentic ad exchange that replaces the traditional
> recall → coarse-rank → fine-rank pipeline with a single LLM-based
> unified scoring model. The key innovation: **hybrid scoring architecture.**
> The LLM batch-scores all ads asynchronously into Redis, and the RTB
> hot path reads cached scores in O(1) — no GPU, no token generation,
> sub-millisecond lookup."

### Five Layers

| Layer | What | Tech |
|-------|------|------|
| 1. Gateway | Rate limiting, OpenRTB 2.5 parse | Nginx + Gin |
| 2. ADX Trading | Qdrant → Redis score cache → Auction | Go, Qdrant, Redis |
| 3. GPR AI | Hybrid: llama.cpp + PyTorch CPU scorer | Qwen2-1.5B, Redis cache |
| 4. Data Loop | Sample → Train → Vector update | Kafka, Python |
| 5. AI Agents | Creative gen + Bidding optimization | LangChain → llama.cpp |

### The Architecture (show this diagram)

```
  RTB Hot Path (<1ms):                   Background (async):
  ┌────────────────────────┐             ┌──────────────────────────┐
  │ Nginx → ADX Pipeline   │             │ cpu_scorer.py (every 30s)│
  │ ├ Qdrant recall (2ms)  │             │ ├ Load all active ads    │
  │ ├ Redis GET scores     │             │ ├ PyTorch batch forward  │
  │ │  gpr_score:<ad_id>   │   Redis     │ ├ HSET gpr_score:*       │
  │ ├ A/B hash split       │◄──────────►│ └ TTL 120s               │
  │ └ Auction → bid        │   cache     │                          │
  └────────────────────────┘             │ llama.cpp (Qwen2-1.5B)   │
                                         │ ├ OpenAI API :8080/v1    │
  Agents (hourly):                       │ └ 7.7 tok/s CPU          │
  ┌────────────────────────┐             │                          │
  │ bidding_agent.py       │──HTTP──────►│                          │
  │ creative_agent.py      │             └──────────────────────────┘
  └────────────────────────┘
```

### Architecture Decisions to Highlight

1. **Hybrid GPR scoring** — The key insight:
   - llama.cpp serves the **agent LLM** (hourly bidding analysis, creative generation)
   - PyTorch CPU batch-scores all ads into **Redis cache** (every 30s)
   - ADX hot path reads Redis in **O(1)** — no GPU, no token generation, <<1ms
   - This is production-grade: async pre-computation + real-time cache lookup

2. **A/B is a first-class component** — CRC32 hash-based routing:
   - `hash(experiment_salt + request_id) % 100 < traffic_ratio * 100`
   - O(1), no DB query in hot path
   - Deterministic: same user always in same group

3. **Agent side-path rule** — Agents connect to llama.cpp OpenAI API:
   - NEVER in the synchronous RTB path
   - Hourly ROAS optimization (MAB arm selection)
   - Creative generation with compliance checking

4. **Full observability** — Prometheus + Grafana + Loki + Jaeger:
   - 8-panel Grafana dashboard (QPS, latency, split, errors)
   - 5 span types in Jaeger (bid_request, vector_recall, gpr_score, baseline_score, auction)
   - Container logs → Promtail → Loki → Grafana unified view

---

## Part 2 — Live Demo: Traffic Flow (5 min)

### 2a. Verify Redis Cache

```bash
# Show that GPR scores are pre-computed in Redis
docker exec adx-redis redis-cli KEYS "gpr_score:*" | head -5
docker exec adx-redis redis-cli HGETALL "gpr_score:1"
```

**Expected output:**
```
gpr_score:1
ctr
0.0453
cvr
0.0187
ecpm
0.85
```

**Point out:** "These scores were pre-computed by `cpu_scorer.py` — a Python service running in the background. Every 30 seconds it loads all active ads from MySQL, runs them through Qwen2-1.5B in a single PyTorch forward pass, and uploads to Redis. The ADX never touches the model directly."

### 2b. Start the Traffic Simulator

```bash
cd sim && go run ssp_sim.go -qps 10 -duration 30
```

**Show output:**
```
sent=300 success=298 fail=2
avg=4.2ms p50=3.8ms p99=12.1ms
```

**What's happening:**
- 10 QPS of synthetic OpenRTB 2.5 bid requests
- Pipeline: Qdrant recall (2ms) → Redis GET scores (<<1ms) → A/B split → auction (1ms)
- Total latency: **4-12ms** — well under the 100ms P99 target

### 2c. Show ADX Logs

```bash
docker logs adx-core --tail 10
```

**Example output:**
```
bid req-1718395200-12345: recall=50 in 2.1ms, gpr=true in 0.3ms, auction in 1.0ms, total=4.2ms
score cache: used pre-computed scores for 50 ads
```

**Point out:**
- `score cache: used pre-computed scores` — Redis cache hit, no model call
- `gpr=true` → treatment group (GPR cached scores)
- `gpr=false` → control group (DeepFM baseline scorer)
- Total latency ~4ms — Redis cache is the key enabler

### 2d. Show llama.cpp Server

```bash
# Test the OpenAI-compatible API
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Optimize bid for campaign 3 targeting tech audience with ROAS 3.0"}],"max_tokens":100}'
```

**Point out:** "The llama.cpp server is idle during RTB — it only handles agent queries. This separation is critical: the LLM is too slow for real-time bidding (12-15s per query), but perfect for hourly analysis. The RTB path never waits for an LLM."

---

## Part 3 — A/B Testing Framework (3 min)

### 3a. Show GPR CPU Scorer Logs

```bash
docker logs adx-gpr-scorer --tail 5
```

**Expected output:**
```
Scored 25 ads in 21347ms (avg 854ms/ad)
Updated 25 scores in Redis
```

### 3b. Show A/B Report

```bash
./demo/run.sh --ab-report
```

**Expected output format:**
```
=== Experiment 1: GPR vs DeepFM Baseline ===
Variant  | Imps    | CTR     | CVR     | eCPM   | Latency
---------|---------|---------|---------|--------|--------
control  | 15234   | 2.3%    | 0.8%    | $1.45  | 3.8ms
treatment| 14891   | 3.1%    | 1.1%    | $1.82  | 4.2ms

CTR Lift: +34.8%  |  CVR Lift: +37.5%  |  eCPM Lift: +25.5%
Latency Delta: +0.4ms (negligible — both read from Redis cache)

VERDICT: SIGNIFICANT IMPROVEMENT
```

---

## Part 4 — Data Loop (2 min)

### Show Kafka Events

```bash
docker exec adx-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic ad_impressions --max-messages 5
```

**Example event:**
```json
{
  "request_id": "req-1718395200-12345",
  "ad_id": 7,
  "campaign_id": 2,
  "bid_price": 1.80,
  "ecpm": 2.15,
  "experiment_id": "1",
  "variant": "treatment",
  "timestamp": "2026-06-15T10:30:00Z"
}
```

---

## Part 5 — AI Agents (3 min)

### 5a. Bidding Agent (MAB Demo)

```bash
python -c "
from agents.mab import EpsilonGreedyMAB, ARM_CONFIG
mab = EpsilonGreedyMAB(arms=ARM_CONFIG, epsilon=0.1)
for i in range(100):
    arm = mab.select_arm()
    reward = arm * 0.8 + 0.2
    mab.update(arm, reward)
print('Best arm:', mab.best_arm())
"
```

**Key features:**
- 6 bid multipliers [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- Connects to llama.cpp for LLM analysis (hourly)
- Writes optimized bids to Redis — never in hot path

### 5b. Creative Agent Demo

```bash
python -c "
from agents.creative_agent import build_creative_agent
agent = build_creative_agent()
result = agent.invoke({'input': 'Generate a summer travel insurance ad targeting young professionals'})
print(result['output'][:500])
"
```

---

## Part 6 — Observability (2 min)

### Show Grafana Dashboard

> Open `http://localhost:3000` (admin/admin)

**8 panels to show:**
1. Bid Request Rate (QPS) — real-time traffic
2. P50/P99 Latency — sub-5ms for Redis cache path
3. GPR vs DeepFM Split — pie chart of traffic distribution
4. Error Rate — near zero
5. Requests by Status — stacked bars
6. Latency Distribution — P50/P95/P99 over time

### Show Jaeger Traces

> Open `http://localhost:16686`

**5 span types visible:**
1. `bid_request` — full lifecycle with attributes (request_id, gpr_used)
2. `vector_recall` — Qdrant search with recall_count
3. `gpr_score` or `baseline_score` — variant marker
4. `auction` — winner selection

---

## Part 7 — Wrap-up (2 min)

### What We Demonstrated

1. ✅ Full RTB pipeline: Nginx → Qdrant → Redis cache → A/B split → Auction → Kafka
2. ✅ Hybrid GPR: llama.cpp for agents + PyTorch CPU batch scorer → Redis cache
3. ✅ RTB hot path <5ms (Redis O(1) lookup, no model calls)
4. ✅ A/B framework: CRC32 hash routing, control vs treatment scoring
5. ✅ Data loop: Kafka events → TSV samples → training trigger
6. ✅ AI agents: creative + bidding via llama.cpp (side-path)
7. ✅ Full observability: Prometheus + Grafana + Loki + Jaeger

### Architecture Advantages

- **Hybrid scoring**: async batch pre-computation + O(1) Redis cache = production-grade latency
- **No GPU required**: llama.cpp (CPU) for agents, PyTorch CPU for batch scoring
- **16 Docker services**: one `docker compose up -d` deploys everything
- **A/B by design**: CRC32 hash routing, not bolted on
- **Agent side-path**: LLM never blocks RTB
- **All open-source**: no vendor lock-in

### Q&A Topics (prepare)

1. **"Why Redis cache instead of live GPR scoring?"** — Live LLM inference takes 12-15s on CPU, far too slow for <100ms RTB. Async pre-computation + cache is a well-established pattern (Google AdX, Meta all pre-compute ad quality scores). The 30s refresh interval keeps scores fresh enough.

2. **"What about cold start?"** — 25 seed ads in Qdrant + DeepFM baseline with category priors. LR fallback scorer for any cache miss. On first startup, DeepFM serves until cpu_scorer.py completes its first batch (takes ~25s).

3. **"Production with GPU?"** — Replace `cpu_scorer.py` with vLLM serving. The ADX pipeline code is identical — it reads from Redis regardless of what populates it. Same architecture, just swap the scorer backend.

4. **"Why CRC32 hash, not random split?"** — Deterministic routing ensures the same user always sees the same variant. No sticky cookies needed.

---

## Quick Reference Card

| Command | Purpose | When |
|---------|---------|------|
| `./demo/run.sh --dry-run` | Architecture overview | Start of presentation |
| `docker compose up -d` | Start 16-service stack | Before live demo |
| `docker compose ps` | Check services | During startup |
| `python deploy/schema/seed_ads.py` | Seed Qdrant | After services healthy |
| `docker exec adx-redis redis-cli KEYS "gpr_score:*"` | Check GPR cache | After ~60s startup |
| `cd sim && go run ssp_sim.go -qps 10 -duration 30` | Send traffic | Live demo |
| `curl -s localhost:9091/metrics \| grep adx_` | Show metrics | After traffic |
| `docker logs adx-core --tail 10` | ADX logs | Show bid processing |
| `docker logs adx-gpr-scorer --tail 5` | GPR scorer logs | Show batch scoring |
| `docker logs adx-llama --tail 3` | llama.cpp logs | Show LLM status |
| `curl http://localhost:8080/v1/chat/completions` | Test llama.cpp API | Agent LLM demo |
| `docker compose down` | Stop everything | End of demo |
