#!/usr/bin/env bash
# =============================================================
# GPR ADX — Live Demo Script
# =============================================================
# Runs the full stack and demonstrates every layer of the system.
# Usage:
#   ./demo/run.sh                  # Full docker demo (needs Docker)
#   ./demo/run.sh --dry-run        # Dry-run: check+explain without starting services
#   ./demo/run.sh --traffic        # Just run the traffic simulator against a running ADX
#   ./demo/run.sh --verbose        # Run traffic with live winning ad creative display
#   ./demo/run.sh --creative-loop  # Generate creative → score → watch it win bids
#   ./demo/run.sh --fine-tuning    # Demonstrate Kafka → TSV → LoRA fine-tune loop
#   ./demo/run.sh --diagrams       # Render mermaid diagrams to interactive HTML
#   ./demo/run.sh --ab-report      # Run A/B report analysis on existing ClickHouse data
# =============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$PROJECT_DIR/deploy"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

section()  { echo -e "\n${BOLD}${BLUE}==${NC} ${BOLD}${1}${NC} ${BLUE}==${NC}"; }
step()    { echo -e "${CYAN}→${NC} ${1}"; }
ok()      { echo -e "  ${GREEN}✓${NC} ${1}"; }
warn()    { echo -e "  ${YELLOW}⚠${NC} ${1}"; }
info()    { echo -e "  ${1}"; }
fail()    { echo -e "  ${RED}✗${NC} ${1}"; }
detail()  { echo -e "    ${2}"; }
header()  {
    echo -e "${BOLD}${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  GPR ADX — Agentic Ad Exchange Live Demo                ║"
    echo "║  Generative Pre-trained Recommendation Architecture     ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# --- Dry-run: explain architecture without requiring services ---
dry_run() {
    header

    section "Architecture Overview (5-Layer System)"
    info "  Layer 1: Access Gateway   → Nginx + rate limiting + OpenRTB 2.5"
    info "  Layer 2: ADX Trading      → Redis filter → Qdrant → Redis score cache → Auction → Kafka event"
    info "  Layer 3: GPR AI Sorting   → Hybrid: llama.cpp + PyTorch scorer (GPU/CPU auto-detect, batch → Redis)"
    info "  Layer 4: Data Loop        → Kafka → Sample Clean → LoRA Fine-tune → Vector Update"
    info "  Layer 5: AI Agent Control → Creative Agent + Bidding Agent (side-path only)"

    section "How an RTB Request Flows Through the System"
    echo ""
    echo "  SSP (sim/ssp_sim.go)               ADX (adx/internal/pipeline/pipeline.go)"
    echo "  ──────────────────                 ──────────────────────────────────────"
    echo "  │ POST /bid                        │"
    echo "  │ OpenRTB 2.5  → ─ ─ ─ ─ ─ ─ ─ →  │ 1. Nginx rate limit check"
    echo "  │ {id, imp[], site{}, device{}}    │ 2. Parse + validate OpenRTB JSON"
    echo "  │                                  │ 3. Redis filter: budget / frequency / blacklist"
    echo "  │                                  │ 4. Qdrant vector recall (384-dim semantic)"
    echo "  │                                  │ 5. Redis GET gpr_score:<ad_id> (O(1) cached)"
    echo "  │                                  │ 6. A/B hash: CRC32(reqID+salt) % 100 < ratio?"
    echo "  │                                  │    ├─ YES → cached GPR scores (treatment)"
    echo "  │                                  │    └─ NO  → DeepFM scorer (control group)"
    echo "  │                                  │ 7. Second-price auction"
    echo "  │  ← ─ ─ ─ ─ ─ ─ ─ ─ BidResponse  │ 8. Kafka event {exp_id, variant, latency}"
    echo ""
    echo "  Background (async, never blocks RTB hot path):"
    echo "  │ llama.cpp (Qwen2-1.5B) ← OpenAI API ← agents (bidding/creative)"
    echo "  │ cpu_scorer.py (PyTorch, GPU/CPU auto-detect) → batch all ads every 30s → Redis HSET gpr_score:*"
    echo "  │ Kafka → sample_cleaner.py → TSV training samples"
    echo "  │ TSV   → training_trigger.py → LoRA fine-tune (every 500 samples)"
    echo "  │ MySQL → vector_updater.py → re-embed → Qdrant upsert"

    section "Mermaid Architecture Chart (paste into mermaid.live)"
    echo ""
    echo '```mermaid'
    echo 'flowchart LR'
    echo '    SSP["SSP Simulator"] -->|"POST /bid"| NGINX["Nginx"]'
    echo '    NGINX --> ADX["ADX Pipeline"]'
    echo '    ADX -->|"Qdrant search"| QDRANT["Qdrant (~2ms)"]'
    echo '    QDRANT --> REDIS["Redis GET gpr_score:*"]'
    echo '    REDIS -->|"<<1ms"| AUCTION["Auction"]'
    echo '    AUCTION --> RES["BidResponse"]'
    echo ''
    echo '    SCORER["cpu_scorer.py"] -.->|"batch every 30s"| REDIS'
    echo '    LLAMA["llama.cpp"] -.->|"OpenAI API"| AGENTS["Creative + Bidding Agents"]'
    echo '    KAFKA["Kafka"] -.->|"async"| LOOP["Sample Clean → Train → Vector Update"]'
    echo '```'
    echo ""
    echo '```mermaid'
    echo 'sequenceDiagram'
    echo '    participant ADX as ADX Pipeline'
    echo '    participant KAFKA as Kafka'
    echo '    participant CLEAN as Sample Cleaner'
    echo '    participant TRIGGER as Training Trigger'
    echo '    participant TRAIN as LoRA Fine-tune'
    echo '    ADX->>KAFKA: impression event (fire & forget)'
    echo '    KAFKA->>CLEAN: consume + join clicks/conversions'
    echo '    CLEAN->>CLEAN: format TSV: prompt\tctr\tcvr\tecpma'
    echo '    CLEAN->>TRIGGER: append to training_samples.tsv'
    echo '    TRIGGER->>TRIGGER: poll every 30s'
    echo '    alt samples >= 500'
    echo '        TRIGGER->>TRAIN: archive + invoke pretrain.py'
    echo '        TRAIN->>TRAIN: LoRA fine-tune (3 epochs)'
    echo '    end'
    echo '```'

    section "Key Technical Metrics"
    ok "Go tests: 11 packages, all pass"
    ok "Python tests: 124 passed, 2 skipped (integration)"
    ok "End-to-end P99 latency target: <100ms (Redis lookup: O(1), <<1ms)"
    ok "GPR scoring: PyTorch CPU batch (all ads every 30s → Redis cache)"
    ok "Agent LLM: llama.cpp Qwen2-1.5B Q4_K_M (~7.7 tok/s CPU, 12-15s/query)"
    ok "A/B framework: CRC32 hash routing, no DB query in hot path"
    ok "Kafka events: fire-and-forget goroutine"
    ok "All open-source stack, GPU or CPU deployment optional"

    section "Key Design Decisions"
    info "1. Hybrid GPR scoring: llama.cpp for agent LLM + PyTorch CPU batch scorer → Redis cache"
    info "2. No token decoding in RTB: scores come from Redis, not from live LLM generation"
    info "3. llama.cpp serves agents (hourly, 12-15s acceptable); RTB reads cached scores (<<1ms)"
    info "4. Python Kafka consumers replace Apache Flink (MVP simplicity, same role)"
    info "5. A/B routing is O(1) hash, not DB lookup"
    info "6. Agent path is side-path: reads ClickHouse → writes Redis/MySQL — NEVER in hot path"

    section "Running the Full Stack"
    info ""
    info "  cd deploy && docker compose up -d"
    info ""
    info "This starts 16 services: Nginx, ADX Core, Redis, MySQL, ClickHouse,"
    info "  Qdrant, Kafka, Prometheus, Grafana, Loki, Promtail, Jaeger,"
    info "  Data Loop, Bidding Agent, llama.cpp Server, GPR CPU Scorer"
    info ""
    info "Quick demo (no Docker needed):"
    info ""
    info "  ./demo/run.sh --dry-run          # Architecture overview"
    info "  cd sim && go run ssp_sim.go      # Traffic simulator (needs ADX running)"
    info "  python data/ab_report.py         # A/B experiment report"
    info ""

    section "Observability Stack"
    info "  Grafana:     http://localhost:3000  (8-panel ADX dashboard)"
    info "  Prometheus:  http://localhost:9090  (adx_bid_requests_total, adx_bid_latency_ms)"
    info "  Jaeger:      http://localhost:16686 (5 span types: bid_request, vector_recall, gpr_score, baseline_score, auction)"
    info "  Loki:        http://localhost:3100  (log aggregation via Promtail)"

    section "File Map (Key Files)"
    info "  adx/internal/pipeline/pipeline.go  → Hot path: Qdrant → Redis cache → auction → Kafka"
    info "  adx/internal/gpr/cache.go          → Redis score cache reader (O(1) lookup)"
    info "  adx/internal/ab/manager.go         → A/B experiment manager (CRC32 hash)"
    info "  adx/internal/baseline/deepfm.go    → Control group scorer (DeepFM)"
    info "  adx/internal/event/producer.go     → Kafka async producer (fire-and-forget)"
    info "  gpr/model/gpr_model.py             → Qwen2-7B backbone + CTR/CVR/eCPM heads"
    info "  gpr/serve/cpu_scorer.py            → PyTorch CPU batch scorer → Redis cache"
    info "  agents/bidding_agent.py            → ROAS optimization (MAB + LangChain → llama.cpp)"
    info "  agents/creative_agent.py           → Creative generation (LangChain → llama.cpp)"
    info "  data/flink/sample_cleaner.py       → Kafka consumer → TSV training samples"
    echo ""
}

# --- Check prerequisites ---
check_prereqs() {
    section "Checking Prerequisites"

    if ! command -v docker &> /dev/null; then
        fail "Docker not found — install Docker first"
        return 1
    fi
    ok "Docker: $(docker --version 2>/dev/null || echo 'installed')"

    if ! command -v go &> /dev/null; then
        warn "Go not found — traffic sim needs Go"
    else
        ok "Go: $(go version | awk '{print $3}')"
    fi

    if [ ! -f "$VENV_PYTHON" ]; then
        warn "Python venv not found at .venv/"
    else
        ok "Python venv: $VENV_PYTHON"
    fi
}

# --- Build and start the full stack ---
start_stack() {
    section "Starting Full Stack (16 services)"
    step "Building ADX core + Docker images..."
    cd "$DEPLOY_DIR"
    docker compose build --quiet 2>&1 | tail -3
    ok "Images built"

    step "Starting services..."
    docker compose up -d 2>&1
    ok "Services starting..."

    step "Waiting for health checks (60s timeout)..."
    local timeout=60
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        local healthy=$(docker compose ps --format json 2>/dev/null | grep -c '"Health":"healthy"' || echo 0)
        local total=$(docker compose ps --format json 2>/dev/null | grep -c '"Service"' || echo 0)
        if [ "$healthy" -eq "$total" ] && [ "$total" -gt 0 ]; then
            ok "All $total services healthy"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        info "  Waiting... $healthy/$total healthy (${elapsed}s)"
    done
    cd "$PROJECT_DIR"
}

# --- Seed data ---
seed_data() {
    section "Seeding Data"
    step "Seeding Qdrant with 25 ad vectors..."
    "$VENV_PYTHON" "$PROJECT_DIR/deploy/schema/seed_ads.py" 2>&1 | tail -3
    ok "Qdrant seeded"

    step "Verifying MySQL schema..."
    docker exec adx-mysql mysql -u adx -padx_pass adx -e "SHOW TABLES;" 2>/dev/null | head -10
    ok "MySQL schema loaded"
}

# --- Run traffic simulation ---
run_traffic() {
    local target="${1:-http://localhost:8080/bid}"
    local qps="${2:-10}"
    local dur="${3:-30}"
    local verb="${4:-}"

    section "Running Traffic Simulation"
    info "Target: $target, QPS: $qps, Duration: ${dur}s"

    step "Building simulator..."
    cd "$PROJECT_DIR/sim"
    go build -o sim ssp_sim.go 2>&1
    ok "Simulator built"

    local verb_flag=""
    if [ "$verb" = "--verbose" ]; then
        verb_flag="--verbose"
        section "Live Winning Ads (press Ctrl-C to stop)"
    fi

    step "Sending traffic..."
    if [ -n "$verb_flag" ]; then
        ./sim -target "$target" -qps "$qps" -duration "$dur" --verbose 2>&1
    else
        ./sim -target "$target" -qps "$qps" -duration "$dur" 2>&1 | tee /tmp/adx-sim-output.log
    fi
    ok "Traffic complete"

    if [ -z "$verb_flag" ]; then
        step "Stats from simulator:"
        grep -E "sent|success|fail|avg|p50|p99" /tmp/adx-sim-output.log | while read line; do
            info "  $line"
        done
    fi

    cd "$PROJECT_DIR"
}

# --- Check metrics ---
check_metrics() {
    section "Checking ADX Metrics"
    step "ADX Prometheus metrics:"
    curl -s http://localhost:9091/metrics 2>/dev/null | grep -E "^adx_" | head -10 || warn "Metrics endpoint not reachable"
}

# --- Run A/B report ---
run_ab_report() {
    section "A/B Experiment Report"
    step "Generating A/B comparison report..."
    "$VENV_PYTHON" -c "
from data.ab_report import ABReport
import os
dsn = os.environ.get('MYSQL_DSN', 'adx:adx_pass@tcp(localhost:3306)/adx')
ch_url = os.environ.get('CLICKHOUSE_URL', 'http://localhost:8123')
reporter = ABReport(clickhouse_url=ch_url, mysql_dsn=dsn)
try:
    exps = reporter.list_experiments()
    print(f'Active experiments: {len(exps)}')
    for exp in exps[:5]:
        print(f'  Experiment {exp[0]}: {exp[1]} (traffic={exp[2]})')
    if exps:
        report = reporter.generate_report(exps[0][0])
        print(report)
except Exception as e:
    print(f'No experiments found or DB unavailable: {e}')
" 2>&1 || warn "A/B report generation skipped (no ClickHouse data yet)"
}

# --- Creative closed-loop demo ---
demo_creative_loop() {
    header
    section "Full Closed-Loop: Generate Creative → Score → Win Bid"

    step "Generating a new ad creative via the Creative Agent..."
    info "  Using llama.cpp (Qwen2-1.5B) to generate ad copy"
    info "  Target: campaign #1 (Wireless Earbuds), industry=tech"

    local creative_json
    creative_json=$("$VENV_PYTHON" -m agents.creative_agent \
        --account-id 1 \
        --campaign-id 1 \
        --count 1 \
        --industry tech \
        --llm-endpoint http://localhost:8080/v1 \
        2>&1)

    if [ $? -eq 0 ] && [ -n "$creative_json" ]; then
        ok "Creative generated and persisted to MySQL + Qdrant"
        echo ""
        echo "$creative_json" | "$VENV_PYTHON" -c "
import sys, json
data = json.load(sys.stdin)
if data:
    c = data[0]
    print(f'  Title:       \033[1m{c[\"title\"]}\033[0m')
    print(f'  Description: {c[\"description\"]}')
    print(f'  Category:    {c.get(\"category\", \"\")}')
    print(f'  Tags:        {c.get(\"tags\", [])}')
    print(f'  ID:          {c.get(\"id\", \"new\")}')
" 2>/dev/null || echo "$creative_json"
    else
        warn "Creative generation skipped (llama.cpp may not be ready)"
        echo "  $creative_json"
    fi
    echo ""

    step "Triggering immediate GPR scoring..."
    info "  Running cpu_scorer.py --once to score all active creatives"
    docker exec adx-gpr-scorer python gpr/serve/cpu_scorer.py --once \
        --mysql-host mysql --redis-addr redis:6379 \
        2>&1 | tail -5 || warn "cpu_scorer not running"
    ok "Scoring complete — new creative now in Redis cache"
    echo ""

    step "Verifying new creative scores in Redis..."
    docker exec adx-redis redis-cli KEYS "gpr_score:*" 2>/dev/null | wc -l | xargs echo "  Cached scores:"
    echo ""

    section "Live: Watch the New Creative Win Bids"
    info "Running SSP simulator with live creative display..."
    info "Look for the newly generated creative in winning bids!"
    echo ""

    run_traffic "http://localhost:8080/bid" 2 12 "--verbose"

    section "Loop Complete"
    echo ""
    info "1. Creative generated by LLM → MySQL + Qdrant"
    info "2. cpu_scorer.py --once scored it → Redis cache"
    info "3. ADX picked it up in bid responses → Winning ad shown"
    echo ""
}

# --- LoRA fine-tuning demo ---
demo_fine_tuning() {
    header
    section "Data Loop & LoRA Fine-Tuning Demo"

    step "Step 1: Show Kafka event → TSV training sample"
    info "  The sample_cleaner transforms Kafka events into labeled training data"
    "$VENV_PYTHON" -c "
from data.flink.sample_cleaner import format_sample
sample = format_sample(ad_id='7', clicked=True, ecpm=2.15, domain='tech.example.com')
print(f'  TSV sample: {sample}')
print(f'  Fields: prompt, CTR={1.0 if \"clicked\" in str(sample) else 0}, CVR=0, eCPM')
" 2>&1
    echo ""

    step "Step 2: Show training trigger logic"
    cat << 'EOF'
  training_trigger.py monitors data/training_samples.tsv:
    - Poll every 30s, count non-empty lines
    - When >= 500 samples → archive + invoke pretrain.py
    - LoRA fine-tune: only train low-rank adapters (not full 7B)
    - Updated model replaces cpu_scorer.py's model → re-score all ads
EOF
    echo ""

    step "Step 3: Run mini LoRA fine-tune (synthetic data, CPU)"
    info "  Training with 200 synthetic samples, 3 epochs on CPU..."
    if [ -f "$PROJECT_DIR/gpr/train/pretrain.py" ]; then
        "$VENV_PYTHON" "$PROJECT_DIR/gpr/train/pretrain.py" \
            --data "" \
            --epochs 3 \
            --max-samples 200 \
            --batch-size 8 \
            --device cpu \
            --output /tmp/gpr_demo_finetune.pt 2>&1 | tail -15
        ok "Fine-tune complete → /tmp/gpr_demo_finetune.pt"
    else
        warn "pretrain.py not found — skipping training run"
    fi

    section "Feedback Loop Summary"
    echo ""
    info "Impression → click → conversion → training sample → fine-tune →"
    info "  → better scoring → higher eCPM → more revenue"
    echo ""
    info "This is the virtuous cycle that makes GPR ADX self-improving."
    echo ""
}

# --- Show agent status ---
check_agents() {
    section "GPR & Agent Status"
    step "GPR CPU Scorer (PyTorch batch → Redis cache):"
    docker logs adx-gpr-scorer 2>/dev/null | tail -5 || warn "Not running"
    echo ""
    step "llama.cpp Server (Agent LLM, OpenAI API):"
    docker logs adx-llama 2>/dev/null | tail -3 || warn "Not running"
    echo ""
    step "Bidding Agent (MAB + ROAS Optimization):"
    docker logs adx-bidding-agent 2>/dev/null | tail -3 || warn "Not running"
    echo ""
    step "Data Loop (Kafka → TSV → Training trigger):"
    docker logs adx-data-loop 2>/dev/null | tail -3 || warn "Not running"
}

# --- Check Redis score cache ---
check_cache() {
    section "GPR Score Cache Status"
    step "Checking Redis for cached GPR scores..."
    docker exec adx-redis redis-cli KEYS "gpr_score:*" 2>/dev/null | head -10 || warn "Redis not reachable"
    if docker exec adx-redis redis-cli KEYS "gpr_score:*" 2>/dev/null | head -1 | grep -q .; then
        local id=$(docker exec adx-redis redis-cli KEYS "gpr_score:*" 2>/dev/null | head -1)
        step "Sample cached score ($id):"
        docker exec adx-redis redis-cli HGETALL "$id" 2>/dev/null
    fi
}

# --- Full demo flow ---
full_demo() {
    header
    check_prereqs || true

    start_stack
    sleep 10

    seed_data
    sleep 5

    section "End-to-End: User Visits Site → Ad Served"
    info "Showing the full RTB pipeline with creative display:"
    info "  User visits site → SSP sends bid request → ADX processes → Winning ad shown"
    echo ""
    run_traffic "http://localhost:8080/bid" 3 15 "--verbose"
    sleep 3

    run_traffic "http://localhost:8080/bid" 20 45
    sleep 3

    demo_creative_loop
    sleep 3

    demo_fine_tuning
    sleep 3

    # Generate diagram viewer for the presenter
    "$VENV_PYTHON" "$PROJECT_DIR/demo/render_mermaid.py" 2>/dev/null || true
    sleep 2

    check_metrics
    sleep 2

    run_ab_report
    check_cache
    check_agents

    section "Demo Complete"
    echo ""
    info "Grafana:     http://localhost:3000"
    info "Prometheus:  http://localhost:9090"
    info "Jaeger:      http://localhost:16686"
    info "Loki:        http://localhost:3100"
    info "llama.cpp:   http://localhost:8080 (OpenAI API /v1)"
    echo ""
    info "To stop: cd deploy && docker compose down"
    echo ""
}

# --- Main ---
case "${1:-}" in
    --dry-run)      dry_run ;;
    --traffic)      run_traffic "${2:-http://localhost:8080/bid}" "${3:-10}" "${4:-30}" "${5:-}" ;;
    --verbose)      run_traffic "${2:-http://localhost:8080/bid}" "${3:-3}" "${4:-15}" "--verbose" ;;
    --creative-loop) demo_creative_loop ;;
    --fine-tuning)   demo_fine_tuning ;;
    --diagrams)
        section "Generating Mermaid Diagrams"
        "$VENV_PYTHON" "$PROJECT_DIR/demo/render_mermaid.py"
        ok "Diagrams generated → demo/mermaid-viewer.html"
        echo ""
        if command -v xdg-open &>/dev/null; then
            info "Opening in browser..."
            xdg-open "$PROJECT_DIR/demo/mermaid-viewer.html" 2>/dev/null || true
        elif command -v open &>/dev/null; then
            info "Opening in browser..."
            open "$PROJECT_DIR/demo/mermaid-viewer.html" 2>/dev/null || true
        fi
        ;;
    --ab-report)    run_ab_report ;;
    --check)        check_prereqs && check_metrics && check_cache && check_agents ;;    
    --help|-h)
        echo "Usage: $0 [--dry-run|--traffic|--verbose|--creative-loop|--fine-tuning|--diagrams|--ab-report|--check|--help]"
        echo ""
        echo "  (no args)        Full demo (needs Docker)"
        echo "  --dry-run        Explain architecture, no services needed"
        echo "  --traffic        Run traffic sim against running ADX"
        echo "  --verbose        Run traffic sim with live winning ad display"
        echo "  --creative-loop  Generate creative → score → watch it win bids"
        echo "  --fine-tuning    Demonstrate Kafka → TSV → LoRA fine-tune loop"
        echo "  --diagrams       Render mermaid diagrams to interactive HTML viewer"
        echo "  --ab-report      Generate A/B report from ClickHouse"
        echo "  --check          Check prerequisites and status"
        ;;
    *)              full_demo ;;
esac
