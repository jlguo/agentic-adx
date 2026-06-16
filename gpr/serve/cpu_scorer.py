"""
GPR CPU Scorer — periodic batch-scoring service.

Loads Qwen2-1.5B backbone + GPR custom heads (PyTorch CPU),
queries MySQL for active ad creatives, batch-scores them in one
forward pass, and writes results to Redis cache.

Redis key format:  HSET gpr_score:<ad_id> ctr <val> cvr <val> ecpm <val>
TTL: 120 seconds so stale scores expire.

Intended for standalone deployment:  python gpr/serve/cpu_scorer.py
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pymysql
import redis
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("cpu_scorer")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_RUNNING = True


def _shutdown_handler(signum: int, _frame: Any) -> None:
    global _RUNNING
    logger.info("Received signal %d, shutting down gracefully...", signum)
    _RUNNING = False


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScorerConfig:
    """Runtime configuration – overridable via env vars or CLI args."""

    # -- Model ----------------------------------------------------------------
    model_path: str = "Qwen/Qwen2-1.5B"
    hidden_size: int = 1536  # Qwen2-1.5B embedding dim
    ctr_hidden: int = 512
    cvr_hidden: int = 512
    ecpm_hidden: int = 512
    dropout: float = 0.1
    max_length: int = 512
    batch_size: int = 64
    dtype: str = "bfloat16"

    # -- Data sources ---------------------------------------------------------
    mysql_host: str = "localhost"
    mysql_user: str = "adx"
    mysql_pass: str = "adx_pass"
    mysql_db: str = "adx"
    mysql_port: int = 3306

    # -- Redis ----------------------------------------------------------------
    redis_addr: str = "localhost:6379"
    redis_db: int = 0
    redis_ttl: int = 120  # seconds

    # -- Scheduling -----------------------------------------------------------
    score_interval: int = 30  # seconds between scoring cycles

    # -- Prompt template ------------------------------------------------------
    prompt_template: str = (
        "User context: tech enthusiast. Ad: {title}. {description} (category: {category})"
    )

    @classmethod
    def from_env_and_args(cls, args: Optional[argparse.Namespace] = None) -> "ScorerConfig":
        """Build config from env vars first, then override with CLI args."""
        cfg = cls()

        # Env var overrides
        cfg.model_path = os.getenv("MODEL_PATH", cfg.model_path)
        cfg.mysql_host = os.getenv("MYSQL_HOST", cfg.mysql_host)
        cfg.mysql_user = os.getenv("MYSQL_USER", cfg.mysql_user)
        cfg.mysql_pass = os.getenv("MYSQL_PASS", cfg.mysql_pass)
        cfg.mysql_db = os.getenv("MYSQL_DB", cfg.mysql_db)
        cfg.redis_addr = os.getenv("REDIS_ADDR", cfg.redis_addr)
        cfg.score_interval = int(os.getenv("SCORE_INTERVAL", str(cfg.score_interval)))

        # CLI arg overrides (highest priority)
        if args is not None:
            for key in (
                "model_path", "mysql_host", "mysql_user", "mysql_pass",
                "mysql_db", "redis_addr", "score_interval", "batch_size",
            ):
                val = getattr(args, key, None)
                if val is not None:
                    setattr(cfg, key, val)

        return cfg


# ---------------------------------------------------------------------------
# GPR Prediction Heads (same architecture as gpr/model/gpr_model.py)
# ---------------------------------------------------------------------------

class CTRHead(nn.Module):
    """CTR prediction: binary classification (click/no-click)."""

    def __init__(self, hidden_size: int, ctr_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ctr_hidden)
        self.fc2 = nn.Linear(ctr_hidden, ctr_hidden // 2)
        self.fc3 = nn.Linear(ctr_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)


class CVRHead(nn.Module):
    """CVR prediction: regression (conversion probability)."""

    def __init__(self, hidden_size: int, cvr_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, cvr_hidden)
        self.fc2 = nn.Linear(cvr_hidden, cvr_hidden // 2)
        self.fc3 = nn.Linear(cvr_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x)).squeeze(-1)


class ECPMHead(nn.Module):
    """eCPM scoring: ranking score for auction ordering."""

    def __init__(self, hidden_size: int, ecpm_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ecpm_hidden)
        self.fc2 = nn.Linear(ecpm_hidden, ecpm_hidden // 2)
        self.fc3 = nn.Linear(ecpm_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)


class GPRHeads(nn.Module):
    """
    Separate nn.Module wrapping all three GPR prediction heads.

    Takes a mean-pooled hidden state vector (batch, hidden_size) and
    returns (ctr_logits, cvr_probs, ecpm_scores).
    """

    def __init__(self, config: ScorerConfig):
        super().__init__()
        self.ctr_head = CTRHead(config.hidden_size, config.ctr_hidden, config.dropout)
        self.cvr_head = CVRHead(config.hidden_size, config.cvr_hidden, config.dropout)
        self.ecpm_head = ECPMHead(config.hidden_size, config.ecpm_hidden, config.dropout)

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled: (batch, hidden_size) mean-pooled representations.

        Returns:
            ctr_logits: (batch,) raw logits (apply sigmoid for probability).
            cvr_probs:  (batch,) already sigmoid'd probabilities.
            ecpm_scores: (batch,) ranking scores.
        """
        ctr_logits = self.ctr_head(pooled)
        cvr_probs = self.cvr_head(pooled)
        ecpm_scores = self.ecpm_head(pooled)
        return ctr_logits, cvr_probs, ecpm_scores


# ---------------------------------------------------------------------------
# Mean-pooling helper
# ---------------------------------------------------------------------------

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Global mean pooling over non-padding tokens (last hidden state)."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


# ---------------------------------------------------------------------------
# Batch Scorer
# ---------------------------------------------------------------------------

class BatchScorer:
    """Loads model, periodically fetches ads, scores, writes to Redis."""

    def __init__(self, config: ScorerConfig):
        self.config = config
        self.device = torch.device("cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.backbone: Optional[nn.Module] = None
        self.heads: Optional[GPRHeads] = None
        self.redis_client: Optional[redis.Redis] = None

    # -- Model loading -------------------------------------------------------

    def load_model(self) -> None:
        """Load Qwen2-1.5B backbone + GPR heads. Fatal on failure."""
        t0 = time.monotonic()

        # Determine dtype
        if self.config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.config.dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model_path = self.config.model_path
        logger.info("Loading tokenizer from %s ...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        assert self.tokenizer is not None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token = eos_token (%s)", self.tokenizer.pad_token)

        logger.info("Loading backbone from %s (dtype=%s) ...", model_path, self.config.dtype)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_path,
            config=hf_config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        assert self.backbone is not None
        self.backbone.eval()

        logger.info("Initialising GPR heads (hidden_size=%d) ...", self.config.hidden_size)
        self.heads = GPRHeads(self.config)
        self.heads.eval()

        elapsed = time.monotonic() - t0
        logger.info("GPR CPU scorer started, model loaded in %.1fs", elapsed)

    # -- Data-source connections ---------------------------------------------

    def _connect_redis(self) -> bool:
        """Establish Redis connection. Returns True on success."""
        try:
            host, _, port_str = self.config.redis_addr.partition(":")
            port = int(port_str) if port_str else 6379
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=self.config.redis_db,
                socket_connect_timeout=3,
                socket_timeout=5,
                decode_responses=False,
            )
            self.redis_client.ping()
            logger.info("Connected to Redis at %s", self.config.redis_addr)
            return True
        except Exception as exc:
            logger.warning("Redis connection failed: %s", exc)
            self.redis_client = None
            return False

    def _connect_mysql(self):
        """Return a fresh pymysql connection. Caller must close it."""
        return pymysql.connect(
            host=self.config.mysql_host,
            port=self.config.mysql_port,
            user=self.config.mysql_user,
            password=self.config.mysql_pass,
            database=self.config.mysql_db,
            charset="utf8mb4",
            connect_timeout=5,
            read_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )

    # -- Data fetching -------------------------------------------------------

    def _fetch_active_ads(self) -> List[Dict[str, Any]]:
        """Query MySQL for all active creatives."""
        conn = self._connect_mysql()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, campaign_id, title, description, category, tags "
                    "FROM creatives WHERE status = 'active'"
                )
                rows = list(cur.fetchall())
            logger.info("Fetched %d active creatives from MySQL", len(rows))
            return rows
        finally:
            conn.close()

    # -- Prompt building -----------------------------------------------------

    def _build_prompt(self, ad: Dict[str, Any]) -> str:
        """Build a scoring prompt for a single ad creative."""
        return self.config.prompt_template.format(
            title=ad.get("title", ""),
            description=ad.get("description", ""),
            category=ad.get("category", ""),
        )

    # -- Batch scoring -------------------------------------------------------

    @torch.no_grad()
    def _score_batch(
        self, ads: List[Dict[str, Any]]
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Score a list of ads in a single forward pass.

        Returns:
            dict[int, tuple[ctr, cvr, ecpm]] keyed by ad ID.
        """
        if not ads:
            return {}

        prompts = [self._build_prompt(ad) for ad in ads]
        ad_ids = [ad["id"] for ad in ads]

        t_start = time.monotonic()

        # Tokenize the whole batch
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # --- Mini-batch loop to avoid OOM on large ad inventories ------------
        n = len(ads)
        bsz = self.config.batch_size
        ctr_probs_list: List[torch.Tensor] = []
        cvr_probs_list: List[torch.Tensor] = []
        ecpm_scores_list: List[torch.Tensor] = []

        for i in range(0, n, bsz):
            j = min(i + bsz, n)
            ids_b = input_ids[i:j]
            mask_b = attention_mask[i:j]

            # Backbone forward pass
            outputs = self.backbone(
                input_ids=ids_b,
                attention_mask=mask_b,
                output_hidden_states=True,
            )
            hidden = outputs.last_hidden_state  # (B, L, hidden_size)

            # Mean pool
            pooled = mean_pool(hidden, mask_b)  # (B, hidden_size)

            # Forward through heads
            ctr_logits, cvr_probs, ecpm_scores = self.heads(pooled)

            ctr_probs_list.append(torch.sigmoid(ctr_logits))
            cvr_probs_list.append(cvr_probs)
            ecpm_scores_list.append(ecpm_scores)

        # Concatenate all mini-batches
        ctr_all = torch.cat(ctr_probs_list, dim=0)
        cvr_all = torch.cat(cvr_probs_list, dim=0)
        ecpm_all = torch.cat(ecpm_scores_list, dim=0)

        elapsed_ms = (time.monotonic() - t_start) * 1000
        avg_ms = elapsed_ms / n if n else 0
        logger.info(
            "Scored %d ads in %.1fms (avg %.2fms/ad)",
            n, elapsed_ms, avg_ms,
        )

        # Build result dict
        result: Dict[int, Tuple[float, float, float]] = {}
        for idx, ad_id in enumerate(ad_ids):
            try:
                ctr = float(ctr_all[idx].item())
                cvr = float(cvr_all[idx].item())
                ecpm = float(ecpm_all[idx].item())
                result[ad_id] = (ctr, cvr, ecpm)
            except Exception as exc:
                logger.warning("Failed to score ad %d: %s", ad_id, exc)

        return result

    # -- Redis writes --------------------------------------------------------

    def _write_scores(self, scores: Dict[int, Tuple[float, float, float]]) -> int:
        """
        Write scores to Redis as HSET + EXPIRE.

        Returns number of successfully written entries.
        """
        if not self.redis_client:
            logger.warning("Redis not connected, skipping score write")
            return 0

        if not scores:
            return 0

        ttl = self.config.redis_ttl
        written = 0

        try:
            pipe = self.redis_client.pipeline(transaction=False)
            for ad_id, (ctr, cvr, ecpm) in scores.items():
                key = f"gpr_score:{ad_id}"
                pipe.hset(key, mapping={"ctr": str(ctr), "cvr": str(cvr), "ecpm": str(ecpm)})
                pipe.expire(key, ttl)
            pipe.execute()
            written = len(scores)
        except Exception as exc:
            logger.warning("Redis write failed: %s", exc)
            return 0

        logger.info("Updated %d scores in Redis", written)
        return written

    # -- Main loop -----------------------------------------------------------

    def run_once(self) -> None:
        """Perform one full scoring cycle: fetch → score → write."""
        # Ensure Redis is connected (reconnect on transient failures)
        if self.redis_client is None:
            self._connect_redis()

        # Fetch active ads
        try:
            ads = self._fetch_active_ads()
        except Exception as exc:
            logger.warning("MySQL error: %s", exc)
            return

        if not ads:
            logger.info("No active ads to score")
            return

        # Score
        try:
            scores = self._score_batch(ads)
        except Exception as exc:
            logger.warning("Scoring error: %s", exc)
            return

        # Write
        self._write_scores(scores)

    def run_loop(self) -> None:
        """Run the scoring loop until SIGTERM/SIGINT."""
        global _RUNNING
        logger.info("Starting scoring loop (interval=%ds)", self.config.score_interval)

        while _RUNNING:
            cycle_start = time.monotonic()
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Unexpected error in scoring cycle: %s", exc)

            # Sleep, accounting for cycle time
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, self.config.score_interval - elapsed)
            if sleep_time > 0 and _RUNNING:
                time.sleep(sleep_time)

        logger.info("Scoring loop stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPR CPU Scorer — batch-scores active ads and writes to Redis",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Qwen/Qwen2-1.5B"),
        help="HF model name or local path (env: MODEL_PATH, default: Qwen/Qwen2-1.5B)",
    )
    parser.add_argument(
        "--mysql-host",
        default=os.getenv("MYSQL_HOST", "localhost"),
        help="MySQL host (env: MYSQL_HOST)",
    )
    parser.add_argument(
        "--mysql-user",
        default=os.getenv("MYSQL_USER", "adx"),
        help="MySQL user (env: MYSQL_USER)",
    )
    parser.add_argument(
        "--mysql-pass",
        default=os.getenv("MYSQL_PASS", "adx_pass"),
        help="MySQL password (env: MYSQL_PASS)",
    )
    parser.add_argument(
        "--mysql-db",
        default=os.getenv("MYSQL_DB", "adx"),
        help="MySQL database (env: MYSQL_DB)",
    )
    parser.add_argument(
        "--redis-addr",
        default=os.getenv("REDIS_ADDR", "localhost:6379"),
        help="Redis address host:port (env: REDIS_ADDR)",
    )
    parser.add_argument(
        "--score-interval",
        type=int,
        default=int(os.getenv("SCORE_INTERVAL", "30")),
        help="Seconds between scoring cycles (env: SCORE_INTERVAL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "64")),
        help="Mini-batch size for model forward pass",
    )
    parser.add_argument(
        "--dtype",
        default=os.getenv("DTYPE", "bfloat16"),
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scoring cycle and exit",
    )

    args = parser.parse_args()
    config = ScorerConfig.from_env_and_args(args)
    config.model_path = args.model_path
    config.dtype = args.dtype
    config.batch_size = args.batch_size

    scorer = BatchScorer(config)

    # Load model (fatal on failure)
    try:
        scorer.load_model()
    except Exception as exc:
        logger.fatal("Failed to load model: %s", exc)
        sys.exit(1)

    if args.once:
        scorer.run_once()
    else:
        scorer.run_loop()


if __name__ == "__main__":
    main()
