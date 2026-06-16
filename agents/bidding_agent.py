"""
Bidding Optimization Agent — LangChain agent that optimizes campaign bids
using Epsilon-Greedy MAB based on ROAS metrics from ClickHouse.

Runs hourly as a side-path task, never in the RTB hot path.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from .mab import EpsilonGreedyMAB


class BiddingAgent:
    """LangChain-based bidding optimization agent.

    Queries ClickHouse for hourly campaign metrics, uses MAB to select
    optimal bid multipliers based on ROAS, generates LLM explanations,
    and updates Redis with new bid settings.

    Args:
        llm_endpoint: vLLM OpenAI-compatible chat endpoint URL.
        clickhouse_url: ClickHouse HTTP interface base URL.
        redis_host: Redis host address.
        redis_port: Redis port.
        epsilon: MAB exploration rate (0.0 = pure exploitation).
    """

    DEFAULT_CLICKHOUSE_DB = "adx_analytics"

    METRICS_QUERY = (
        "SELECT campaign_id, sum(impressions), sum(clicks), "
        "sum(conversions), sum(cost), sum(revenue) "
        "FROM ad_metrics "
        "WHERE timestamp > now() - INTERVAL 1 HOUR "
        "GROUP BY campaign_id"
    )

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:8000/v1",
        clickhouse_url: str = "http://localhost:8123",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        epsilon: float = 0.1,
    ):
        self.llm_endpoint = llm_endpoint.rstrip("/")
        self.clickhouse_url = clickhouse_url.rstrip("/")
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.mab = EpsilonGreedyMAB(epsilon=epsilon)
        self._redis = None

    def _get_redis(self):
        """Lazy-init Redis connection."""
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
            )
        return self._redis

    def _query_clickhouse(self, sql: str) -> List[Dict[str, str]]:
        """Execute SQL query against ClickHouse HTTP interface.

        ClickHouse returns TSV by default. We request JSONEachRow format.

        Args:
            sql: SQL query string.

        Returns:
            List of row dicts keyed by column name.
        """
        url = f"{self.clickhouse_url}/"
        params: Dict[str, Any] = {
            "query": sql,
            "database": self.DEFAULT_CLICKHOUSE_DB,
        }
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"ClickHouse query failed: {e}")

        text = resp.text.strip()
        if not text:
            return []

        rows = []
        lines = text.split("\n")
        if not lines:
            return []

        headers = lines[0].split("\t")
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split("\t")
            row = {}
            for i, header in enumerate(headers):
                row[header] = values[i] if i < len(values) else ""
            rows.append(row)
        return rows

    def _update_redis_bid(self, campaign_id: str, multiplier: float) -> None:
        """Set bid multiplier for a campaign in Redis.

        Key pattern: bid_multiplier:{campaign_id}
        """
        r = self._get_redis()
        key = f"bid_multiplier:{campaign_id}"
        r.set(key, str(multiplier))

    def _update_redis_budget(self, campaign_id: str, budget: float) -> None:
        """Set budget for a campaign in Redis.

        Key pattern: budget:{campaign_id}
        """
        r = self._get_redis()
        key = f"budget:{campaign_id}"
        r.set(key, str(budget))

    def _calculate_roas(
        self, row: Dict[str, str]
    ) -> Tuple[float, float, float, float, float]:
        """Extract metrics from ClickHouse row.

        Returns:
            Tuple of (cost, revenue, impressions, clicks, conversions).
        """
        cost = float(row.get("sum(cost)", 0) or 0)
        revenue = float(row.get("sum(revenue)", 0) or 0)
        impressions = float(row.get("sum(impressions)", 0) or 0)
        clicks = float(row.get("sum(clicks)", 0) or 0)
        conversions = float(row.get("sum(conversions)", 0) or 0)
        return cost, revenue, impressions, clicks, conversions

    def _adjust_budget(
        self,
        campaign_id: str,
        current_budget: float,
        roas: float,
        target: float,
    ) -> float:
        """Calculate new budget based on ROAS vs target.

        If ROAS > target: increase budget by 10% (campaign performing well).
        If ROAS < target: decrease budget by 10% (campaign underperforming).
        If cost is zero or ROAS is zero: keep budget unchanged.

        Args:
            campaign_id: Campaign identifier.
            current_budget: Current budget amount.
            roas: Current ROAS (revenue / cost).
            target: Target ROAS.

        Returns:
            Adjusted budget amount (never below 0).
        """
        if current_budget <= 0 or roas <= 0:
            return current_budget

        if roas > target:
            new_budget = current_budget * 1.10
        elif roas < target:
            new_budget = current_budget * 0.90
        else:
            new_budget = current_budget

        return max(new_budget, 0.0)

    def _get_current_budget(self, campaign_id: str) -> float:
        """Get current campaign budget from Redis.

        Returns 0.0 if not set.
        """
        r = self._get_redis()
        key = f"budget:{campaign_id}"
        val = r.get(key)
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _explain_decision(
        self,
        campaign_id: str,
        metrics: Dict[str, Any],
        action: Dict[str, Any],
    ) -> str:
        """Generate natural language explanation using LLM.

        Args:
            campaign_id: Campaign identifier.
            metrics: Campaign metrics dict (roas, cost, revenue, etc.).
            action: Action taken dict (bid_multiplier, budget_change, etc.).

        Returns:
            Natural language explanation string.
        """
        prompt = self._build_explanation_prompt(campaign_id, metrics, action)
        explanation = self._call_llm(prompt)
        return explanation if explanation else self._fallback_explanation(campaign_id, action)

    def _build_explanation_prompt(
        self,
        campaign_id: str,
        metrics: Dict[str, Any],
        action: Dict[str, Any],
    ) -> str:
        """Build the LLM prompt for decision explanation."""
        roas = metrics.get("roas", 0)
        cost = metrics.get("cost", 0)
        revenue = metrics.get("revenue", 0)
        impressions = metrics.get("impressions", 0)
        conversions = metrics.get("conversions", 0)
        multiplier = action.get("bid_multiplier", 1.0)
        budget_change = action.get("budget_change", "unchanged")
        arm_stats = action.get("arm_stats", {})

        return (
            f"You are a bidding optimization analyst for an ad exchange. "
            f"Explain the following bidding decision in 2-3 concise sentences:\n\n"
            f"Campaign: {campaign_id}\n"
            f"Metrics (last hour): ROAS={roas:.2f}, Cost=${cost:.2f}, "
            f"Revenue=${revenue:.2f}, Impressions={int(impressions)}, "
            f"Conversions={int(conversions)}\n"
            f"Decision: Set bid multiplier to {multiplier}x, "
            f"Budget change: {budget_change}\n"
            f"MAB arm performance: {json.dumps(arm_stats)}\n\n"
            f"Explain the reasoning for this bid adjustment."
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call vLLM chat completions endpoint.

        Returns:
            Response text or None on failure.
        """
        url = f"{self.llm_endpoint}/chat/completions"
        payload = {
            "model": "gpr-adx-agent",
            "messages": [
                {"role": "system", "content": "You are a concise bidding analyst."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError) as e:
            return None

    def _fallback_explanation(self, campaign_id: str, action: Dict[str, Any]) -> str:
        """Generate a template-based explanation when LLM is unavailable."""
        multiplier = action.get("bid_multiplier", 1.0)
        budget_change = action.get("budget_change", "unchanged")
        return (
            f"Campaign {campaign_id}: bid multiplier adjusted to {multiplier}x "
            f"based on recent ROAS performance. Budget {budget_change}."
        )

    def optimize_campaigns(
        self, roas_target: float = 3.0, dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Run one optimization cycle for all active campaigns.

        Queries ClickHouse for hourly metrics, selects bid multipliers via
        MAB based on ROAS, generates LLM explanations, and updates Redis.

        Args:
            roas_target: Target ROAS (revenue/cost ratio).
            dry_run: If True, skip Redis writes and LLM calls.

        Returns:
            List of decision records, one per campaign.
        """
        rows = self._query_clickhouse(self.METRICS_QUERY)
        results: List[Dict[str, Any]] = []

        for row in rows:
            campaign_id = row.get("campaign_id", "")
            if not campaign_id:
                continue

            cost, revenue, impressions, clicks, conversions = self._calculate_roas(row)
            roas = revenue / cost if cost > 0 else 0.0

            # MAB selects bid multiplier based on ROAS reward
            multiplier = self.mab.select_arm(campaign_id)
            self.mab.update(campaign_id, multiplier, roas)

            # Budget adjustment
            current_budget = self._get_current_budget(campaign_id) if not dry_run else 1000.0
            new_budget = self._adjust_budget(campaign_id, current_budget, roas, roas_target)

            if roas > roas_target:
                budget_change = "increased"
            elif roas < roas_target:
                budget_change = "decreased"
            else:
                budget_change = "unchanged"

            metrics = {
                "roas": roas,
                "cost": cost,
                "revenue": revenue,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
            }
            action = {
                "bid_multiplier": multiplier,
                "new_budget": new_budget,
                "budget_change": budget_change,
                "arm_stats": self.mab.get_stats(campaign_id),
            }

            if dry_run:
                explanation = self._fallback_explanation(campaign_id, action)
            else:
                explanation = self._explain_decision(campaign_id, metrics, action)

            if not dry_run:
                self._update_redis_bid(campaign_id, multiplier)
                self._update_redis_budget(campaign_id, new_budget)

            results.append({
                "campaign_id": campaign_id,
                "metrics": metrics,
                "action": action,
                "explanation": explanation,
            })

        return results


def main():
    """CLI entry point for the bidding agent.

    Usage:
        python -m agents.bidding_agent --roas-target 3.0 --epsilon 0.1 --dry-run
    """
    parser = argparse.ArgumentParser(description="GPR ADX Bidding Optimization Agent")
    parser.add_argument(
        "--roas-target", type=float, default=3.0,
        help="Target ROAS (revenue/cost ratio, default: 3.0)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="MAB exploration rate (default: 0.1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without modifying Redis or calling LLM",
    )
    parser.add_argument(
        "--llm-endpoint", type=str,
        default=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1"),
        help="vLLM endpoint URL",
    )
    parser.add_argument(
        "--clickhouse-url", type=str,
        default=os.environ.get("CLICKHOUSE_URL", "http://localhost:8123"),
        help="ClickHouse HTTP URL",
    )
    parser.add_argument(
        "--redis-host", type=str,
        default=os.environ.get("REDIS_HOST", "localhost"),
        help="Redis host",
    )
    parser.add_argument(
        "--redis-port", type=int,
        default=int(os.environ.get("REDIS_PORT", "6379")),
        help="Redis port",
    )

    args = parser.parse_args()

    agent = BiddingAgent(
        llm_endpoint=args.llm_endpoint,
        clickhouse_url=args.clickhouse_url,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        epsilon=args.epsilon,
    )

    results = agent.optimize_campaigns(
        roas_target=args.roas_target,
        dry_run=args.dry_run,
    )

    if results:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("No active campaigns found in the last hour.")


if __name__ == "__main__":
    main()
