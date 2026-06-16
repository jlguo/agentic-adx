import math
import random
from unittest.mock import MagicMock, patch

import pytest

from agents.bidding_agent import BiddingAgent
from agents.mab import EpsilonGreedyMAB


# ============================================================
# EpsilonGreedyMAB tests
# ============================================================


class TestEpsilonGreedyMAB:
    """Tests for the Epsilon-Greedy Multi-Armed Bandit."""

    def test_init_default_arms(self):
        mab = EpsilonGreedyMAB()
        assert mab.arms == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        assert mab.epsilon == 0.1

    def test_init_custom_epsilon_and_arms(self):
        mab = EpsilonGreedyMAB(epsilon=0.2, arms=[0.8, 1.0, 1.2])
        assert mab.epsilon == 0.2
        assert mab.arms == [0.8, 1.0, 1.2]

    def test_init_rejects_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedyMAB(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedyMAB(epsilon=1.5)

    def test_select_arm_exploration_with_seed(self):
        """Epsilon=1.0 forces pure exploration (random)."""
        mab = EpsilonGreedyMAB(epsilon=1.0)
        random.seed(42)
        arm = mab.select_arm("camp_1")
        assert arm in mab.arms

    def test_select_arm_exploitation_with_seed(self):
        """Epsilon=0.0 forces pure exploitation (best known arm)."""
        mab = EpsilonGreedyMAB(epsilon=0.0)
        # Give arm 1.5 a high reward
        mab.update("camp_1", 1.5, 5.0)
        mab.update("camp_1", 1.0, 1.0)
        mab.update("camp_1", 0.5, 0.5)
        arm = mab.select_arm("camp_1")
        assert arm == 1.5  # Should pick best arm

    def test_select_arm_returns_deterministic_with_fixed_seed(self):
        """With epsilon=0.5 and fixed seed, selection is reproducible."""
        mab = EpsilonGreedyMAB(epsilon=0.5)
        random.seed(123)
        results = [mab.select_arm("camp_x") for _ in range(5)]
        random.seed(123)
        results2 = [mab.select_arm("camp_x") for _ in range(5)]
        # Same seed + fresh campaign = same random sequence
        # But second batch hits initialized stats, so compare first call only
        mab2 = EpsilonGreedyMAB(epsilon=0.5)
        random.seed(123)
        results3 = [mab2.select_arm("camp_y") for _ in range(5)]
        assert results == results3

    def test_update_incremental_mean(self):
        mab = EpsilonGreedyMAB(epsilon=0.1)
        mab.update("camp_1", 1.0, 2.0)
        mab.update("camp_1", 1.0, 4.0)
        stats = mab.get_stats("camp_1")
        assert stats[1.0]["count"] == 2
        assert stats[1.0]["mean"] == pytest.approx(3.0)

    def test_update_unknown_arm_raises(self):
        mab = EpsilonGreedyMAB(epsilon=0.1)
        with pytest.raises(ValueError, match="Unknown arm"):
            mab.update("camp_1", 3.0, 1.0)

    def test_convergence_higher_reward_arm_selected_more(self):
        """Over many updates, arm with higher reward gets selected more (epsilon=0.1)."""
        mab = EpsilonGreedyMAB(epsilon=0.1, arms=[0.5, 1.0, 2.0])
        random.seed(99)

        # Arm 2.0 has consistently higher ROAS
        for _ in range(200):
            arm = mab.select_arm("camp_cv")
            if arm == 2.0:
                mab.update("camp_cv", arm, random.uniform(3.5, 5.0))
            else:
                mab.update("camp_cv", arm, random.uniform(0.5, 1.5))

        stats = mab.get_stats("camp_cv")
        best_arm_mean = stats[2.0]["mean"]
        worst_arm_mean = stats[0.5]["mean"]
        assert best_arm_mean > worst_arm_mean

        # Verify best arm is chosen in exploitation
        mab2 = EpsilonGreedyMAB(epsilon=0.0, arms=[0.5, 1.0, 2.0])
        mab2._stats = mab._stats  # Copy stats
        best_selected = mab2.select_arm("camp_cv")
        assert best_selected == 2.0

    def test_get_stats_returns_copy(self):
        mab = EpsilonGreedyMAB()
        mab.update("camp_1", 1.0, 3.0)
        stats = mab.get_stats("camp_1")
        stats[1.0]["mean"] = 999.0  # Mutate the copy
        # Original should be unchanged
        assert mab.get_stats("camp_1")[1.0]["mean"] == pytest.approx(3.0)

    def test_get_stats_unknown_campaign(self):
        mab = EpsilonGreedyMAB()
        assert mab.get_stats("nonexistent") == {}

    def test_campaign_count(self):
        mab = EpsilonGreedyMAB()
        assert mab.campaign_count == 0
        mab.select_arm("camp_a")
        assert mab.campaign_count == 1
        mab.select_arm("camp_b")
        assert mab.campaign_count == 2

    def test_multiple_campaigns_independent_stats(self):
        mab = EpsilonGreedyMAB(epsilon=0.1)
        mab.update("camp_a", 1.0, 5.0)
        mab.update("camp_b", 1.0, 1.0)
        assert mab.get_stats("camp_a")[1.0]["mean"] == pytest.approx(5.0)
        assert mab.get_stats("camp_b")[1.0]["mean"] == pytest.approx(1.0)


# ============================================================
# BiddingAgent tests
# ============================================================


class TestBiddingAgent:
    """Tests for the BiddingAgent."""

    @pytest.fixture
    def agent(self):
        """Create BiddingAgent with epsilon=0 for deterministic selection."""
        return BiddingAgent(
            llm_endpoint="http://localhost:8000/v1",
            clickhouse_url="http://localhost:8123",
            redis_host="localhost",
            redis_port=6379,
            epsilon=0.0,
        )

    # --- Budget adjustment tests ---

    def test_adjust_budget_increase_when_roas_above_target(self, agent):
        new_budget = agent._adjust_budget("c1", 1000.0, roas=5.0, target=3.0)
        assert new_budget == pytest.approx(1100.0)

    def test_adjust_budget_decrease_when_roas_below_target(self, agent):
        new_budget = agent._adjust_budget("c1", 1000.0, roas=1.0, target=3.0)
        assert new_budget == pytest.approx(900.0)

    def test_adjust_budget_unchanged_when_roas_equals_target(self, agent):
        new_budget = agent._adjust_budget("c1", 1000.0, roas=3.0, target=3.0)
        assert new_budget == pytest.approx(1000.0)

    def test_adjust_budget_zero_cost_keeps_budget(self, agent):
        new_budget = agent._adjust_budget("c1", 1000.0, roas=0.0, target=3.0)
        assert new_budget == pytest.approx(1000.0)

    def test_adjust_budget_non_negative(self, agent):
        new_budget = agent._adjust_budget("c1", 10.0, roas=0.5, target=3.0)
        assert new_budget >= 0.0

    def test_adjust_budget_zero_budget_stays_zero(self, agent):
        new_budget = agent._adjust_budget("c1", 0.0, roas=5.0, target=3.0)
        assert new_budget == 0.0

    # --- ROAS calculation tests ---

    def test_calculate_roas_normal(self, agent):
        row = {
            "sum(cost)": "100",
            "sum(revenue)": "350",
            "sum(impressions)": "1000",
            "sum(clicks)": "50",
            "sum(conversions)": "10",
        }
        cost, revenue, impressions, clicks, conversions = agent._calculate_roas(row)
        assert cost == 100.0
        assert revenue == 350.0
        assert impressions == 1000.0
        assert clicks == 50.0
        assert conversions == 10.0

    def test_calculate_roas_missing_fields_default_zero(self, agent):
        row = {"campaign_id": "c1"}
        cost, revenue, impressions, clicks, conversions = agent._calculate_roas(row)
        assert cost == 0.0
        assert revenue == 0.0

    # --- ClickHouse query tests ---

    def test_query_clickhouse_parses_tsv(self, agent):
        tsv_response = (
            "campaign_id\tsum(impressions)\tsum(clicks)\tsum(conversions)\tsum(cost)\tsum(revenue)\n"
            "camp_1\t1000\t50\t10\t100\t350\n"
            "camp_2\t500\t20\t5\t50\t200"
        )
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = tsv_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            rows = agent._query_clickhouse("SELECT ...")
            assert len(rows) == 2
            assert rows[0]["campaign_id"] == "camp_1"
            assert rows[0]["sum(revenue)"] == "350"
            assert rows[1]["campaign_id"] == "camp_2"

    def test_query_clickhouse_empty_response(self, agent):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = ""
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            rows = agent._query_clickhouse("SELECT ...")
            assert rows == []

    def test_query_clickhouse_http_error(self, agent):
        import requests as req_lib
        with patch("requests.get") as mock_get:
            mock_get.side_effect = req_lib.ConnectionError("no route to host")
            with pytest.raises(RuntimeError, match="ClickHouse query failed"):
                agent._query_clickhouse("SELECT ...")

    # --- LLM explanation tests ---

    def test_explain_decision_llm_success(self, agent):
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = "Campaign camp_x shows strong ROAS of 5.0, increasing bid multiplier to 1.5x."
            explanation = agent._explain_decision(
                "camp_x",
                {"roas": 5.0, "cost": 100, "revenue": 500, "impressions": 1000, "conversions": 10},
                {"bid_multiplier": 1.5, "budget_change": "increased", "arm_stats": {}},
            )
            assert "camp_x" in explanation
            assert "1.5" in explanation

    def test_explain_decision_llm_failure_fallback(self, agent):
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = None
            explanation = agent._explain_decision(
                "camp_x",
                {"roas": 5.0, "cost": 100, "revenue": 500},
                {"bid_multiplier": 1.5, "budget_change": "increased"},
            )
            assert "camp_x" in explanation
            assert "1.5" in explanation

    def test_fallback_explanation(self, agent):
        explanation = agent._fallback_explanation(
            "camp_99",
            {"bid_multiplier": 0.75, "budget_change": "decreased"},
        )
        assert "camp_99" in explanation
        assert "0.75" in explanation
        assert "decreased" in explanation

    # --- optimize_campaigns integration test (dry_run, fully mocked) ---

    def test_optimize_campaigns_dry_run(self, agent):
        mock_rows = [
            {
                "campaign_id": "camp_a",
                "sum(impressions)": "2000",
                "sum(clicks)": "100",
                "sum(conversions)": "20",
                "sum(cost)": "200",
                "sum(revenue)": "800",
            },
            {
                "campaign_id": "camp_b",
                "sum(impressions)": "500",
                "sum(clicks)": "10",
                "sum(conversions)": "1",
                "sum(cost)": "100",
                "sum(revenue)": "50",
            },
        ]

        with patch.object(agent, "_query_clickhouse", return_value=mock_rows):
            results = agent.optimize_campaigns(roas_target=3.0, dry_run=True)

        assert len(results) == 2

        # camp_a: ROAS = 800/200 = 4.0 > 3.0, budget should increase
        camp_a = next(r for r in results if r["campaign_id"] == "camp_a")
        assert camp_a["metrics"]["roas"] == pytest.approx(4.0)
        assert camp_a["action"]["budget_change"] == "increased"
        assert camp_a["action"]["new_budget"] == pytest.approx(1100.0)
        assert camp_a["action"]["bid_multiplier"] in agent.mab.arms

        # camp_b: ROAS = 50/100 = 0.5 < 3.0, budget should decrease
        camp_b = next(r for r in results if r["campaign_id"] == "camp_b")
        assert camp_b["metrics"]["roas"] == pytest.approx(0.5)
        assert camp_b["action"]["budget_change"] == "decreased"
        assert camp_b["action"]["new_budget"] == pytest.approx(900.0)

        # Explanation should use fallback in dry_run
        assert "camp_a" in camp_a["explanation"]
        assert "camp_b" in camp_b["explanation"]

    def test_optimize_campaigns_roas_equals_target(self, agent):
        mock_rows = [
            {
                "campaign_id": "camp_c",
                "sum(impressions)": "1000",
                "sum(clicks)": "30",
                "sum(conversions)": "5",
                "sum(cost)": "100",
                "sum(revenue)": "300",
            },
        ]

        with patch.object(agent, "_query_clickhouse", return_value=mock_rows):
            results = agent.optimize_campaigns(roas_target=3.0, dry_run=True)

        camp_c = results[0]
        assert camp_c["metrics"]["roas"] == pytest.approx(3.0)
        assert camp_c["action"]["budget_change"] == "unchanged"
        assert camp_c["action"]["new_budget"] == pytest.approx(1000.0)

    def test_optimize_campaigns_empty_clickhouse(self, agent):
        with patch.object(agent, "_query_clickhouse", return_value=[]):
            results = agent.optimize_campaigns(roas_target=3.0, dry_run=True)
        assert results == []

    def test_optimize_campaigns_skips_empty_campaign_id(self, agent):
        mock_rows = [
            {"campaign_id": "", "sum(cost)": "10", "sum(revenue)": "20"},
        ]
        with patch.object(agent, "_query_clickhouse", return_value=mock_rows):
            results = agent.optimize_campaigns(roas_target=3.0, dry_run=True)
        assert results == []

    # --- Non-dry-run Redis writes ---

    def test_optimize_campaigns_writes_to_redis(self, agent):
        mock_rows = [
            {
                "campaign_id": "camp_r",
                "sum(impressions)": "1000",
                "sum(clicks)": "50",
                "sum(conversions)": "10",
                "sum(cost)": "100",
                "sum(revenue)": "500",
            },
        ]

        mock_redis = MagicMock()
        mock_redis.get.return_value = "500.0"

        with patch.object(agent, "_query_clickhouse", return_value=mock_rows), \
             patch.object(agent, "_get_redis", return_value=mock_redis), \
             patch.object(agent, "_call_llm", return_value="Test explanation."):
            results = agent.optimize_campaigns(roas_target=3.0, dry_run=False)

        assert len(results) == 1
        # Verify Redis writes
        set_calls = {call[0][0] for call in mock_redis.set.call_args_list}
        assert "bid_multiplier:camp_r" in set_calls
        assert "budget:camp_r" in set_calls

    # --- LLM call tests ---

    def test_call_llm_success(self, agent):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Test response."}}]
            }
            mock_post.return_value = mock_resp

            result = agent._call_llm("Test prompt")
            assert result == "Test response."
            mock_post.assert_called_once()
            call_args = mock_post.call_args[0][0]
            assert call_args.endswith("/chat/completions")

    def test_call_llm_network_error(self, agent):
        import requests as req_lib
        with patch("requests.post") as mock_post:
            mock_post.side_effect = req_lib.Timeout("timeout")
            result = agent._call_llm("Test prompt")
            assert result is None

    def test_call_llm_bad_response(self, agent):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"invalid": "structure"}
            mock_post.return_value = mock_resp

            result = agent._call_llm("Test prompt")
            assert result is None
