"""Unit tests for data/ab_report.py — all external calls are mocked."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from data.ab_report import ABReport, _parse_mysql_dsn, _query_clickhouse


# ---------------------------------------------------------------------------
# DSN parser tests
# ---------------------------------------------------------------------------


class TestParseMysqlDsn:
    def test_full_dsn(self):
        params = _parse_mysql_dsn("adx:adx_pass@localhost:3306/adx")
        assert params["user"] == "adx"
        assert params["password"] == "adx_pass"
        assert params["host"] == "localhost"
        assert params["port"] == 3306
        assert params["database"] == "adx"

    def test_minimal_dsn(self):
        params = _parse_mysql_dsn("root@localhost/adx")
        assert params["user"] == "root"
        assert params["password"] == ""
        assert params["host"] == "localhost"
        assert params["port"] == 3306
        assert params["database"] == "adx"

    def test_default_values(self):
        params = _parse_mysql_dsn("adx:adx_pass@db.example.com/adxdb")
        assert params["host"] == "db.example.com"
        assert params["port"] == 3306  # default
        assert params["database"] == "adxdb"


# ---------------------------------------------------------------------------
# calculate_lift tests
# ---------------------------------------------------------------------------


class TestCalculateLift:
    """Test lift calculation with known values — no mocking needed."""

    def test_ctr_lift_33_percent(self):
        reporter = ABReport()
        control = {
            "requests": 10000,
            "imps": 10000,
            "clicks": 300,  # CTR = 0.03
            "convs": 15,
            "cost": 500.0,
            "rev": 600.0,
            "avg_latency": 45.0,
        }
        treatment = {
            "requests": 10000,
            "imps": 10000,
            "clicks": 400,  # CTR = 0.04
            "convs": 24,
            "cost": 510.0,
            "rev": 640.0,
            "avg_latency": 48.0,
        }
        lift = reporter.calculate_lift(control, treatment)
        # 0.04 - 0.03 = 0.01, / 0.03 = 0.333... -> +33.33%
        assert round(lift["ctr_lift_pct"], 1) == 33.3
        assert lift["ctr_control"] == 0.03
        assert lift["ctr_treatment"] == 0.04

    def test_cvr_lift_25_percent(self):
        reporter = ABReport()
        control = {
            "requests": 5000,
            "imps": 5000,
            "clicks": 200,  # CTR irrelevant here
            "convs": 4,  # CVR = 0.02
            "cost": 200.0,
            "rev": 300.0,
            "avg_latency": 40.0,
        }
        treatment = {
            "requests": 5000,
            "imps": 5000,
            "clicks": 200,
            "convs": 5,  # CVR = 0.025
            "cost": 205.0,
            "rev": 310.0,
            "avg_latency": 42.0,
        }
        lift = reporter.calculate_lift(control, treatment)
        # 0.025 - 0.02 = 0.005, / 0.02 = 0.25 -> +25%
        assert round(lift["cvr_lift_pct"], 1) == 25.0
        assert lift["cvr_control"] == 0.02
        assert lift["cvr_treatment"] == 0.025

    def test_negative_latency_delta(self):
        reporter = ABReport()
        control = {"requests": 100, "imps": 100, "clicks": 10, "convs": 2,
                   "cost": 10, "rev": 12, "avg_latency": 60.0}
        treatment = {"requests": 100, "imps": 100, "clicks": 12, "convs": 3,
                     "cost": 11, "rev": 14, "avg_latency": 55.0}
        lift = reporter.calculate_lift(control, treatment)
        assert lift["latency_delta"] == -5.0

    def test_ecpm_lift(self):
        reporter = ABReport()
        control = {"requests": 1000, "imps": 1000, "clicks": 50, "convs": 5,
                   "cost": 100, "rev": 200, "avg_latency": 50}
        # control eCPM = 200 / 1000 * 1000 = 200
        treatment = {"requests": 1000, "imps": 1000, "clicks": 55, "convs": 6,
                     "cost": 105, "rev": 250, "avg_latency": 52}
        # treatment eCPM = 250 / 1000 * 1000 = 250
        # lift = (250 - 200) / 200 * 100 = 25%
        lift = reporter.calculate_lift(control, treatment)
        assert round(lift["ecpm_lift_pct"], 1) == 25.0

    def test_zero_imps_handled_safely(self):
        reporter = ABReport()
        control = {"requests": 0, "imps": 0, "clicks": 0, "convs": 0,
                   "cost": 0, "rev": 0, "avg_latency": 0}
        treatment = {"requests": 0, "imps": 0, "clicks": 0, "convs": 0,
                     "cost": 0, "rev": 0, "avg_latency": 0}
        lift = reporter.calculate_lift(control, treatment)
        assert lift["ctr_lift_pct"] == 0.0
        assert lift["cvr_lift_pct"] == 0.0
        assert lift["ecpm_lift_pct"] == 0.0


# ---------------------------------------------------------------------------
# get_experiment_metrics tests (mock ClickHouse)
# ---------------------------------------------------------------------------


class TestGetExperimentMetrics:
    def test_returns_per_variant_dict(self):
        """Simulate ClickHouse returning two rows (control + treatment)."""
        mock_response_data = json.dumps({
            "meta": [
                {"name": "variant"},
                {"name": "requests"},
                {"name": "imps"},
                {"name": "clicks"},
                {"name": "convs"},
                {"name": "cost"},
                {"name": "rev"},
                {"name": "avg_latency"},
            ],
            "data": [
                ["control", 1000, 1000, 30, 3, 50.0, 80.0, 45.5],
                ["treatment", 1000, 1000, 40, 5, 52.0, 98.0, 48.0],
            ],
        })
        fake_response = BytesIO(mock_response_data.encode("utf-8"))

        with patch("urllib.request.urlopen", return_value=fake_response):
            reporter = ABReport(clickhouse_url="http://localhost:8123")
            metrics = reporter.get_experiment_metrics(experiment_id=1)

        assert "control" in metrics
        assert "treatment" in metrics
        assert metrics["control"]["clicks"] == 30
        assert metrics["treatment"]["clicks"] == 40
        assert metrics["control"]["avg_latency"] == 45.5
        assert metrics["treatment"]["avg_latency"] == 48.0

    def test_empty_variant_treated_as_unknown(self):
        mock_response_data = json.dumps({
            "meta": [
                {"name": "variant"},
                {"name": "requests"},
                {"name": "imps"},
                {"name": "clicks"},
                {"name": "convs"},
                {"name": "cost"},
                {"name": "rev"},
                {"name": "avg_latency"},
            ],
            "data": [
                ["", 500, 500, 10, 2, 20.0, 30.0, 42.0],
            ],
        })
        fake_response = BytesIO(mock_response_data.encode("utf-8"))

        with patch("urllib.request.urlopen", return_value=fake_response):
            reporter = ABReport(clickhouse_url="http://localhost:8123")
            metrics = reporter.get_experiment_metrics(experiment_id=2)

        assert "unknown" in metrics
        assert metrics["unknown"]["clicks"] == 10


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_contains_expected_sections(self):
        """Verify key fields appear in the generated text report."""
        reporter = ABReport(
            clickhouse_url="http://fake:8123",
            mysql_dsn="adx:pass@localhost:3306/adx",
        )

        # mock ClickHouse metrics
        ch_data = json.dumps({
            "meta": [
                {"name": "variant"},
                {"name": "requests"},
                {"name": "imps"},
                {"name": "clicks"},
                {"name": "convs"},
                {"name": "cost"},
                {"name": "rev"},
                {"name": "avg_latency"},
            ],
            "data": [
                ["control", 5000, 5000, 150, 10, 200.0, 300.0, 45.0],
                ["treatment", 5000, 5000, 200, 16, 210.0, 350.0, 48.0],
            ],
        })
        ch_resp = BytesIO(ch_data.encode("utf-8"))

        # mock MySQL experiment info
        mysql_rows = [
            {
                "id": 1,
                "name": "GPR vs DeepFM Baseline",
                "traffic_ratio": "0.50",
                "variant": "control",
                "hash_salt": "exp-gpr-baseline-v1",
                "status": "running",
                "description": "Primary experiment",
                "started_at": "2025-01-01",
                "ended_at": None,
            }
        ]

        with patch("urllib.request.urlopen", return_value=ch_resp):
            with patch("data.ab_report._query_mysql", return_value=mysql_rows):
                report = reporter.generate_report(experiment_id=1)

        assert "A/B EXPERIMENT REPORT" in report
        assert "Experiment ID     : 1" in report
        assert "GPR vs DeepFM Baseline" in report
        assert "LIFT ANALYSIS" in report
        assert "VERDICT" in report
        assert "CTR" in report and "lift" in report
        assert "CVR" in report and "lift" in report
        assert "eCPM " in report
        assert "Impressions" in report
        assert "Clicks" in report
        assert "Conversions" in report
        assert "Cost ($)" in report
        assert "Revenue ($)" in report
        assert "Avg Latency (ms)" in report

    def test_report_verdict_significant(self):
        """Treatment clearly beats control → significant verdict."""
        reporter = ABReport(
            clickhouse_url="http://fake:8123",
            mysql_dsn="adx:pass@fake:3306/adx",
        )
        ch_data = json.dumps({
            "meta": [
                {"name": "variant"}, {"name": "requests"}, {"name": "imps"},
                {"name": "clicks"}, {"name": "convs"}, {"name": "cost"},
                {"name": "rev"}, {"name": "avg_latency"},
            ],
            "data": [
                ["control", 1000, 1000, 30, 2, 40.0, 50.0, 45.0],
                ["treatment", 1000, 1000, 50, 6, 42.0, 80.0, 48.0],
            ],
        })

        mysql_rows = [{
            "id": 1, "name": "Test Exp", "traffic_ratio": "0.50",
            "variant": "control", "hash_salt": "s1", "status": "running",
            "description": "test", "started_at": "2025-01-01", "ended_at": None,
        }]

        with patch("urllib.request.urlopen", return_value=BytesIO(ch_data.encode("utf-8"))):
            with patch("data.ab_report._query_mysql", return_value=mysql_rows):
                report = reporter.generate_report(experiment_id=1)

        assert "GPR significantly outperforms baseline" in report

    def test_report_no_control_data(self):
        """When no control data exists, report still generates gracefully."""
        reporter = ABReport(
            clickhouse_url="http://fake:8123",
            mysql_dsn="adx:pass@fake:3306/adx",
        )
        ch_data = json.dumps({
            "meta": [
                {"name": "variant"}, {"name": "requests"}, {"name": "imps"},
                {"name": "clicks"}, {"name": "convs"}, {"name": "cost"},
                {"name": "rev"}, {"name": "avg_latency"},
            ],
            "data": [],
        })
        mysql_rows = [{
            "id": 1, "name": "No Data Exp", "traffic_ratio": "0.50",
            "variant": "control", "hash_salt": "s1", "status": "running",
            "description": "test", "started_at": "2025-01-01", "ended_at": None,
        }]

        with patch("urllib.request.urlopen", return_value=BytesIO(ch_data.encode("utf-8"))):
            with patch("data.ab_report._query_mysql", return_value=mysql_rows):
                report = reporter.generate_report(experiment_id=1)

        # Should not crash; should contain header
        assert "A/B EXPERIMENT REPORT" in report
        assert "No Data Exp" in report


# ---------------------------------------------------------------------------
# list_experiments tests (mock MySQL)
# ---------------------------------------------------------------------------


class TestListExperiments:
    def test_returns_list_of_experiments(self):
        mock_rows = [
            {
                "id": 1, "name": "GPR vs DeepFM Baseline",
                "traffic_ratio": "0.50", "variant": "control",
                "hash_salt": "exp-gpr-baseline-v1", "status": "running",
                "description": "Primary experiment",
                "started_at": "2025-01-01", "ended_at": None,
            },
            {
                "id": 2, "name": "Creative Agent A/B",
                "traffic_ratio": "0.50", "variant": "treatment",
                "hash_salt": "exp-creative-agent-v1", "status": "completed",
                "description": "AI vs human creatives",
                "started_at": "2025-02-01", "ended_at": "2025-03-01",
            },
        ]

        with patch("data.ab_report._query_mysql", return_value=mock_rows):
            reporter = ABReport(mysql_dsn="adx:pass@fake:3306/adx")
            exps = reporter.list_experiments()

        assert len(exps) == 2
        assert exps[0]["id"] == 1
        assert exps[0]["name"] == "GPR vs DeepFM Baseline"
        assert exps[1]["id"] == 2
        assert exps[1]["status"] == "completed"

    def test_raises_without_mysql_dsn(self):
        reporter = ABReport()  # no mysql_dsn
        try:
            reporter.list_experiments()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as exc:
            assert "mysql_dsn" in str(exc)


# ---------------------------------------------------------------------------
# verdict logic tests
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_significant_verdict(self):
        lift = {
            "ctr_lift_pct": 30.0,
            "cvr_lift_pct": 25.0,
            "ecpm_lift_pct": 15.0,
            "latency_delta": 3.0,
        }
        verdict = ABReport._compute_verdict(lift)
        assert "significantly outperforms" in verdict

    def test_no_significant_difference(self):
        lift = {
            "ctr_lift_pct": -2.0,
            "cvr_lift_pct": -1.0,
            "ecpm_lift_pct": -3.0,
            "latency_delta": -5.0,
        }
        verdict = ABReport._compute_verdict(lift)
        assert verdict == "No significant difference"

    def test_moderate_improvement(self):
        lift = {
            "ctr_lift_pct": 5.0,   # positive
            "cvr_lift_pct": -2.0,  # negative
            "ecpm_lift_pct": -1.0, # negative
            "latency_delta": 0.5,
        }
        verdict = ABReport._compute_verdict(lift)
        assert "moderate improvement" in verdict
