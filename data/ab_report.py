"""
A/B experiment report generator for GPR ADX.

Queries ClickHouse for per-variant metrics and compares control (baseline)
vs treatment (GPR) groups, computing lift percentages and significance indicators.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from typing import Any, Optional


# ---------------------------------------------------------------------------
# ClickHouse HTTP query helper
# ---------------------------------------------------------------------------

def _query_clickhouse(ch_url: str, sql: str) -> list[dict[str, Any]]:
    """Send a SQL query to ClickHouse HTTP interface and return rows as dicts."""
    query_url = f"{ch_url.rstrip('/')}/?query={urllib.parse.quote(sql)}"
    req = urllib.request.Request(query_url)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # ClickHouse JSONCompact returns [meta, data, ...]
    # Try to normalise into list-of-dicts
    if isinstance(data, dict) and "data" in data:
        rows = data["data"]
        meta = data.get("meta", [])
        col_names = [m["name"] for m in meta]
        return [dict(zip(col_names, row)) for row in rows]
    if isinstance(data, list):
        # JSONCompact format: [meta, [row, ...], stats]
        if len(data) >= 2 and isinstance(data[0], list) and isinstance(data[1], list):
            meta = data[0]
            col_names = [m["name"] for m in meta]
            return [dict(zip(col_names, row)) for row in data[1]]
    raise RuntimeError(f"Unexpected ClickHouse response format: {type(data)}")


# ---------------------------------------------------------------------------
# MySQL DSN parser / connection helper
# ---------------------------------------------------------------------------

_MYSQL_DSN_RE = re.compile(
    r"^(?:(?P<user>[^:@]+)(?::(?P<pass>[^@]*))?@)?"
    r"(?:tcp\()?"  # optional Go-style tcp() wrapper
    r"(?P<host>[^:/)]+)(?::(?P<port>\d+))?"
    r"(?:\))?"     # close tcp() wrapper
    r"(?:/(?P<db>[^?]+))?$"
)


def _parse_mysql_dsn(dsn: str) -> dict:
    """Parse 'user:pass@host:port/dbname' into a dict."""
    # Strip query parameters (e.g., ?parseTime=true) for parsing
    dsn_clean = dsn.split("?")[0]
    m = _MYSQL_DSN_RE.match(dsn_clean)
    if not m:
        raise ValueError(f"Invalid MySQL DSN: {dsn}")
    return {
        "user": m.group("user") or "root",
        "password": m.group("pass") or "",
        "host": m.group("host") or "localhost",
        "port": int(m.group("port") or 3306),
        "database": m.group("db") or "",
    }


def _query_mysql(dsn: str, sql: str) -> list[dict[str, Any]]:
    """Execute a SQL query against MySQL and return rows as dicts."""
    import pymysql

    params = _parse_mysql_dsn(dsn)
    conn = pymysql.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database=params["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return list(rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ABReport
# ---------------------------------------------------------------------------


class ABReport:
    """Generate A/B experiment comparison reports from ClickHouse analytics."""

    def __init__(
        self,
        clickhouse_url: str = "http://localhost:8123",
        mysql_dsn: Optional[str] = None,
    ):
        self.clickhouse_url = clickhouse_url
        self.mysql_dsn = mysql_dsn

    # -- data retrieval ---------------------------------------------------

    def get_experiment_metrics(self, experiment_id: int) -> dict:
        """Query ClickHouse for per-variant metrics of an experiment.

        Returns a dict keyed by variant name, e.g.
        {"control": {...}, "treatment": {...}}
        """
        # Use adx_analytics database via query parameter
        ch_url = self.clickhouse_url.rstrip("/") + "/?database=adx_analytics"
        sql = (
            "SELECT "
            "  variant, "
            "  count() AS requests, "
            "  sum(impressions) AS imps, "
            "  sum(clicks) AS clicks, "
            "  sum(conversions) AS convs, "
            "  sum(cost) AS cost, "
            "  sum(revenue) AS rev, "
            "  avg(latency_ms) AS avg_latency "
            "FROM ad_metrics "
            f"WHERE experiment_id = {experiment_id} "
            "GROUP BY variant"
        )
        rows = _query_clickhouse(ch_url, sql)
        result: dict = {}
        for row in rows:
            variant = row.get("variant", "unknown")
            if variant == "":
                variant = "unknown"
            result[variant] = {
                "requests": int(row.get("requests", 0)),
                "imps": int(row.get("imps", 0)),
                "clicks": int(row.get("clicks", 0)),
                "convs": int(row.get("convs", 0)),
                "cost": float(row.get("cost", 0)),
                "rev": float(row.get("rev", 0)),
                "avg_latency": float(row.get("avg_latency", 0)),
            }
        return result

    def list_experiments(self) -> list[dict]:
        """Query MySQL for all experiments."""
        if not self.mysql_dsn:
            raise RuntimeError("mysql_dsn is required for list_experiments")
        sql = (
            "SELECT id, name, traffic_ratio, variant, hash_salt, "
            "status, description, started_at, ended_at "
            "FROM experiments ORDER BY id"
        )
        return _query_mysql(self.mysql_dsn, sql)

    # -- lift calculation -------------------------------------------------

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Return numerator/denominator or 0.0 if denominator is 0."""
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def calculate_lift(
        self, control: dict, treatment: dict
    ) -> dict:
        """Compute lift percentages between control and treatment metrics.

        Args:
            control: metrics dict with keys imps, clicks, convs, rev, avg_latency
            treatment: same shape as control

        Returns a dict with CTR, CVR, eCPM lifts and latency delta.
        """
        # derived rates
        c_ctr = self._safe_ratio(control.get("clicks", 0), control.get("imps", 0))
        t_ctr = self._safe_ratio(treatment.get("clicks", 0), treatment.get("imps", 0))
        c_cvr = self._safe_ratio(control.get("convs", 0), control.get("clicks", 0))
        t_cvr = self._safe_ratio(treatment.get("convs", 0), treatment.get("clicks", 0))
        c_ecpm = self._safe_ratio(control.get("rev", 0), control.get("imps", 0)) * 1000
        t_ecpm = self._safe_ratio(treatment.get("rev", 0), treatment.get("imps", 0)) * 1000

        c_latency = control.get("avg_latency", 0)
        t_latency = treatment.get("avg_latency", 0)

        def _lift_pct(t_val: float, c_val: float) -> float:
            if c_val == 0:
                return 0.0
            return (t_val - c_val) / c_val * 100

        return {
            "ctr_control": c_ctr,
            "ctr_treatment": t_ctr,
            "ctr_lift_pct": _lift_pct(t_ctr, c_ctr),
            "cvr_control": c_cvr,
            "cvr_treatment": t_cvr,
            "cvr_lift_pct": _lift_pct(t_cvr, c_cvr),
            "ecpm_control": c_ecpm,
            "ecpm_treatment": t_ecpm,
            "ecpm_lift_pct": _lift_pct(t_ecpm, c_ecpm),
            "latency_control": c_latency,
            "latency_treatment": t_latency,
            "latency_delta": t_latency - c_latency,
        }

    # -- report generation ------------------------------------------------

    def generate_report(self, experiment_id: int) -> str:
        """Generate a formatted text report for the given experiment."""
        exp_info = self._get_experiment_info(experiment_id)
        metrics = self.get_experiment_metrics(experiment_id)

        control = metrics.get("control", {})
        treatment = metrics.get("treatment", {})

        # built-in fallback: treat empty variant as "control" if "control" key missing
        if not control and "unknown" in metrics:
            control = metrics["unknown"]

        lift = self.calculate_lift(control, treatment) if control else {}

        lines: list[str] = []
        # -- header -------------------------------------------------------
        lines.append("=" * 62)
        lines.append(f"  A/B EXPERIMENT REPORT")
        lines.append("=" * 62)
        lines.append(f"  Experiment ID     : {experiment_id}")
        lines.append(f"  Name              : {exp_info.get('name', 'N/A')}")
        lines.append(f"  Status            : {exp_info.get('status', 'N/A')}")
        if exp_info.get("started_at"):
            lines.append(f"  Started           : {exp_info['started_at']}")
        if exp_info.get("ended_at"):
            lines.append(f"  Ended             : {exp_info['ended_at']}")
        lines.append(f"  Traffic Split     : {exp_info.get('traffic_ratio', 0.50)} treatment")
        if exp_info.get("description"):
            lines.append(f"  Description       : {exp_info['description']}")
        lines.append("")

        # -- metrics table -----------------------------------------------
        c_imps = control.get("imps", 0)
        t_imps = treatment.get("imps", 0)
        c_clicks = control.get("clicks", 0)
        t_clicks = treatment.get("clicks", 0)
        c_convs = control.get("convs", 0)
        t_convs = treatment.get("convs", 0)
        c_rev = control.get("rev", 0)
        t_rev = treatment.get("rev", 0)
        c_lat = control.get("avg_latency", 0)
        t_lat = treatment.get("avg_latency", 0)
        c_cost = control.get("cost", 0)
        t_cost = treatment.get("cost", 0)

        lines.append("-" * 62)
        lines.append(f"  {'Metric':<22} {'Control':>15} {'Treatment':>15}")
        lines.append("-" * 62)
        lines.append(f"  {'Impressions':<22} {c_imps:>15,} {t_imps:>15,}")
        lines.append(f"  {'Clicks':<22} {c_clicks:>15,} {t_clicks:>15,}")
        lines.append(f"  {'Conversions':<22} {c_convs:>15,} {t_convs:>15,}")
        lines.append(f"  {'Cost ($)':<22} {c_cost:>15,.2f} {t_cost:>15,.2f}")
        lines.append(f"  {'Revenue ($)':<22} {c_rev:>15,.2f} {t_rev:>15,.2f}")

        c_ctr = lift.get("ctr_control", 0) if lift else 0
        t_ctr = lift.get("ctr_treatment", 0) if lift else 0
        c_cvr = lift.get("cvr_control", 0) if lift else 0
        t_cvr = lift.get("cvr_treatment", 0) if lift else 0
        c_ecpm = lift.get("ecpm_control", 0) if lift else 0
        t_ecpm = lift.get("ecpm_treatment", 0) if lift else 0

        lines.append(f"  {'CTR':<22} {c_ctr:>15.4f} {t_ctr:>15.4f}")
        lines.append(f"  {'CVR':<22} {c_cvr:>15.4f} {t_cvr:>15.4f}")
        lines.append(f"  {'eCPM ($)':<22} {c_ecpm:>15.4f} {t_ecpm:>15.4f}")
        lines.append(f"  {'Avg Latency (ms)':<22} {c_lat:>15.2f} {t_lat:>15.2f}")
        lines.append("-" * 62)
        lines.append("")

        # -- lift comparison ---------------------------------------------
        if lift:
            lines.append("  LIFT ANALYSIS")
            lines.append("  " + "-" * 50)
            _print_lift_line(lines, "CTR  ", lift.get("ctr_lift_pct", 0))
            _print_lift_line(lines, "CVR  ", lift.get("cvr_lift_pct", 0))
            _print_lift_line(lines, "eCPM ", lift.get("ecpm_lift_pct", 0))
            _print_lift_line(lines, "Latency", lift.get("latency_delta", 0), suffix=" ms")
            lines.append("")

        # -- verdict -----------------------------------------------------
        lines.append("  VERDICT")
        lines.append("  " + "-" * 50)
        verdict = self._compute_verdict(lift) if lift else "Insufficient data for comparison"
        lines.append(f"  {verdict}")
        lines.append("=" * 62)

        return "\n".join(lines)

    def _get_experiment_info(self, experiment_id: int) -> dict:
        """Fetch experiment metadata from MySQL."""
        if not self.mysql_dsn:
            return {}
        try:
            rows = _query_mysql(
                self.mysql_dsn,
                f"SELECT * FROM experiments WHERE id = {experiment_id}",
            )
            return rows[0] if rows else {}
        except Exception:
            return {}

    @staticmethod
    def _compute_verdict(lift: dict) -> str:
        """Compute a plain-English verdict from lift metrics."""
        ctr_lift = lift.get("ctr_lift_pct", 0)
        cvr_lift = lift.get("cvr_lift_pct", 0)
        ecpm_lift = lift.get("ecpm_lift_pct", 0)
        pos = sum(1 for v in (ctr_lift, cvr_lift, ecpm_lift) if v > 0)
        if pos >= 2 and ecpm_lift > 0:
            return "GPR significantly outperforms baseline"
        if pos == 0:
            return "No significant difference"
        return "GPR shows moderate improvement over baseline"


def _print_lift_line(
    lines: list[str], label: str, value: float, suffix: str = "%"
):
    """Format a single lift line with sign-aware coloring indicator."""
    if value > 0:
        indicator = "▲ +"
    elif value < 0:
        indicator = "▼ "
    else:
        indicator = "  "
    lines.append(
        f"  {label} lift: {indicator}{value:.2f}{suffix}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_json(data):
    """Convert data to JSON string, handling datetime objects."""
    def _default(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(data, indent=2, default=_default)


def main():
    parser = argparse.ArgumentParser(
        description="GPR ADX A/B experiment report generator"
    )
    parser.add_argument(
        "--experiment-id", type=int, help="Experiment ID to generate report for"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all experiments"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--clickhouse-url",
        default="http://localhost:8123",
        help="ClickHouse HTTP endpoint (default: http://localhost:8123)",
    )
    parser.add_argument(
        "--mysql-dsn",
        default="adx:adx_pass@localhost:3306/adx",
        help="MySQL DSN in user:pass@host:port/db format",
    )

    args = parser.parse_args()
    reporter = ABReport(
        clickhouse_url=args.clickhouse_url,
        mysql_dsn=args.mysql_dsn,
    )

    if args.list:
        try:
            exps = reporter.list_experiments()
            if args.output == "json":
                print(_format_json(exps))
            else:
                for exp in exps:
                    print(
                        f"  [{exp['id']}] {exp['name']} "
                        f"(status={exp['status']}, "
                        f"ratio={exp.get('traffic_ratio', 0.5)})"
                    )
        except Exception as exc:
            print(f"Error listing experiments: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.experiment_id is None:
        parser.print_help()
        sys.exit(1)

    try:
        report = reporter.generate_report(args.experiment_id)
        print(report)
    except Exception as exc:
        print(f"Error generating report: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
