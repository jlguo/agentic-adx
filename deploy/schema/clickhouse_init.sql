CREATE TABLE IF NOT EXISTS adx_analytics.ad_metrics (
    timestamp       DateTime DEFAULT now(),
    experiment_id   Int64 DEFAULT 0,
    variant         LowCardinality(String) DEFAULT '',
    campaign_id     UInt64,
    impressions     UInt64 DEFAULT 0,
    clicks          UInt64 DEFAULT 0,
    conversions     UInt64 DEFAULT 0,
    cost            Float64 DEFAULT 0,
    revenue         Float64 DEFAULT 0,
    gpr_used        UInt8 DEFAULT 0,
    latency_ms      Float64 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (experiment_id, campaign_id, timestamp)
PARTITION BY toYYYYMMDD(timestamp)
TTL timestamp + INTERVAL 90 DAY;
