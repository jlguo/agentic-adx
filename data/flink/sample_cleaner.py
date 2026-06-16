"""
Sample cleaner: consumes impression/click/conversion events from Kafka,
joins them into labeled training samples, and writes TSV output for
GPR model training.

Kafka event format: {"timestamp":"RFC3339","bid_id":"...","ad_id":"...","ecpm":...}
TSV output: prompt_text\\tctr\\tcvr\\tecpma
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, Optional

from kafka import KafkaConsumer


def format_sample(
    ad_id: str,
    clicked: bool = False,
    converted: bool = False,
    ecpm: float = 0.0,
    domain: str = "site",
) -> str:
    """Format a training sample as a TSV line.

    The output matches what gpr/train/pretrain.py CTRDataset._parse_line expects:
    field 0 is embedded into the prompt, fields 1-3 are ctr/cvr/ecpm floats.
    """
    prompt = f"User browsing on {domain}. Ad: {ad_id}."
    ctr = 1.0 if clicked else 0.0
    cvr = 1.0 if converted else 0.0
    return f"{prompt}\t{ctr:.1f}\t{cvr:.1f}\t{ecpm:.6f}"


class SampleCleaner:
    """Kafka consumer that joins ad events into labeled training samples."""

    TTL_SECONDS = 600

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "sample-cleaner",
        output_file: str = "data/training_samples.tsv",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.output_file = output_file
        self.buffer: list[str] = []
        self.pending: Dict[str, dict] = {}
        self._running = True
        self._consumer: Optional[KafkaConsumer] = None

    def _create_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            "ad_impressions",
            "ad_clicks",
            "ad_conversions",
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=1000,
        )

    def run(self) -> None:
        """Main loop: consume events, join into samples, flush to TSV."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._consumer = self._create_consumer()
        assert self._consumer is not None
        print(
            f"Sample cleaner started. "
            f"Bootstrap: {self.bootstrap_servers}, "
            f"Output: {self.output_file}"
        )

        try:
            for message in self._consumer:
                if not self._running:
                    break
                self._process_message(message)
                self._expire_stale_samples()
        except Exception as e:
            print(
                f"ERROR: No Kafka brokers available at {self.bootstrap_servers}. "
                f"Ensure Kafka is running and reachable.",
                file=sys.stderr,
            )
            sys.exit(1)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self._running = False

    def _process_message(self, message) -> None:
        topic = message.topic
        data = message.value
        bid_id = data.get("bid_id")
        if not bid_id:
            return

        if topic == "ad_impressions":
            self._on_impression(bid_id, data)
        elif topic == "ad_clicks":
            self._on_click(bid_id)
        elif topic == "ad_conversions":
            self._on_conversion(bid_id)

    def _on_impression(self, bid_id: str, data: dict) -> None:
        self.pending[bid_id] = {
            "ad_id": data.get("ad_id", "unknown"),
            "ecpm": float(data.get("ecpm", 0.0)),
            "clicked": False,
            "converted": False,
            "received_at": time.time(),
        }

    def _on_click(self, bid_id: str) -> None:
        if bid_id in self.pending:
            self.pending[bid_id]["clicked"] = True

    def _on_conversion(self, bid_id: str) -> None:
        if bid_id in self.pending:
            self.pending[bid_id]["converted"] = True
        self._flush_sample(bid_id)

    def _expire_stale_samples(self) -> None:
        now = time.time()
        expired = [
            bid_id
            for bid_id, sample in self.pending.items()
            if now - sample.get("received_at", now) >= self.TTL_SECONDS
        ]
        for bid_id in expired:
            self._flush_sample(bid_id)

    def _flush_sample(self, bid_id: str) -> None:
        sample = self.pending.pop(bid_id, None)
        if sample is None:
            return

        line = format_sample(
            ad_id=sample["ad_id"],
            clicked=sample["clicked"],
            converted=sample["converted"],
            ecpm=sample["ecpm"],
        )
        self.buffer.append(line)
        if len(self.buffer) >= 100:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self.buffer:
            return
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        with open(self.output_file, "a") as f:
            for line in self.buffer:
                f.write(line + "\n")
        count = len(self.buffer)
        self.buffer.clear()
        print(f"Flushed {count} samples to {self.output_file}")

    def _shutdown(self) -> None:
        print("Shutting down: flushing pending samples...")
        for bid_id in list(self.pending.keys()):
            self._flush_sample(bid_id)
        self._flush_buffer()
        if self._consumer:
            self._consumer.close()
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Sample cleaner: join Kafka ad events into GPR training samples"
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--group-id",
        default="sample-cleaner",
        help="Kafka consumer group ID (default: sample-cleaner)",
    )
    parser.add_argument(
        "--output-file",
        default="data/training_samples.tsv",
        help="Output TSV file path (default: data/training_samples.tsv)",
    )
    args = parser.parse_args()

    cleaner = SampleCleaner(
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        output_file=args.output_file,
    )
    cleaner.run()


if __name__ == "__main__":
    main()
