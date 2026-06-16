"""
Training trigger — monitors training sample accumulation and fires LoRA fine-tuning.

Polls data/training_samples.tsv for new samples. When line count reaches a threshold,
invokes gpr/train/pretrain.py as a subprocess and archives the consumed samples.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("training_trigger")

_RUNNING = True


def _shutdown_handler(signum, frame):
    global _RUNNING
    logger.info("Received signal %s, shutting down gracefully...", signum)
    _RUNNING = False


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


def count_lines(path: str) -> int:
    """Count non-empty lines in a file."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def archive_data_file(path: str) -> str:
    """Rename the data file with a timestamp suffix."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"{path}_{timestamp}"
    os.rename(path, archive_path)
    logger.info("Archived %s -> %s", path, archive_path)
    return archive_path


def trigger_training(data_path: str, epochs: int, batch_size: int) -> int:
    """Run pretrain.py as a subprocess and return its exit code."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "..", "gpr", "train", "pretrain.py"),
        "--data", data_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    logger.info("Triggering training: %s", " ".join(cmd))

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.getcwd())
    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
        logger.info("Training completed successfully (exit code %d)", result.returncode)
    else:
        logger.error("Training failed (exit code %d)", result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Monitor training samples and trigger LoRA fine-tuning")
    parser.add_argument("--threshold", type=int, default=500,
                        help="Minimum samples before triggering training (default: 500)")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between file checks (default: 30)")
    parser.add_argument("--data-file", default="data/training_samples.tsv",
                        help="Path to training samples file (default: data/training_samples.tsv)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    args = parser.parse_args()

    data_path = os.path.normpath(args.data_file)
    logger.info("Monitoring %s (threshold: %d, interval: %ds)",
                data_path, args.threshold, args.poll_interval)

    while _RUNNING:
        line_count = count_lines(data_path)
        logger.info("Sample count: %d / %d", line_count, args.threshold)

        if line_count >= args.threshold:
            logger.info("Threshold reached (%d >= %d), starting training...",
                        line_count, args.threshold)

            # Archive current file BEFORE starting training so new samples
            # go into a fresh file while training reads the archive.
            archived = archive_data_file(data_path)
            trigger_training(archived, args.epochs, args.batch_size)

        time.sleep(args.poll_interval)

    logger.info("Training trigger stopped.")


if __name__ == "__main__":
    main()
