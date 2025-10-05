"""Command-line entry point for the cirrus classification pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from cirrusclassify.processing.batch import BatchProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    processor = BatchProcessor.from_config(config)
    processor.run()


if __name__ == "__main__":
    main()
