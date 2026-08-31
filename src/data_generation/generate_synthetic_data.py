"""CLI to generate synthetic data for DNH water-demand forecasting.

This is a stub entrypoint. The full generator implementation will be in
`synthetic_generator.py`.
"""
import argparse
import json
from pathlib import Path

from .synthetic_generator import SyntheticGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    gen = SyntheticGenerator(cfg)
    gen.generate_all()


if __name__ == "__main__":
    main()
