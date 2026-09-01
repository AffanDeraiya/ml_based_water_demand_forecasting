"""Command-line interface for Phase 3 synthetic DNH-total data generation.

Usage:
    python -m src.data_generation.generate_synthetic_dnh_total --config configs/synthetic_dnh_total_v2.json
"""

import json
import sys
from pathlib import Path
import argparse

from src.data_generation.synthetic_dnh_total_generator import SyntheticDNHTotalGenerator
from src.validation.validate_dnh_total_dataset import validate_dnh_total_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DNH-total monthly water-demand dataset (Version 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_dnh_total_v2.json",
        help="Path to config JSON"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        gen = SyntheticDNHTotalGenerator(config)
    except Exception as e:
        print(f"ERROR: Generator initialization failed: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        df = gen.generate_all()
    except Exception as e:
        print(f"ERROR: Data generation failed: {e}", file=sys.stderr)
        sys.exit(3)

    # Validate
    validation_result = validate_dnh_total_dataset(df, config)
    if not validation_result["ok"]:
        print("ERROR: Validation failed:", file=sys.stderr)
        for error in validation_result["errors"]:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(4)

    # Write outputs
    try:
        gen.write_output(df, validation_result=validation_result)
    except Exception as e:
        print(f"ERROR: Failed to write outputs: {e}", file=sys.stderr)
        sys.exit(5)

    print("SUCCESS: Synthetic data generated and validated")
    sys.exit(0)


if __name__ == "__main__":
    main()
