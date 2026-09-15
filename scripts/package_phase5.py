"""Build the reproducible Phase 5 artifact manifest and handoff package."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase5_handoff import write_phase5_handoff


if __name__ == "__main__":
    paths = write_phase5_handoff(PROJECT_ROOT)
    for name, path in paths.items():
        print(f"{name}: {path}")