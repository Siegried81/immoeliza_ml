"""Non-interactive entry point for MLflow experiment tracking.

main.py is an interactive menu, so it cannot be used in CI or a scheduled run.
This script is the headless counterpart: it trains the comparison regressors
and records each one to a local MLflow file store under `mlruns/`. No server,
no network. Open the results afterwards with `mlflow ui` pointed at that folder.

    python scripts/track_training.py
    python scripts/track_training.py --models xgboost random_forest

The real training logic lives in src/mlflow_tracking.py; this file only parses
arguments and prints a summary.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_tracking import MODEL_BUILDERS, track_models  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=None,
        help="Path to the cleaned listings JSON (defaults to data/clean/cleaned_data.json).",
    )
    parser.add_argument(
        "--tracking-uri", default=None,
        help="MLflow tracking URI (defaults to the repo's mlruns/ file store).",
    )
    parser.add_argument(
        "--experiment", default="immoeliza_regressors",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, choices=list(MODEL_BUILDERS),
        help="Subset of models to track (defaults to all).",
    )
    args = parser.parse_args(argv)

    summary = track_models(
        data_path=args.data,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        models=args.models,
    )

    print(f"Tracked {len(summary['runs'])} runs in experiment {summary['experiment_id']}")
    for r in summary["runs"]:
        m = r["metrics"]
        print(f"  {r['name']:<18} R2={m['r2']:.4f}  MAE={m['mae']:,.0f} EUR  MAPE={m['mape']:.1f}%")
    print(f"Best by test R2: {summary['best']}")


if __name__ == "__main__":
    main()
