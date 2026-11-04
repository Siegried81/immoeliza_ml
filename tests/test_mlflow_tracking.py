"""The tracking entry records a run, with params and metrics, for each model.

Runs end to end against a tiny synthetic dataset and a throwaway MLflow file
store in tmp_path. There is no tracking server and no HTTP: a `file:` store is
pure disk I/O, so there is nothing to mock, and the test asserts the store is a
local file URI to keep it that way. A failure here means the tracking harness
stopped writing params, metrics or the model artifact that the MLflow UI reads.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import mlflow  # noqa: E402

from src.mlflow_tracking import MODEL_BUILDERS, track_models  # noqa: E402


def _synthetic_listings(n=80, seed=0):
    """A frame with the columns the pipelines expect, including the two traps.

    price_per_m2 is the leaking column the tracking harness must drop by
    derivation; property_id and address are the non-features it must ignore.
    Prices follow surface so the regressors have a real signal to fit.
    """
    rng = np.random.default_rng(seed)
    surface = rng.uniform(50, 300, n)
    price = (2500 * surface * rng.uniform(0.9, 1.1, n)).round().astype(int)
    return pd.DataFrame({
        "property_type": rng.choice(["house", "apartment"], n),
        "property_id": [f"id_{i}" for i in range(n)],
        "postcode": rng.integers(1000, 9999, n),
        "city": rng.choice(["Gent", "Liege", "Namur", "Mons"], n),
        "province": rng.choice(["East Flanders", "Liege", "Namur"], n),
        "address": [f"street {i}" for i in range(n)],
        "latitude": rng.uniform(49.5, 51.5, n),
        "longitude": rng.uniform(2.5, 6.0, n),
        "price": price,
        "property_state": rng.choice(["good", "to renovate", "new"], n),
        "build_year": rng.integers(1900, 2024, n).astype(float),
        "bedroom_count": rng.integers(1, 6, n).astype(float),
        "livable_surface": surface,
        "total_surface": surface + rng.uniform(0, 100, n),
        "garage": rng.integers(0, 2, n),
        "terrace": rng.integers(0, 2, n).astype(float),
        "energy_consumption_kWh/m2/year": rng.uniform(50, 500, n),
        "swimming_pool": rng.integers(0, 2, n),
        "preschool_distance_m": rng.uniform(100, 5000, n),
        "train_station_distance_m": rng.uniform(100, 10000, n),
        "supermarket_distance_m": rng.uniform(100, 5000, n),
        "nearest_city": rng.choice(["Brussels", "Antwerp", "Charleroi"], n),
        "nearest_city_distance_km": rng.uniform(1, 80, n),
        "price_per_m2": (price / surface).round(2),  # leakage trap
    })


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "cleaned_data.json"
    _synthetic_listings().to_json(path)
    return path


@pytest.fixture
def tracking_uri(tmp_path):
    return (tmp_path / "mlruns").as_uri()


def test_every_model_gets_a_run_with_params_and_metrics(dataset, tracking_uri):
    summary = track_models(
        data_path=dataset,
        tracking_uri=tracking_uri,
        experiment_name="test_exp",
    )

    # A local file store, so the test made no network call.
    assert tracking_uri.startswith("file:")
    assert len(summary["runs"]) == len(MODEL_BUILDERS)

    # Read the runs back from the store the way the MLflow UI would.
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test_exp")
    assert experiment is not None
    stored = client.search_runs([experiment.experiment_id])
    assert len(stored) == len(MODEL_BUILDERS)

    for run in stored:
        # The model name and the train/test sizes are always logged as params.
        assert run.data.params.get("model") in MODEL_BUILDERS
        assert int(run.data.params["n_train"]) > 0
        assert int(run.data.params["n_test"]) > 0
        # Euro-scale metrics are all present and finite.
        for metric in ("r2", "mae", "rmse", "mape"):
            assert metric in run.data.metrics
            assert np.isfinite(run.data.metrics[metric])


def test_best_run_is_tagged_and_model_artifact_is_logged(dataset, tracking_uri):
    summary = track_models(
        data_path=dataset,
        tracking_uri=tracking_uri,
        experiment_name="test_exp",
        models=["linear_regression", "xgboost"],
    )

    assert summary["best"] in {"linear_regression", "xgboost"}

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test_exp")
    runs = {r.data.tags.get("mlflow.runName"): r for r in client.search_runs([experiment.experiment_id])}

    # Exactly the best run carries the best_model tag.
    tagged = [name for name, r in runs.items() if r.data.tags.get("best_model") == "true"]
    assert tagged == [summary["best"]]

    # Each run logged a model artifact that loads back into a usable estimator.
    mlflow.set_tracking_uri(tracking_uri)
    for entry in summary["runs"]:
        assert entry["model_uri"]
        loaded = mlflow.sklearn.load_model(entry["model_uri"])
        assert hasattr(loaded, "predict")


def test_leaking_and_non_feature_columns_are_not_trained_on(dataset, tracking_uri):
    """The harness must drop price_per_m2, property_id and address before fit,
    the same way the pipelines do, so tracked scores are not inflated by the
    leak. Checked through the builders, which are what track_models feeds fit.
    """
    from src.mlflow_tracking import _build_xgboost

    df = pd.read_json(dataset)
    X, y, _, _ = _build_xgboost(df)

    assert "price_per_m2" not in X.columns
    assert "price" not in X.columns
    assert "property_id" not in X.columns
    assert "address" not in X.columns
