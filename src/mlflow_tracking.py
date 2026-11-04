"""Experiment tracking for the price regressors, on a local MLflow file store.

Why this lives beside the pipelines instead of inside them: the model modules
in `models/` are driven by an interactive menu (`main.py`), each prints its
metrics and pops a plot, and none returns its numbers. That is fine for a human
reading one run, but it leaves nothing to compare runs across time. This module
is the non-interactive counterpart: it trains the comparison regressors, writes
each run's params and metrics to `mlruns/` (a plain directory, no server), and
logs every fitted model plus a tag on the best one, so two training runs can be
diffed in the MLflow UI later.

Reuse, not rewrite: the estimator builders that the pipelines already expose as
functions are imported and used as-is (`models.linear_svr.build_model`,
`models.Random_Forest.get_preprocessor`). Linear regression and XGBoost build
their estimator inline inside their `run_*` function, so there is nothing to
import; those two are reconstructed here with the same configuration, and the
originating module is named in a comment so the two stay in step. The model
`run_*` functions are left untouched.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import xgboost as xgb

# The repository root, so this module imports the pipeline builders whatever the
# working directory is, mirroring how the model modules derive their own paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import paths  # noqa: E402
from models.linear_svr import build_model as build_linear_svr  # noqa: E402
from models.Random_Forest import get_preprocessor as rf_get_preprocessor  # noqa: E402

# Default location of the file store. file:// keeps MLflow entirely on disk, so
# no tracking server and no network connection are ever involved.
DEFAULT_MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Libraries needed to load a logged pipeline back. Passed to log_model so
# MLflow does not introspect the whole environment on every run: that
# inference costs several seconds per model and shells out to pip, and the
# pipelines only ever need these few packages to unpickle and predict.
_MODEL_REQUIREMENTS = ["scikit-learn", "xgboost", "numpy", "pandas", "cloudpickle"]

# Columns that are never model inputs: price is the target, property_id is a
# listing number and address is free text. Price-derived columns (price_per_m2)
# are dropped separately, by name derivation, in _split_features.
_NON_FEATURES = ["price", "address", "property_id"]


def _load_dataframe(data_path):
    """Load the cleaned listings and drop rows that cannot train or score a
    regressor: no price, or a non-positive one.

    The price>0 drop is the one `run_xgboost_pipeline` and `run_random_forest`
    make; the other pipelines drop only null prices. Applying it to every model
    here is a harmless strictening, since a non-positive price is not a
    trainable target.
    """
    df = pd.read_json(data_path)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()
    return df


def _split_features(df):
    """Return (X, y, num_cols, cat_cols) for the full-feature models.

    The leakage drop is done by derivation, not by a fixed name list, so a
    future `price_*` column cannot silently reappear as a feature. Same rule as
    the pipelines, kept in one place.
    """
    X = df.drop(columns=_NON_FEATURES, errors="ignore")
    leaking = [c for c in X.columns if "price" in c.lower()]
    X = X.drop(columns=leaking)
    y = df["price"]
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, num_cols, cat_cols


# --- model builders -------------------------------------------------------
# Each builder takes the loaded DataFrame and returns (X, y, estimator, params).
# params are the hyperparameters worth recording for a run; they are logged
# verbatim, so what shows up in MLflow is what actually trained.

def _build_linear_regression(df):
    """Single-feature linear regression on livable surface (see
    models/linear_regression.py). Reconstructed, not imported, because that
    module builds the estimator inline inside run_linear_regression()."""
    sub = df.dropna(subset=["livable_surface"])
    X = sub[["livable_surface"]]
    y = sub["price"]
    estimator = LinearRegression()
    params = {"features": "livable_surface"}
    return X, y, estimator, params


def _build_random_forest(df):
    """Random forest on a log-price target, reusing the preprocessor factory
    from models/Random_Forest.py so the encoding stays identical."""
    # Feature selection mirrors models/Random_Forest.py and doesn't use
    # _NON_FEATURES: property_id and address are strings, so select_dtypes
    # on numbers never picks them, and the fixed cat_cols list never names them.
    cat_cols = ["property_type", "city", "province", "property_state", "nearest_city"]
    num_cols = df.select_dtypes(include=[np.number]).columns.drop("price").tolist()
    num_cols = [c for c in num_cols if "price" not in c.lower()]
    cat_cols = [c for c in cat_cols if c in df.columns]
    X = df[num_cols + cat_cols]
    y = df["price"]
    estimator = TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", rf_get_preprocessor(num_cols, cat_cols)),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )
    rf = estimator.regressor.named_steps["rf"]
    params = {"n_estimators": rf.n_estimators, "random_state": rf.random_state}
    return X, y, estimator, params


def _build_linear_svr(df):
    """Linear SVR, reusing models.linear_svr.build_model unchanged. The menu
    pipeline tunes C/epsilon with a randomized search; tracking fits the base
    estimator with its defaults instead, so a run is deterministic and cheap.
    The defaults that train are what gets logged."""
    X, y, num_cols, cat_cols = _split_features(df)
    estimator = build_linear_svr(num_cols, cat_cols)
    svr = estimator.regressor.named_steps["svr"]
    params = {"C": svr.C, "epsilon": svr.epsilon, "tuned": False}
    return X, y, estimator, params


def _build_xgboost(df):
    """XGBoost on a log-price target (see models/XGBoost.py). Reconstructed,
    not imported, because that module builds the estimator inline inside
    run_xgboost_pipeline(); the configuration below is kept equal to it."""
    X, y, num_cols, cat_cols = _split_features(df)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="passthrough",
    )
    estimator = Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )),
    ])
    booster = estimator.named_steps["model"].regressor
    params = {k: getattr(booster, k) for k in ("n_estimators", "learning_rate", "max_depth", "random_state")}
    return X, y, estimator, params


# Registry of the regressors compared in one tracking session. Cheapest first.
# kNN / tuned-XGBoost / non-linear SVR from the menu are left out on purpose:
# they wrap a cross-validated search (and the non-linear SVR a 100-component
# SVD), which is a search to track, not a single regressor.
MODEL_BUILDERS = {
    "linear_regression": _build_linear_regression,
    "random_forest": _build_random_forest,
    "linear_svr": _build_linear_svr,
    "xgboost": _build_xgboost,
}


def _evaluate(y_true, y_pred):
    """Metrics on the real euro scale, the scale the pipelines print. MSE is
    left out (RMSE carries it); MAPE and its median are added because
    src/train.py reports them and the README reads them together.

    R2 on a log target is a different, more flattering number; every metric
    here is computed on euros so runs are comparable to the printed output and
    to each other.
    """
    ape = np.abs(y_true - y_pred) / y_true * 100
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape": float(ape.mean()),
        "mape_median": float(np.median(ape)),
    }


def track_models(
    data_path=None,
    tracking_uri=None,
    experiment_name="immoeliza_regressors",
    models=None,
    test_size=0.2,
    random_state=42,
):
    """Train each regressor, log its params and metrics to a local MLflow file
    store, and log every fitted model plus a tag on the best one.

    Parameters
    ----------
    data_path : path to the cleaned listings JSON. Defaults to the repository's
        cleaned dataset (paths.CLEAN_DATA).
    tracking_uri : MLflow tracking URI. Defaults to the repo's `mlruns/` folder
        as a `file:` store, so nothing leaves the machine. Tests pass a tmp dir.
    experiment_name : the MLflow experiment the runs are grouped under.
    models : subset of MODEL_BUILDERS keys to run; None runs them all.

    Returns a summary dict: the experiment id, one entry per run (name, run id,
    params, metrics) and the name of the best run by test R2. The return value
    is what a caller or a test inspects; the durable record is in the store.
    """
    data_path = Path(data_path) if data_path is not None else paths.CLEAN_DATA
    if tracking_uri is None:
        tracking_uri = DEFAULT_MLRUNS_DIR.as_uri()

    # MLflow 3 gates the on-disk file store behind this flag (it steers users
    # toward a database backend). A local file store is exactly what this
    # project wants: no server, no network. Opt in so the entry runs as-is.
    if tracking_uri.startswith("file") or "://" not in tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    names = list(models) if models is not None else list(MODEL_BUILDERS)
    df = _load_dataframe(data_path)

    runs = []
    for name in names:
        builder = MODEL_BUILDERS[name]
        X, y, estimator, params = builder(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        with mlflow.start_run(run_name=name) as run:
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            metrics = _evaluate(y_test.to_numpy(), np.asarray(y_pred))

            mlflow.log_param("model", name)
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log the fitted estimator as an artifact so a run is reproducible
            # and the model can be pulled back out of the store directly.
            # cloudpickle, not the skops default: skops refuses to round-trip
            # the tree node storage behind RandomForest/XGBoost, which would
            # break logging the very models being compared.
            model_info = mlflow.sklearn.log_model(
                estimator,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                pip_requirements=_MODEL_REQUIREMENTS,
            )

            runs.append({
                "name": name,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "params": params,
                "metrics": metrics,
            })

    # Best run by test R2, tagged in place so the winner is findable in the UI.
    # Picked on the test split, like the README's comparison table: there is no
    # separate validation set, so the winner's R2 is a slightly optimistic
    # estimate.
    best = max(runs, key=lambda r: r["metrics"]["r2"]) if runs else None
    if best is not None:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        client.set_tag(best["run_id"], "best_model", "true")

    return {
        "experiment_id": experiment_id,
        "runs": runs,
        "best": best["name"] if best else None,
    }


if __name__ == "__main__":
    summary = track_models()
    print(f"Tracked {len(summary['runs'])} runs in experiment {summary['experiment_id']}")
    for r in summary["runs"]:
        print(f"  {r['name']:<18} R2={r['metrics']['r2']:.4f}  MAE={r['metrics']['mae']:,.0f} EUR")
    print(f"Best by test R2: {summary['best']}")
