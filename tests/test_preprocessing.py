"""
Tests for the pure helpers and for the path wiring of the package.

They use small in-memory frames and never read the real dataset, so the suite
runs without data, without network and without training anything. The path
tests exist because every module used to hardcode one developer's absolute
Windows path, which made the whole package unrunnable anywhere else.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.preprocessing import get_preprocessor, remove_highly_correlated_features


class TestGetPreprocessor:
    def test_imputes_and_scales_numeric_columns(self):
        df = pd.DataFrame({"surface": [100.0, 200.0, None], "city": ["Gent", "Gent", "Liege"]})
        out = get_preprocessor(["surface"], ["city"]).fit_transform(df)

        assert not np.isnan(out).any()
        # StandardScaler centres the column it was fitted on.
        assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-9)

    def test_one_hot_encodes_categories(self):
        df = pd.DataFrame({"surface": [100.0, 120.0], "city": ["Gent", "Liege"]})
        out = get_preprocessor(["surface"], ["city"]).fit_transform(df)

        # One scaled numeric column plus one indicator per city.
        assert out.shape == (2, 3)

    def test_unknown_category_does_not_raise(self):
        train = pd.DataFrame({"surface": [100.0, 120.0], "city": ["Gent", "Liege"]})
        preprocessor = get_preprocessor(["surface"], ["city"]).fit(train)

        out = preprocessor.transform(pd.DataFrame({"surface": [110.0], "city": ["Mons"]}))

        # handle_unknown="ignore": the unseen city becomes all-zero indicators.
        assert out[0, 1:].sum() == 0

    def test_is_returned_unfitted(self):
        preprocessor = get_preprocessor(["surface"], ["city"])
        with pytest.raises(Exception):
            preprocessor.transform(pd.DataFrame({"surface": [1.0], "city": ["Gent"]}))


class TestRemoveHighlyCorrelatedFeatures:
    def test_drops_one_of_two_redundant_columns(self):
        base = np.arange(20, dtype=float)
        df = pd.DataFrame({"a": base, "a_copy": base, "b": base[::-1] % 7})

        out = remove_highly_correlated_features(df, threshold=0.90)

        assert len(out.columns) == 2
        assert "a" in out.columns and "a_copy" not in out.columns

    def test_keeps_independent_columns(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})

        out = remove_highly_correlated_features(df, threshold=0.90)

        assert list(out.columns) == ["a", "b"]

    def test_uses_absolute_correlation(self):
        base = np.arange(20, dtype=float)
        df = pd.DataFrame({"a": base, "mirror": -base})

        out = remove_highly_correlated_features(df, threshold=0.90)

        assert list(out.columns) == ["a"]


class TestPathsAreLocationIndependent:
    """Every module must derive its paths from __file__, not from a literal."""

    MODULES = [
        "src/cleaning.py", "src/heatmap.py", "src/overfitting_XGBoost.py",
        "src/plot_missing.py", "src/predict.py", "src/preprocessing.py", "src/train.py",
        "models/XGBoost.py", "models/linear_regression.py",
        "models/linear_svr.py", "models/non_linear_svr.py", "models/Random_Forest.py",
        "models/tuning_XGBoost.py", "models/cross_validation_kNN_XGBoost.py",
    ]

    @pytest.mark.parametrize("relpath", MODULES)
    def test_no_absolute_path_literal(self, relpath):
        source = (BASE_DIR / relpath).read_text(encoding="utf-8")
        assert "D:\\Users" not in source
        assert "Path.cwd()" not in source

    def test_data_and_model_dirs_resolve_under_the_repo(self):
        from src import train

        assert train.DATA_CLEAN == BASE_DIR / "data" / "clean"
        assert train.MODEL_DIR == BASE_DIR / "models"


class TestMetricScale:
    """
    Pins why train.py reports R² on the euro scale.

    log1p compresses the expensive tail, so the same predictions score much
    higher on the log scale. Reporting that number next to a euro MAE and RMSE
    overstated the model.
    """

    def test_log_scale_r2_flatters_the_same_predictions(self):
        from sklearn.metrics import r2_score

        rng = np.random.default_rng(0)
        y_real = np.exp(rng.normal(12.5, 0.8, size=1000))  # skewed like prices
        y_log = np.log1p(y_real)
        pred_log = y_log + rng.normal(0, 0.3, size=1000)

        r2_log = r2_score(y_log, pred_log)
        r2_real = r2_score(y_real, np.expm1(pred_log))

        assert r2_log > r2_real

    def test_train_reports_r2_on_the_real_scale(self):
        source = (BASE_DIR / "src" / "train.py").read_text(encoding="utf-8")
        assert "r2 = r2_score(y_test_real, y_pred_real)" in source
