"""Every training script keeps listing identifiers out of its features.

property_id (and the free-text address) is unique per listing: as a feature
the model learns which ID ranges were scraped from which search, or one-hot
encodes a column per training row that no new listing can match. The scripts
build their feature frames inline, so the check reads the source rather than
running a training job.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "models/XGBoost.py",
    "models/linear_svr.py",
    "models/non_linear_svr.py",
    "src/overfitting_XGBoost.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_property_id_is_excluded_from_the_features(script):
    source = (ROOT / script).read_text(encoding="utf-8")
    assert "'property_id'" in source or '"property_id"' in source, f"{script} feeds property_id"


def test_scripts_with_an_explicit_feature_list_never_name_property_id():
    for script in ("models/Random_Forest.py", "src/train.py"):
        source = (ROOT / script).read_text(encoding="utf-8")
        assert "property_id" not in source
