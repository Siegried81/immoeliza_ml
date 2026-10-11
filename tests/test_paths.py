"""Every path the scripts use comes from src/paths.py and lives under the repo.

Pinned because the alternative is how it was: thirteen files each carrying one
developer's desktop path, so a clone anywhere else failed at the first read.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import paths  # noqa: E402


def test_no_script_carries_an_absolute_drive_path():
    offenders = [
        str(p.relative_to(ROOT))
        for p in list(ROOT.glob("src/*.py")) + list(ROOT.glob("models/*.py")) + [ROOT / "main.py"]
        if re.search(r"[A-Za-z]:\\\\", p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], offenders


def test_every_path_resolves_under_the_repository_or_the_override():
    for name in ("RAW_DATA", "CLEAN_DATA", "MODEL_PATH", "IMAGE_DIR", "MISSING_VALUES_PLOT"):
        value = getattr(paths, name)
        assert value.is_relative_to(ROOT) or value.is_relative_to(paths.DATA_DIR), name


def test_the_shipped_model_is_where_predict_looks_for_it():
    """`models/best_xgboost.joblib` is committed; predict.py must find it from any cwd."""
    assert paths.MODEL_PATH.exists()


def test_every_script_still_imports():
    """The rewrite touched the import block of thirteen files; a syntax slip in any
    one of them must fail here, not at a user's first run."""
    import importlib.util

    for path in list(ROOT.glob("src/*.py")) + list(ROOT.glob("models/*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        if "paths." in source and path.name != "paths.py":
            assert "from src import paths" in source, path.name


def test_main_imports_every_model_it_wires_up():
    """`main.py` names seven model modules; a typo in one (`Ramdom_Forest`) or a
    stale package init meant none of them could be reached. Importing main is
    the check, and its __main__ guard keeps the training from running here."""
    import importlib

    importlib.import_module("main")
