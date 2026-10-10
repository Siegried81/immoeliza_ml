"""The data pipeline: cleaning, preprocessing, training, prediction.

Deliberately empty. The previous init imported every module of the package,
including seven that live in `models/` and not here, so `import src` raised
`ModuleNotFoundError` on a fresh clone and took `main.py` down with it. An
init that imports nothing also means `from src import paths` costs one small
module and not scikit-learn, XGBoost and matplotlib. Import the function you
need from its module: `from src.preprocessing import get_preprocessor`.
"""
