"""The linear SVR stays inside the market when one input is absurd.

Listings carry data-entry outliers (an energy consumption of 20 million
kWh/m2/year, 100 bedrooms). A linear model on a log target extrapolates them
exponentially, so the guard is a prediction on such a row, not a metric.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.linear_svr import build_model


def _listings(n=300, seed=0):
    rng = np.random.default_rng(seed)
    surface = rng.uniform(50, 300, n)
    energy = rng.uniform(50, 500, n)
    city = rng.choice(["Gent", "Liege", "Namur"], n)
    price = 2500 * surface * (1 - energy / 2000) * rng.uniform(0.9, 1.1, n)
    return pd.DataFrame({"livable_surface": surface, "energy": energy, "city": city}), pd.Series(price)


def test_an_absurd_input_does_not_produce_an_absurd_price():
    X, y = _listings()
    model = build_model(["livable_surface", "energy"], ["city"]).fit(X, y)
    outlier = pd.DataFrame({"livable_surface": [150.0], "energy": [2.0e7], "city": ["Gent"]})
    price = model.predict(outlier)[0]
    assert y.min() / 10 < price < 10 * y.max()


def test_ordinary_listings_are_priced_sensibly():
    X, y = _listings()
    model = build_model(["livable_surface", "energy"], ["city"]).fit(X, y)
    X_new, y_new = _listings(seed=1)
    error = np.median(np.abs(model.predict(X_new) - y_new) / y_new)
    assert error < 0.25
