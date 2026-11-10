# Decisions

Dated record of choices that change what a number means or how it's measured.
Newest first.

## 2026-11-10: what the R² figures measure, and they don't reproduce exactly

**What.** The README now says what its R² figures are: test scores on the
3,150 held-out rows of the 80/20 split (`random_state=42`). It also says they
aren't the deployed model's figures and that they drift with library versions.

**Why.** The question was whether those rows also served tuning. They don't:
every randomized search runs 3-fold CV on the training rows only, and
`XGBoost.py` has fixed settings and no early stopping. The one weak spot is
model selection: eight models are compared on the same 3,150 rows and the best
is picked there, so the top score is a bit flattering. The R² 0.79 (0.7956 and
0.7908) also kept being quoted as the deployed service's score. It isn't: the
service in immo-eliza-deployment is a different pipeline, with R² 0.646 for
its standard model on its own sealed test split (2026-11-10).

**What it means for the numbers.** Re-run on 2026-11-10 in python:3.12 Linux
with xgboost 3.2.0 and scikit-learn 1.9.0, same data (15,746 rows):

| Script | README | pandas 2.3.3 | pandas 3.1.1 |
|---|---:|---:|---:|
| `models/XGBoost.py` | 0.7956 | 0.7881 | 0.7749 |
| `models/tuning_XGBoost.py` | 0.7908 | 0.7886 | 0.7886 |

Python 3.14 on Windows gives the same 0.7881 and 0.7749 for `XGBoost.py`. The
tuned search now picks `n_estimators=200, max_depth=7, learning_rate=0.1,
subsample=0.9, colsample_bytree=0.7`, not the parameters in the README. The
table wasn't rebuilt: the README keeps the measured figures and says they move.

**Revisit if.** `requirements.txt` gets pinned. Then rebuild the table once
and the figures stop moving.

## 2026-11-04 — MLflow tracking logs linear_svr untuned

**What.** `src/mlflow_tracking.py` fits `linear_svr` with LinearSVR's defaults
(C=1, epsilon=0). The menu pipeline in `models/linear_svr.py` runs a
`RandomizedSearchCV(n_iter=10, cv=3)` over C and epsilon first and reports the
best estimator. The tracker doesn't.

**Why.** A tracked run should be one regressor with fixed params, cheap and
deterministic. The search is 30 fits and picks params per run, which is a
search to track, not a single model. Porting it is a separate change.

**What it means for the numbers.** The tracked `linear_svr` R² is lower than
the README's 0.6624 and isn't comparable to that table row. The other three
match the table to the digit (measured 2026-11-04: LR R²=0.4404, RF 0.7283,
XGB 0.7956). The params logged for the run are read from the estimator, so
MLflow shows what actually trained: C=1.0, epsilon=0.0, tuned=False.

**Revisit if.** The randomized search is ported into the tracker. Then log the
chosen C and epsilon, set `tuned=True`, and drop this caveat from the README.
