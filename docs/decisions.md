# Decisions

Dated record of choices that change what a number means or how it's measured.
Newest first.

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
