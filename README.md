# Immo Eliza Machine Learning

## Description

This project is the next stage of the **Immo Eliza** data science pipeline. After collecting real estate data through web scraping and cleaning the dataset during the previous projects, the goal is now to build a machine learning model capable of predicting the selling price of residential properties in Belgium.

Several regression algorithms were implemented, trained, and compared using the same preprocessing pipeline. Their performances were evaluated using common regression metrics to identify the most accurate and reliable model.

The final solution is based on **XGBoost**, which achieved the best balance between prediction accuracy, robustness, and generalization on unseen data.

---

## Project Objectives

The main objectives of this project are to:

- Build a complete and reusable machine learning pipeline.
- Prepare the dataset for machine learning using professional preprocessing techniques.
- Compare several regression algorithms.
- Evaluate each model using multiple performance metrics.
- Prevent data leakage by using Scikit-Learn Pipelines.
- Save the trained models for future predictions.
- Predict Belgian house prices as accurately as possible.

---

## Dataset

The dataset comes from the previous **Immo Eliza** projects.

It contains information about residential properties listed for sale across Belgium, including numerical and categorical features such as:

- Property type
- Province
- Region
- Postal code
- Living area
- Total surface
- Number of bedrooms
- Building condition
- Terrace
- Swimming pool
- Geographic coordinates
- And additional characteristics

The target variable is:

**`price`** (property sale price in euros)

Before training the models, the dataset was cleaned and preprocessed to improve data quality and model performance.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Siegried81/immo-eliza-ml.git
cd immo-eliza-ml
```

### 2. Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
immoeliza_ml/
├── analysis/                           # exploratory notebooks
├── data/                               # not versioned, see .gitignore
│   ├── raw/dataframe.json              # output of the scraping project
│   ├── clean/cleaned_data.json         # produced by src/cleaning.py
│   ├── input/new_properties.json       # properties to predict
│   └── output/predictions.json         # produced by src/predict.py
├── images/                             # not versioned: plots from the models
├── models/                             # one script per algorithm (.joblib files not versioned)
│   ├── linear_regression.py
│   ├── Random_Forest.py
│   ├── linear_svr.py                   # same code as linear_SVM.py
│   ├── non_linear_svr.py
│   ├── XGBoost.py
│   ├── tuning_XGBoost.py
│   └── cross_validation_kNN_XGBoost.py
├── src/
│   ├── cleaning.py                     # raw JSON -> cleaned_data.json
│   ├── preprocessing.py                # the shared ColumnTransformer
│   ├── train.py                        # final XGBoost pipeline -> best_xgboost.joblib
│   ├── predict.py                      # best_xgboost.joblib -> predictions.json
│   ├── heatmap.py                      # correlation heatmap
│   ├── plot_missing.py                 # missing-value bar chart
│   └── overfitting_XGBoost.py          # train/test gap diagnostic
├── tests/
│   └── test_preprocessing.py
├── main.py                             # menu that runs one model pipeline
├── requirements.txt
└── .gitignore
```

The project follows a modular structure to improve readability, maintainability, and reproducibility.

- **analysis/** contains exploratory notebook and model experiments.
- **data/** contains the raw and cleaned datasets. It is not versioned: copy the scraping project's `dataframe.json` into `data/raw/` and run `python src/cleaning.py`.
- **images/** stores figures used during model evaluation. Not versioned.
- **models/** holds one script per algorithm. The trained `.joblib` files it also receives are not versioned, because they are large, binary and tied to the installed library versions.
- **src/** contains the project's Python modules, including preprocessing, training, evaluation, and prediction scripts.

Every module derives its paths from `Path(__file__)`, so the scripts run from
any working directory and from any clone of the repository.

Keeping preprocessing, training, evaluation, and prediction in separate modules makes the workflow easier to understand, test, and reuse.

### Running it

```bash
python src/cleaning.py        # data/raw/dataframe.json -> data/clean/cleaned_data.json
python src/train.py           # trains the final pipeline -> models/best_xgboost.joblib
python src/predict.py         # data/input/new_properties.json -> data/output/predictions.json
python main.py                # menu: pick one algorithm to train and plot
```

`main.py` prints a menu and runs the chosen pipeline. Each pipeline loads the
cleaned dataset and builds the preprocessing its own feature selection needs,
so the menu passes nothing to it. Failures are not caught: a broken pipeline
shows its traceback instead of a one-line message.

### Tests

```bash
pip install pytest
pytest
```

Expected: `25 passed`. The suite uses small in-memory frames only, so it needs
no dataset, no network, and trains nothing. It covers `get_preprocessor`,
`remove_highly_correlated_features`, the absence of absolute path literals in
every module, and the metric scale described below.

# Data Preparation & Preprocessing

Building an accurate machine learning model starts with high-quality data. Before training any algorithm, the dataset was cleaned and transformed to ensure that every model received consistent, reliable, and meaningful input.

The entire preprocessing workflow was implemented using **Scikit-Learn Pipelines**, making the process reproducible and preventing data leakage.

---

# Data Cleaning

Several cleaning operations were performed before training the models.

## Missing Values

The dataset contained missing values in both numerical and categorical features.

Instead of removing every incomplete row—which would significantly reduce the dataset size—missing values were imputed during preprocessing:

- **Numerical features:** median imputation (`SimpleImputer(strategy="median")`)
- **Categorical features:** most frequent value or constant category, depending on the feature

This approach preserves as much information as possible while allowing every property to be processed by the models.

---

## Invalid Data

To avoid unrealistic predictions, invalid observations were removed before training.

Examples include:

- Properties with a price equal to or below zero.
- Duplicate observations.
- Rows containing invalid target values.

Removing these records improves the overall quality of the training data and reduces noise.

---

## Feature Types

The dataset contains both numerical and categorical variables.

Examples of numerical features include:

- Living surface
- Number of bedrooms
- Postal code
- Latitude
- Longitude

Examples of categorical features include:

- Property type
- Province
- Region
- Building condition

Because machine learning algorithms only work with numerical inputs, categorical variables must be converted into numerical representations before training.

---

# Feature Selection

Before building the models, a correlation analysis was performed on the numerical variables.

A correlation heatmap helped identify the strongest relationships with the target variable (`price`) while also highlighting possible redundancy between predictors.

## Main Findings

The variables with the strongest correlation to property price were:

| Feature | Correlation with Price |
|----------|-----------------------:|
| Livable Surface | **0.65** |
| Bedroom Count | **0.50** |
| Total Surface | **0.38** |

As expected, larger properties generally have higher selling prices.

The analysis also showed a relatively strong relationship between **livable surface** and **bedroom count** (0.72). Although these variables are correlated, they do not exceed the commonly accepted multicollinearity threshold (≈0.80), so both were retained.

Another observation concerned the feature:

`energy_consumption_kWh/m²/year`

This variable showed almost no correlation with the other numerical features. While this does not necessarily mean it is useless, it suggests that:

- the information may be noisy,
- the feature may contain many missing values,
- or its relationship with price is non-linear.

Rather than removing it immediately, it was kept for the machine learning models, especially tree-based algorithms that can detect more complex relationships.

---

## Why No Features Were Removed

Since no pair of variables showed excessive multicollinearity, all available predictors were retained, except the address (useless) and the build year (43.4% of missing values).

An earlier `run_preprocessing()` in `src/preprocessing.py` wrote a
`final_processed_data.csv` by keeping the columns whose correlation with
`price` was at least 0.05. That step was removed, for two reasons:

- It was **selection leakage**: the correlation was measured on the whole
  dataset, test rows included, so the choice of features already knew the
  answers it was going to be scored on.
- It scaled `price` **as if it were a feature**, which puts the target inside
  the design matrix.

Nothing read the file (`src/train.py` loads `cleaned_data.json` directly), so
removing it loses no step of the pipeline and takes away the trap. Feature
selection, if it comes back, belongs inside the Pipeline so it is fitted on the
training split only — which is what `get_preprocessor()` is for.
`remove_highly_correlated_features()` is kept: it compares features with each
other and never with the target, so it does not leak.

Keeping more features allows ensemble methods such as **Random Forest** and **XGBoost** to automatically determine which variables are the most informative during training.

If future experiments show that certain variables contribute little to prediction quality, feature selection or feature engineering could be applied.

---

# Preprocessing Pipeline

A complete preprocessing pipeline was built using **Scikit-Learn's Pipeline and ColumnTransformer**.

This ensures that exactly the same preprocessing steps are applied during both training and prediction.

The pipeline performs the following operations automatically:

1. Missing value imputation.
2. Encoding of categorical variables.
3. Feature scaling when required.
4. Delivery of the processed data to the machine learning model.

Using a pipeline also makes the project cleaner, easier to maintain, and easier to reuse with new datasets.

---

# Encoding Categorical Variables

Machine learning models cannot directly process text values such as:

- Apartment
- House
- Brussels
- To renovate

These categories must first be converted into numerical values.

Depending on the algorithm, encoding methods such as **One-Hot Encoding** or **Ordinal Encoding** were used.

This transformation allows the models to learn patterns from categorical information while keeping the preprocessing fully reproducible.

---

# Feature Scaling

Some machine learning algorithms are sensitive to differences in feature magnitude.

For example:

| Feature | Example Value |
|----------|--------------:|
| Bedrooms | 3 |
| Living Surface | 180 |
| Postal Code | 4000 |

Without scaling, variables with larger numerical ranges may dominate the learning process.

For models that require normalization, numerical variables were standardized using **StandardScaler**.

Standardization transforms each feature to have:

- a mean close to **0**
- a standard deviation close to **1**

This improves optimization, accelerates convergence, and ensures that all numerical features contribute fairly during training.

---

# Why Tree-Based Models Do Not Require Scaling

Not every machine learning algorithm benefits from feature scaling.

Tree-based methods—including **Decision Trees**, **Random Forest**, and **XGBoost**—split the data using threshold values.

For example:

```
Living Surface > 160 m²
```

instead of comparing absolute feature magnitudes.

Because these algorithms evaluate each feature independently, scaling does not change the position of the optimal split.

For this reason, tree-based models naturally handle variables with different ranges and generally achieve the same performance with or without normalization.

This is one of the reasons why ensemble tree methods are particularly well suited for structured tabular datasets such as Belgian real estate data.

---

# Why Use a Scikit-Learn Pipeline?

Using a Pipeline provides several important advantages:

- Prevents **data leakage** by fitting preprocessing steps only on the training data.
- Guarantees identical preprocessing during prediction.
- Makes the workflow modular and reusable.
- Simplifies experimentation with different machine learning models.
- Follows industry best practices for production-ready machine learning projects.

Thanks to this design, changing from Linear Regression to Random Forest or XGBoost requires only replacing the model while keeping exactly the same preprocessing workflow.

# Model Training & Evaluation

To predict Belgian property prices, several regression algorithms were trained and evaluated. The goal was not only to obtain the highest prediction accuracy, but also to compare different machine learning approaches and understand their strengths and limitations.

Each model was trained using the same train/test split and evaluated with identical performance metrics to ensure a fair comparison.

---

# Evaluation Metrics

The following metrics were used to evaluate every model:

### R² Score (Coefficient of Determination)

The R² score measures how well the model explains the variability of house prices.

- **1.00** = perfect predictions
- **0.00** = no better than predicting the average price
- Negative values indicate the model performs worse than a simple average.

Higher values indicate better predictive performance.

---

### MAE (Mean Absolute Error)

MAE represents the average absolute difference between the predicted price and the actual selling price.

It is easy to interpret because it is expressed directly in euros.

Lower values indicate better predictions.

---

### RMSE (Root Mean Squared Error)

RMSE measures the average prediction error while giving greater importance to large mistakes.

Because large prediction errors are penalized more heavily, RMSE is particularly useful for evaluating real estate models where a few poor predictions can significantly affect performance.

Again, lower values are better.

---

# Models Tested


## 1. Linear Regression (Baseline)

Linear Regression was used as the baseline model because it is one of the simplest regression algorithms.

It assumes a linear relationship between the input features and the property price.

### Advantages

* Very fast to train.
* Easy to interpret.
* Provides a useful performance baseline.

### Limitations

Belgian real estate prices depend on many complex interactions between variables.

Linear Regression cannot capture these non-linear relationships, making it less suitable for this problem.

### Results

| Metric |       Value |
| ------ | ----------: |
| R²     |  **0.4404** |
| RMSE   | **€290,384** |
| MAE    | **€159,358** |

Although it provides a reasonable starting point, the model struggles to explain price variations and is highly sensitive to outliers.

---

## 2. Random Forest Regressor

Random Forest is an ensemble learning algorithm that builds multiple decision trees and averages their predictions.

### Advantages

* Captures non-linear relationships.
* Handles feature interactions automatically.
* Robust to outliers.
* Works well on tabular data without heavy preprocessing.

### Results

| Metric |        Value |
| ------ | -----------: |
| R²     |   **0.7283** |
| RMSE   | **€202,930** |
| MAE    |  **€89,471** |

Random Forest shows strong predictive power and serves as a solid benchmark
model, though XGBoost beats it on every metric here.

---

## 3. Support Vector Regression (SVR)

### Linear SVR

| Metric |         Value |
| ------ | ------------: |
| R²     |  **−83.1243** |
| RMSE   | **€3,570,956** |
| MAE    |   **€193,492** |

The model failed to converge properly and performed very poorly, indicating misconfiguration and incompatibility with the dataset.

---

### Non-Linear SVR (RBF + SVD)

| Metric |         Value |
| ------ | ------------: |
| R²     |    **0.4456** |
| RMSE   |   **€289,885** |
| MAE    |   **€128,259** |

Despite dimensionality reduction (SVD) and tuning, the model remains far weaker
than tree-based approaches - and, once the price-derived column is dropped, no
better than plain Linear Regression (0.4456 against 0.4404).

---

## 4. k-Nearest Neighbors (kNN)

kNN regression predicts values based on the average of the nearest observations in feature space.

Best configuration:

* `n_neighbors = 15`
* `weights = distance`
* `metric = euclidean`
* log-transform applied on target variable

### Results

| Metric |         Value |
| ------ | ------------: |
| R²     |    **0.6546** |
| RMSE   |   **€228,816** |

Although functional, kNN struggles with high-dimensional, non-linear relationships typical of real estate pricing.

Its distance-based logic is not robust enough for this dataset, leading to its rejection.

---

## 5. XGBoost

XGBoost (Extreme Gradient Boosting) is a powerful ensemble method that builds trees sequentially, each correcting previous errors.

### Tuned Model Results

Hyperparameters:

```text
subsample = 0.9
n_estimators = 500
max_depth = 7
learning_rate = 0.1
colsample_bytree = 0.8
```

Performance:

| Metric |         Value |
| ------ | ------------: |
| R²     |    **0.7654** |
| RMSE   | **≈ €66,237** |
| MAE    | **≈ €47,521** |

This tuned version represents one of the strongest configurations obtained during experimentation.

---

### Final Pipeline Evaluation (train.py)

When evaluated through the full preprocessing + training pipeline:

Measured on the 3,150 held-out rows (20% split, `random_state=42`), mean price
€410,285. Every figure is on the euro scale:

| Metric |          Value |
| ------ | -------------: |
| R²     |     **0.7650** |
| RMSE   | **≈ €188,753** |
| MAE    |  **≈ €85,768** |
| MAPE   |     **20.90 %** |
| median APE | **15.10 %** |

`train.py` also prints **R² 0.8332 on the `log1p` scale**. That one is a
diagnostic, not a headline: the model optimises log price, and `log1p`
compresses the expensive tail where this model makes its largest errors, so
less variance is left to explain and the figure flatters the fit. It must never
sit in a table beside a euro MAE and RMSE.

> Measuring this settled a question this README used to leave open. The 0.7663
> printed before was assumed to be the log-scale number and expected to drop
> once measured in euros; it did not. The euro R² is 0.7650 — the same figure
> within run-to-run noise — and it is the LOG scale that is higher, at 0.8332.
> So the old R² was already comparable with the MAE and RMSE beside it.

**Read MAPE with its median.** 20.90% is pulled up by the cheap end of the
market: a €30k error is 30% of a €100k flat and 3% of a €1M villa. The median
absolute percentage error, 15.10%, is what a typical listing sees. 36.1% of
held-out properties are predicted within 10% of their price, and 62.9% within
20%.

---

## Model Comparison

| Model                                      |         R² |        RMSE |        MAE | Conclusion                |
| ------------------------------------------ | ---------: | ----------: | ---------: | ------------------------- |
| Linear Regression                          |     0.4404 |    €290,384 |   €159,358 | Weak baseline             |
| Non-Linear SVR (RBF)                       |     0.4456 |    €289,885 |   €128,259 | No better than linear     |
| kNN (`cross_validation_kNN_XGBoost.py`)    |     0.6546 |    €228,816 |          — | Weak                      |
| Random Forest                              |     0.7283 |    €202,930 |    €89,471 | Strong benchmark          |
| XGBoost (`src/train.py` pipeline)          |     0.7650 |    €188,753 |    €85,768 | Best MAE                  |
| XGBoost (`models/XGBoost.py`)              |     0.7835 |    €181,141 |    €86,440 | Same model, fewer columns |
| XGBoost tuned (`models/tuning_XGBoost.py`) | **0.7908** | **€178,057** |   €89,256 | Best final model          |
| Linear SVR                                 |   −83.1243 |  €3,570,956 |   €193,492 | Failed to converge        |

Every row comes from one run of that script on `data/clean/cleaned_data.json`
(15,746 rows, the same 80/20 split, 3,150 held-out rows), with `price_per_m2`
dropped everywhere. So the table compares models, not preprocessing: R², RMSE and
MAE are all on the euro scale and all rank the same way.

Read RMSE and MAE together. XGBoost tuned wins on R² and RMSE but has a *worse*
MAE than the plain pipeline (€89,256 vs €85,768): tuning bought accuracy on the
expensive properties that dominate the squared error, and paid for it on the
typical listing. `src/train.py` is the better choice if the median property
matters more than the tail.

> An earlier version of this table reported much smaller errors for Random
> Forest (~€82.5k RMSE), SVR (~€101.6k) and kNN (~€89k) alongside a similar R².
> Those pairs are not mathematically consistent with each other on one dataset,
> and they do not reproduce: they were measured on a different, smaller extract.
> The figures above replace them.

---

## Why Some Models Were Rejected

### Linear Regression

Too simplistic to capture interactions between variables influencing real estate prices.

### SVR

* Linear SVR failed to converge
* RBF SVR improved but remained less accurate and computationally expensive

### kNN

Distance-based method struggles in high-dimensional feature space and cannot model complex interactions.

---

## Why Tree-Based Models Performed Better

Real estate pricing depends on non-linear interactions:

* location + surface area interaction
* property condition + size effects
* categorical + numerical feature dependencies

Tree-based models naturally capture these patterns without explicit feature engineering.

---

## Final Model Choice: XGBoost

XGBoost was selected as the final model because it offers:

* strong predictive performance (best pipeline result)
* robustness across different preprocessing conditions
* built-in regularization (reduced overfitting risk)
* scalability and production readiness
* stable performance across tuning and pipeline evaluation

Even though Random Forest achieved competitive results, XGBoost was retained as the final model due to its better generalization behavior and suitability for deployment pipelines.


---

# Overfitting Analysis

To verify that the selected model generalizes well, performance should always be compared on both the training and testing datasets.

A large gap between the two scores would indicate overfitting.

Using Scikit-Learn Pipelines and separating preprocessing from evaluation helps ensure that the test data remains completely unseen during training, producing reliable and unbiased performance estimates.

Overall, the tree-based models demonstrated excellent generalization and significantly outperformed the simpler regression approaches.

# Saving the Models

Once each model was trained and evaluated, it was saved using **Joblib**. Saving trained models avoids retraining them every time a prediction is needed and makes the project reusable in a production environment.

The following artifacts are written to the `models/` directory when the
corresponding script is run:

- `linear_regression.joblib`
- `random_forest.joblib`
- `xgboost.joblib`
- `linear_svr.joblib`
- `non_linear_svr.joblib`
- `best_knn.joblib`
- `best_xgboost.joblib` (the final pipeline, from `src/train.py`)

Each file holds a full scikit-learn `Pipeline`, preprocessing included, so new
data goes through exactly the same transformations as the training data. There
is no separate `preprocessing_pipeline.joblib`. These files are not versioned
(see `.gitignore`): re-run the script that produces the one you need.

Example:

```python
import joblib

joblib.dump(model, "models/best_xgboost.joblib")
```

---

# Predicting New Properties

After training, the saved model can estimate the price of a new property using unseen data.

The prediction workflow is straightforward:

1. Load the preprocessing pipeline.
2. Load the trained model.
3. Create a DataFrame containing the new property's features.
4. Apply the preprocessing pipeline.
5. Generate the price prediction.

Example:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/best_xgboost.joblib")

# Example property
new_property = pd.DataFrame([{
    "postcode": 1000,
    "latitude": 50.85,
    "longitude": 4.35,
    "livable_surface": 140,
    "bedroom_count": 3
    # Add all remaining features used during training
}])

prediction = model.predict(new_property)

print(f"Estimated price: €{prediction[0]:,.0f}")
```

This demonstrates how the trained model can easily be reused to estimate the value of new Belgian properties.
I've tried with a real propertiy, I've a result of ~222k€ and on the site it is 199k€.

---

# Technologies Used

This project was developed using the following Python libraries:

| Library | Purpose |
|---------|---------|
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-Learn | Preprocessing, pipelines and machine learning |
| XGBoost | Gradient Boosting model |
| Joblib | Saving and loading trained models |

---

# Repository Workflow

The complete machine learning workflow follows these steps:

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Selection
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Train / Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Model Comparison
      │
      ▼
Save Best Model
      │
      ▼
Predict New Properties
```

Using this modular workflow makes the project reproducible, maintainable, and easy to extend.

---

# Key Takeaways

Throughout this project, several important machine learning concepts were applied:

- Building reusable Scikit-Learn Pipelines.
- Preventing data leakage through proper preprocessing.
- Handling missing values using imputation.
- Encoding categorical variables.
- Comparing multiple regression algorithms.
- Evaluating models using several performance metrics.
- Saving trained models for future use.
- Predicting house prices from unseen data.

This project demonstrates the complete workflow of a supervised machine learning regression task, from raw data preparation to model deployment.

---

# Conclusion

The objective of this project was to build a machine learning model capable of accurately predicting Belgian property prices.

Several regression algorithms were trained and compared, including **Linear Regression**, **Support Vector Regression**, **Random Forest**, and **XGBoost**.

The results clearly show that **tree-based ensemble methods** outperform traditional linear models on structured real estate data. Their ability to model complex, non-linear relationships between property characteristics leads to significantly better predictions.

Among the tested algorithms, XGBoost was selected as the final model. On the
same 3,150 held-out rows, the two XGBoost scripts now agree:

| | `src/train.py` (pipeline) | `models/tuning_XGBoost.py` (tuned) |
|---|---:|---:|
| R² (euros) | 0.7650 | **0.7908** |
| RMSE | €188,753 | **€178,057** |
| MAE | €85,768 | €89,256 |
| MAPE | **20.90%** | 24.70% |
| median APE | **15.10%** | 15.41% |

They did not agree before, and reconciling them found a bug worth stating. The
tuned script picked its features with `select_dtypes`, which swallowed every
numeric column — including `price_per_m2`, which `cleaning.py` writes into
`cleaned_data.json` as `price / livable_surface`. With `livable_surface` also a
feature, the model could recover the answer as `price_per_m2 * livable_surface`,
and scored R² 0.9066 with a 4.03% median error by doing so. On a real listing
`price_per_m2` cannot be known without the price, so those figures measured
arithmetic, not prediction. Dropping it costs R² 0.9066 → 0.7908 and median APE
4.03% → 15.41%. `src/train.py` was never affected: it names its columns.

**Every per-model script now carries the same guard.** `XGBoost.py`,
`Random_Forest.py`, `Ramdom_Forest.py`, `linear_svr.py`, `linear_SVM.py`,
`non_linear_svr.py`, `cross_validation_kNN_XGBoost.py` and
`overfitting_XGBoost.py` all selected features with `select_dtypes` and none of
them dropped the price-derived column; each one now does, and prints the column
it dropped so a run says out loud what it removed. The drop is by derivation
(any column whose name contains `price`), not by naming `price_per_m2`, so a
future price-derived column cannot quietly reintroduce the leak. The Model
Comparison table above was rebuilt from their output after the fix.

`Ramdom_Forest.py` is a byte-identical copy of `Random_Forest.py`, and
`linear_SVM.py` of `linear_svr.py` — the duplicates are kept in place but only
one of each was run for the table.

Beyond these metrics, XGBoost offers an ideal balance of robustness,
scalability, and resistance to overfitting, making it the most reliable solution
for this tabular dataset compared to the other architectures tested.

This project also highlights the importance of reproducible preprocessing, modular code organization, and rigorous model evaluation. By combining Scikit-Learn Pipelines with modern ensemble learning techniques, the resulting workflow is accurate, reusable, and aligned with industry best practices.

---

# Author

**Siegried Camus**

Machine Learning Project completed as part of the **BeCode AI & Data Science Bootcamp**.
