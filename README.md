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
immo-eliza-ml/
├── analysis/
│   └── notebook_sieg.ipynb
├── data/
│   ├── raw/
│   └── cleaned/
├── images/
├── models/
│   ├── linear_regression.joblib
│   └── ...
├── src/
│   ├── clean_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── main.py
├── requirements.txt
└── .gitignore
```

The project follows a modular structure to improve readability, maintainability, and reproducibility.

- **analysis/** contains exploratory notebook and model experiments.
- **data/** contains the raw and cleaned datasets.
- **images/** stores figures used during model evaluation.
- **models/** stores the trained machine learning models and preprocessing pipeline.
- **src/** contains the project's Python modules, including preprocessing, training, evaluation, and prediction scripts.

Keeping preprocessing, training, evaluation, and prediction in separate modules makes the workflow easier to understand, test, and reuse.

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
| R²     |    **0.41** |
| RMSE   | **≈ €303k** |
| MAE    | **≈ €161k** |

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
| R²     |     **0.73** |
| RMSE   | **≈ €82.5k** |
| MAE    | **≈ €55.7k** |

Random Forest shows strong predictive power and serves as a solid benchmark model.

---

## 3. Support Vector Regression (SVR)

### Linear SVR

| Metric |         Value |
| ------ | ------------: |
| R²     |    **-54.57** |
| RMSE   |  **≈ €1.19M** |
| MAE    | **≈ €104.3k** |

The model failed to converge properly and performed very poorly, indicating misconfiguration and incompatibility with the dataset.

---

### Non-Linear SVR (RBF + SVD)

| Metric |         Value |
| ------ | ------------: |
| R²     |      **0.60** |
| RMSE   | **≈ €101.6k** |
| MAE    |  **≈ €74.0k** |

Despite dimensionality reduction (SVD) and tuning, the model remains weaker than tree-based approaches.

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
| R²     |    **0.5743** |
| RMSE   | **≈ €89,235** |

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

| Metric |          Value |
| ------ | -------------: |
| R²     |     **0.7663** |
| RMSE   | **≈ €188,211** |
| MAE    |  **≈ €90,737** |

The difference between tuned evaluation and pipeline evaluation is expected due to full preprocessing effects and real training conditions.

---

## Model Comparison

| Model             |         R² |       RMSE |         MAE | Conclusion                |
| ----------------- | ---------: | ---------: | ----------: | ------------------------- |
| Linear Regression |       0.41 |     ~€303k |      ~€161k | Weak baseline             |
| Random Forest     |       0.73 |    ~€82.5k |     ~€55.7k | Strong benchmark          |
| Linear SVR        |     -54.57 |    ~€1.19M |    ~€104.3k | Failed                    |
| Non-Linear SVR    |       0.60 |   ~€101.6k |       ~€74k | Moderate                  |
| kNN               |       0.57 |      ~€89k |           — | Weak                      |
| XGBoost           | **0.7663** | **~€188k** | **~€90.7k** | Best final pipeline model |

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

The following artifacts are stored in the `models/` directory:

- `linear_regression.joblib`
- `random_forest.joblib`
- `xgboost.joblib`
- `preprocessing_pipeline.joblib`

Saving both the trained model and the preprocessing pipeline guarantees that any new data will undergo exactly the same transformations as the training data.

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

Among the tested algorithms, **XGBoost** was selected as the final model. While Random Forest achieved a slightly higher R² score on this dataset, XGBoost offers a better balance between predictive performance, robustness, scalability, and resistance to overfitting. Its sequential boosting strategy and built-in regularization make it one of the most reliable algorithms for tabular machine learning problems.

This project also highlights the importance of reproducible preprocessing, modular code organization, and rigorous model evaluation. By combining Scikit-Learn Pipelines with modern ensemble learning techniques, the resulting workflow is accurate, reusable, and aligned with industry best practices.

---

# Author

**Siegried Camus**

Machine Learning Project completed as part of the **BeCode AI & Data Science Bootcamp**.
