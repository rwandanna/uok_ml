## MIT91207 — 09 Machine Learning


> **Course:** MIT91207 — Machine Learning  
> **Assignment:** Assignment 1
>
> **Note:** The answers below are written for direct submission. Code is included where requested. Numerical results are reported only where they can be supported by the supplied data.

---

# Question 1 — Early-Warning System for At-Risk Students

## (a) Problem formulation

The university's problem can be formulated as a **supervised machine-learning prediction problem**. Student information such as attendance, previous grades, assessment scores, extracurricular participation and LMS engagement can be used as input features \(X\) to predict an academic outcome \(y\).

If the target is defined as **at-risk/not-at-risk**, or as categories such as low, medium and high risk, the problem is a **classification problem** because the target is categorical.

If the target is defined as a continuous value such as final GPA or a future assessment score, the problem is a **regression problem** because the target is numerical and continuous.

For the university's main decision — identifying students who may need academic intervention — classification is the most direct formulation. A classification model such as logistic regression can estimate the probability that a student is at risk:

\[
P(\text{at risk}\mid X)
\]

The university can then select an intervention threshold and refer students whose predicted risk exceeds that threshold.

Regression can also be useful if the university wants to predict a continuous academic outcome, such as future GPA, and identify students whose predicted outcome falls below an intervention threshold.

Therefore:

- **Classification:** predicts a categorical risk outcome.
- **Regression:** predicts a continuous academic outcome.
- **Early intervention:** can use predicted risk probabilities and an appropriate decision threshold.

---

## (b) Two modelling approaches

### 1. Classification — Logistic Regression

Logistic regression can use attendance, previous grades, assessment scores, extracurricular participation and LMS engagement to predict the probability that a student is at risk.

The output is:

\[
P(\text{at risk}\mid X)
\]

A threshold can then be applied. For example, students whose predicted risk exceeds the university's chosen threshold can be referred to academic advisors.

### 2. Regression — Ridge Regression

Ridge regression can use the same or similar features to predict a continuous academic outcome, such as expected final GPA or a future assessment score.

Students whose predicted outcome falls below a predefined academic threshold can then be identified for intervention.

Thus, logistic regression directly predicts academic risk, while Ridge regression predicts a continuous academic outcome from which risk can subsequently be determined.

---

## (c) Two factors beyond predictive accuracy

### 1. Fairness and bias

The model should be evaluated for systematic differences in performance across relevant student groups. Historical data may contain existing inequalities, which can cause the model to reproduce or amplify them.

### 2. Interpretability and actionability

Academic advisors should be able to understand the factors contributing to a student's risk prediction. This helps them decide what intervention is appropriate and supports responsible use of the model.

---

# Question 2 — Course-Recommendation Preprocessing

## (a) Preprocessing pipeline

### 1. Missing ratings

The meaning of a missing rating should first be established. In a recommendation system, a missing rating often means that a learner did not provide an explicit rating; it does not necessarily mean that the learner gave the course an average or zero rating.

Therefore, missing ratings should not automatically be replaced with zero or the global mean. A missingness indicator can be created, and if a rating-prediction model requires numerical imputation, the imputation value should be learned from the training data only.

Behavioural information such as course views and completion should be retained as separate signals.

### 2. Duplicated records

Duplicate learner-course interactions should be identified using an appropriate key, such as learner ID, course ID and timestamp. Genuine duplicates should be removed. Where repeated records represent updates to the same interaction, the valid or latest record should be retained according to the data definition.

This prevents a single interaction from being counted multiple times.

### 3. Inconsistent category names

Category labels should be standardised by removing unnecessary whitespace, applying consistent case and mapping equivalent labels to a canonical representation.

For example:

- `Data Science`
- `data science`
- `Data-Science`

should be mapped consistently to one category.

### 4. Different numerical scales

Numerical variables such as previous courses taken and engagement measures may have very different ranges. For algorithms that are sensitive to feature scale, such as k-nearest neighbours and many regularised models, numerical features should be standardised.

Scaling prevents variables with larger numerical ranges from disproportionately influencing the model.

---

## (b) Python/pandas implementation

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "learner_id": [1, 1, 2, 3, 3],
    "course_title": [
        "Intro to Python",
        "Intro to Python",
        "Data Science",
        "Data Science",
        "SQL Basics"
    ],
    "category": [
        "Programming",
        "Programming",
        "Data Science ",
        " data science",
        "Databases"
    ],
    "rating": [4.5, 4.5, None, 3.0, None],
    "courses_taken": [3, 3, 12, 12, 1]
})

# 1. Remove duplicate learner-course records
df = df.drop_duplicates(
    subset=["learner_id", "course_title"],
    keep="last"
)

# 2. Standardise category names
df["category"] = (
    df["category"]
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# 3. Create a missing-rating indicator
df["rating_missing"] = df["rating"].isna().astype(int)

# 4. Example imputation if a numerical rating model requires it
df["rating"] = df["rating"].fillna(df["rating"].median())

# 5. Scale a numerical feature
scaler = StandardScaler()

df["courses_taken_scaled"] = scaler.fit_transform(
    df[["courses_taken"]]
)

print(df)
```

In a real machine-learning workflow, the scaler and imputation statistics should be fitted using training data only and then applied to validation, test and future records.

---

## (c) Two engineered features

### 1. Course-completion rate

\[
\text{Completion Rate}
=
\frac{\text{Courses Completed}}
{\text{Courses Started}}
\]

This captures learner commitment and follow-through better than simply counting the number of courses started.

### 2. Recency-weighted engagement

Recent learner activity can be given greater weight than older activity, for example using days since the last interaction or exponentially decreasing weights.

This captures current learner interests because preferences can change over time.

---

## (d) Preventing temporal leakage — [1 Mark]

Features must be calculated using only information that was available **before the prediction time**.

For example, when predicting a learner's future course preference, interactions that occurred after the prediction date must not be included in the learner's historical features.

For timestamped recommendation data, chronological training, validation and test periods are therefore preferable to randomly mixing past and future interactions.

---

# Question 3 — Missing Data in Loan-Default Prediction

## (a) Critique of zero-imputation

Replacing every missing numerical value with zero is inappropriate because **zero is often a meaningful financial value**, whereas a missing value means that the actual value is unknown.

For example, if income is missing and is replaced by zero, the model interprets the applicant as having no income rather than as having an unknown income.

Similarly, zero missed payments may represent a genuinely clean repayment history. If missing repayment history is also replaced with zero, the model cannot distinguish between an applicant with no missed payments and an applicant whose repayment history is unavailable.

Zero-imputation can therefore:

1. create an artificial concentration of observations at zero;
2. distort the distribution of the variable;
3. distort relationships between the variable and loan default;
4. introduce systematic prediction errors;
5. potentially produce unfair lending decisions.

The key distinction is:

\[
\text{Missing} \neq \text{Zero}
\]

---

## (b) Improved missing-data strategy

The strategy should depend on both the variable type and the reason for missingness.

For numerical variables such as income, **median imputation** can be used where appropriate, preferably with the statistic calculated from the training data. A missingness indicator can also be added when the fact that a value is missing may itself contain predictive information.

For categorical variables such as employment information, missing values can be represented using a separate category such as **`Unknown`**.

If the missingness mechanism is systematic, simple imputation should be treated cautiously because the fact that information is missing may be related to the unobserved value.

---

## (c) Python function

```python
import pandas as pd

def handle_missing_financial(
    df,
    numeric_cols,
    categorical_cols,
    group_col=None
):
    df = df.copy()

    # Numerical variables
    for col in numeric_cols:
        # Preserve information about missingness
        df[f"{col}_missing"] = df[col].isna().astype(int)

        # Optional group-based median imputation
        if group_col is not None:
            group_median = df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = group_median

        # Fallback to overall median
        df[col] = df[col].fillna(df[col].median())

    # Categorical variables
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df
```

For production use, the imputation values should be learned from the training set and then applied unchanged to validation, test and future data.

---

## (d) Consequences for model performance and fairness

Inappropriate missing-data treatment can distort the relationships learned by the model and reduce predictive performance, including discrimination and calibration.

It can also affect applicants unfairly. For example, treating missing income as zero may cause an applicant to receive an unnecessarily high estimated default risk.

Therefore, poor missing-data treatment can create both **statistical errors and unfair lending decisions**.

---

# Question 4 — Feature Engineering for Subscription Renewal

## (a) Feature-engineering pipeline

For numerical variables such as age, monthly expenditure, previous transactions and account duration, I would standardise the variables when using scale-sensitive algorithms such as logistic regression, SVM or k-nearest neighbours.

For categorical variables, the encoding method should depend on whether the variable is nominal or genuinely ordinal:

- **Nominal variables**, such as geographical region and payment method, should normally use one-hot encoding.
- A variable such as subscription tier should use ordinal encoding only if its categories have a genuine order, such as Basic < Standard < Premium.

The preprocessing should be fitted on the training data and incorporated into a reproducible pipeline.

---

## (b) Encoding consequences

### One-hot encoding

One-hot encoding is appropriate for nominal categorical variables with no meaningful order. Each category is represented by a binary feature.

Its main advantage is that it does not impose an artificial numerical relationship between categories. A disadvantage is that it can increase the number of features when a categorical variable has many unique categories.

### Ordinal encoding

Ordinal encoding is appropriate when categories have a genuine order. For example:

\[
\text{Basic} < \text{Standard} < \text{Premium}
\]

The numerical representation preserves that ordering.

### Inappropriate numerical encoding

Assigning arbitrary numbers such as:

\[
\text{Kigali}=1,\quad
\text{Musanze}=2,\quad
\text{Huye}=3
\]

to a nominal variable is generally inappropriate because it creates a false ordering and numerical distance between categories.

---

## (c) Python implementation

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.DataFrame({
    "age": [25, 41, 33, 52],
    "monthly_expenditure": [45.0, 120.5, 78.2, 210.0],
    "previous_transactions": [3, 22, 11, 40],
    "account_duration_months": [6, 34, 15, 60],
    "subscription_type": [
        "basic", "premium", "standard", "premium"
    ],
    "region": [
        "Kigali", "Musanze", "Huye", "Kigali"
    ],
    "payment_method": [
        "mobile_money", "card", "mobile_money", "card"
    ]
})

numeric_cols = [
    "age",
    "monthly_expenditure",
    "previous_transactions",
    "account_duration_months"
]

categorical_cols = [
    "subscription_type",
    "region",
    "payment_method"
]

preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numeric_cols
    ),
    (
        "cat",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ),
        categorical_cols
    )
])

pipeline = Pipeline([
    ("preprocessor", preprocessor)
])

X_ready = pipeline.fit_transform(df)

print(X_ready)
```

---

## (d) Ensuring deployment consistency

The preprocessing transformations should be fitted on the training data and saved as part of the same pipeline as the model.

The same fitted scaler and encoder must then be used for validation, testing and newly arriving customer records. They should not be independently refitted on new data.

For categorical variables, `handle_unknown="ignore"` can prevent the pipeline from failing when a previously unseen category appears.

This ensures that the model receives the same feature representation during training and deployment.

---

# Question 5 — Hospital Readmission Prediction

## (a) End-to-end ML workflow

A suitable machine-learning workflow is:

### 1. Data acquisition

Collect historical patient, admission, diagnosis and treatment data under appropriate privacy and data-governance controls.

### 2. Data cleaning and integration

Remove duplicates, resolve inconsistent records, standardise variables and handle missing values.

### 3. Exploratory data analysis

Examine distributions, missingness, readmission rates and important patient subgroups.

### 4. Feature engineering

Create variables such as previous admission count, number of comorbidities and time since previous discharge, using only information available at the prediction time.

### 5. Data partitioning

Separate the data into training, validation and test sets using a strategy that reflects deployment conditions.

### 6. Model training and tuning

Train candidate models on the training data and tune their hyperparameters using validation data or cross-validation.

### 7. Evaluation

Evaluate the final model on an untouched test set using appropriate predictive metrics and subgroup analysis.

### 8. Deployment and monitoring

Integrate the model into the hospital information system, monitor model performance and data drift, and retain appropriate human oversight because the prediction supports rather than replaces clinical judgement.

---

## (b) Data partitioning

The data should preferably be split **chronologically**, with earlier admissions used for training and later admissions used for validation and testing. This better represents the real deployment situation in which the model predicts future admissions.

Where patients have multiple admissions, records should also be **grouped by patient** so that the same patient does not appear in both training and evaluation sets when evaluating generalisation to unseen patients.

A careless random row-level split can allow information from the same patient to appear in both training and test data, producing overly optimistic performance estimates.

---

## (c) High overall accuracy but subgroup differences

High overall accuracy does not guarantee reliable performance for every patient group.

The model should therefore be evaluated separately across relevant subgroups using measures such as:

- recall and false-negative rate;
- precision;
- ROC-AUC or precision-recall measures where appropriate;
- calibration;
- confusion matrices.

If substantial disparities are found, the team should investigate data quality, target definition, feature representation and model design before deployment.

The appropriate metric depends on the consequences of false positives and false negatives. For readmission prediction, missing a genuinely high-risk patient may be particularly important, so recall and false-negative rate deserve careful attention.

---

## (d) Python partitioning

```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Sort by admission time
df = df.sort_values("admission_date")

# Reserve the most recent 20% of the data as a future test set
cutoff = df["admission_date"].quantile(0.80)

train_val = df[df["admission_date"] <= cutoff]
test = df[df["admission_date"] > cutoff]

# Split the earlier data by patient
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    gss.split(
        train_val,
        groups=train_val["patient_id"]
    )
)

train = train_val.iloc[train_idx]
validation = train_val.iloc[val_idx]

print("Training:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)
```

This approach combines a future holdout with patient-level grouping.

---

# Question 6 — Crime-Rate Prediction

## (a) Feature-selection strategy

I would use the following feature-selection procedure:

1. Inspect data types, missing values, duplicate observations and identifier variables.
2. Remove variables that are identifiers, constant or contain no useful predictive information.
3. Examine relationships between each predictor and `Crime`, including correlations for numerical variables.
4. Examine correlations between predictors to identify multicollinearity and redundant variables.
5. Retain variables that provide useful predictive information while avoiding unnecessary redundancy.
6. Use a multivariate model such as regularised regression and cross-validation to confirm whether selected variables improve predictive performance.
7. Avoid selecting variables based only on correlation because correlation measures bivariate association and does not establish causation.

The key principles are:

\[
\text{Relevance} + \text{Redundancy} + \text{Validation}
\]

---

## (b) Exploratory analysis and correlation matrix

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("UScrime.csv")

# Basic data inspection
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Correlations with Crime
print("\nCorrelations with Crime:")
print(
    corr_matrix["Crime"]
    .sort_values(ascending=False)
)

# Visualise correlation matrix
plt.figure(figsize=(10, 8))

plt.imshow(
    corr_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()
```

The correlation matrix can be used to identify variables that have relatively strong positive or negative associations with `Crime`.

It should also be used to identify highly correlated predictors. For example, `Po1` and `Po2` are strongly correlated in the well-known UScrime dataset. Keeping both may therefore introduce multicollinearity.

However, correlation should not be used as the only feature-selection criterion because:

1. it primarily measures pairwise linear association;
2. a weakly correlated variable may become useful in combination with other variables;
3. highly correlated predictors may contain redundant information;
4. correlation does not establish causation.

---

## (c) Multicollinearity

Highly correlated predictors can cause **multicollinearity**.

For example, if `Po1` and `Po2` contain almost the same information, an OLS model may have difficulty distinguishing their individual effects.

Multicollinearity can produce:

- unstable coefficient estimates;
- inflated coefficient uncertainty;
- coefficients that change substantially with small changes in the data.

Possible solutions include:

1. removing one of the highly correlated variables;
2. combining them into a meaningful composite variable;
3. using Ridge regression to stabilise coefficient estimates.

---

## (d) Government recommendations

Variables that are statistically associated with crime can be used as **predictive indicators** for identifying areas that may require further analysis or resource planning.

However, these associations should not automatically be interpreted as causal relationships.

For example, a positive association between police expenditure and crime does not prove that increased police expenditure causes higher crime. Areas experiencing more crime may increase police spending, creating reverse causality. Other factors may also affect both variables.

Therefore, predictive associations can support planning and prioritisation, but causal policy conclusions require stronger evidence such as longitudinal studies, controlled comparisons or other causal research designs.

---

# Question 7 — ETA Prediction for YEGO

## (a) Experimental procedure

I would compare OLS, Ridge and Lasso using the same experimental procedure:

1. Inspect the dataset and remove identifiers that should not be predictive.
2. Separate predictors \(X\) from the target `eta_minutes`.
3. Split the data into training and test sets while keeping the test set untouched.
4. One-hot encode the categorical `road_type` variable.
5. Standardise numerical variables because Ridge and Lasso are scale-sensitive.
6. Train OLS, Ridge and Lasso using the same training data and preprocessing.
7. Tune Ridge and Lasso's regularisation parameter \(\alpha\) using cross-validation on the training data.
8. Evaluate the final models on the untouched test set using MAE, RMSE and \(R^2\).
9. Compare predictive performance and model complexity.

---

## (b) Python implementation and results

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load data
df = pd.read_csv("synthetic_ETA.csv")

# Remove identifier
df = df.drop(columns=["observation_id"])

# Separate features and target
X = df.drop(columns=["eta_minutes"])
y = df["eta_minutes"]

# Identify column types
num_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = X.select_dtypes(
    exclude=np.number
).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

def make_preprocessor():
    return ColumnTransformer([
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            cat_cols
        )
    ])

# Cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# OLS
ols = Pipeline([
    ("prep", make_preprocessor()),
    ("model", LinearRegression())
])

ols.fit(X_train, y_train)

# Ridge
ridge_search = GridSearchCV(
    Pipeline([
        ("prep", make_preprocessor()),
        ("model", Ridge())
    ]),
    {
        "model__alpha": [
            0.001, 0.01, 0.1, 0.5,
            1, 2, 5, 10, 20, 50, 100
        ]
    },
    cv=cv,
    scoring="r2"
)

ridge_search.fit(X_train, y_train)

# Lasso
lasso_search = GridSearchCV(
    Pipeline([
        ("prep", make_preprocessor()),
        ("model", Lasso(max_iter=20000))
    ]),
    {
        "model__alpha": [
            0.001, 0.005, 0.01, 0.05,
            0.1, 0.2, 0.5, 1, 2, 5
        ]
    },
    cv=cv,
    scoring="r2"
)

lasso_search.fit(X_train, y_train)

# Final models
models = [
    ("OLS", ols),
    ("Ridge", ridge_search.best_estimator_),
    ("Lasso", lasso_search.best_estimator_)
]

# Evaluate on untouched test set
results = []

for name, model in models:

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

print(results_df)

print(
    "Best Ridge alpha:",
    ridge_search.best_params_
)

print(
    "Best Lasso alpha:",
    lasso_search.best_params_
)
```

### Test-set results

| Model |   MAE |  RMSE |    R² |
| ----- | ----: | ----: | ----: |
| OLS   | 4.380 | 6.237 | 0.800 |
| Ridge | 4.378 | 6.235 | 0.800 |
| Lasso | 4.358 | 6.226 | 0.801 |

The selected regularisation parameters are:

- Ridge: \(\alpha = 1.0\)
- Lasso: \(\alpha = 0.05\)

The three models therefore have very similar predictive performance.

---

## (c) Model choice

Lasso is a reasonable model choice because it provides predictive performance comparable to OLS and Ridge while also shrinking coefficients and potentially setting some coefficients exactly to zero.

This can produce a simpler model and provide a form of feature selection, which is useful when some predictors are weak or redundant.

However, the numerical performance difference is very small. Therefore, it would be incorrect to claim that Lasso is dramatically more accurate.

The appropriate conclusion is:

> **OLS, Ridge and Lasso perform almost equally, while Lasso provides an additional simplicity and feature-selection advantage.**

---

## (d) Reducing overfitting

### 1. Cross-validation and a strictly held-out test set

Cross-validation should be used on the training data to tune hyperparameters such as \(\alpha\), while the final test set remains untouched until final evaluation.

### 2. Regularisation and evaluation on unseen data

Ridge and Lasso introduce regularisation that constrains model complexity and reduces sensitivity to noise.

The final model should also be evaluated on genuinely new journeys. For time-dependent ETA prediction, chronological or rolling validation may be more realistic than a purely random split because future traffic patterns should not influence evaluation of earlier predictions.

---

# Question 8 — Customer Survey Analysis

## (a) EDA strategy

I would begin with a data-quality assessment by checking:

- number of respondents and variables;
- data types;
- missing responses;
- duplicate records;
- inconsistent category names;
- unusual or invalid values.

Next, I would perform univariate analysis:

- bar charts for categorical variables such as occupation;
- ordered bar charts for ordinal satisfaction or willingness levels;
- histograms or boxplots for numerical variables where appropriate.

I would then perform bivariate and subgroup analysis, examining relationships such as:

- willingness to purchase versus age group;
- willingness to purchase versus previous product use;
- willingness to purchase versus satisfaction;
- willingness to purchase versus expected price range.

Finally, I would compare customer segments to identify groups with substantially different purchase intentions.

---

## (b) Python visualisations

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load survey data
df = pd.read_csv("customer_survey.csv")

# Basic data-quality checks
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())


# 1. Willingness to purchase by age group
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="age_group",
    hue="willingness_to_purchase"
)

plt.title(
    "Willingness to Purchase by Age Group"
)

plt.xlabel("Age Group")
plt.ylabel("Number of Respondents")

plt.tight_layout()
plt.show()


# 2. Willingness to purchase by previous product use
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="previous_use",
    hue="willingness_to_purchase"
)

plt.title(
    "Willingness to Purchase by Previous Product Use"
)

plt.xlabel("Previous Product Use")
plt.ylabel("Number of Respondents")

plt.tight_layout()
plt.show()


# 3. Satisfaction by willingness to purchase
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="willingness_to_purchase",
    y="satisfaction_score"
)

plt.title(
    "Satisfaction by Willingness to Purchase"
)

plt.xlabel("Willingness to Purchase")
plt.ylabel("Satisfaction Score")

plt.tight_layout()
plt.show()
```

These visualisations examine three decision-relevant relationships:

1. age group and purchase willingness;
2. previous product use and purchase willingness;
3. satisfaction and purchase willingness.

The specific patterns should be reported only after running the code on the actual survey dataset.

---

## (c) Two insights for management

The analysis should identify at least two decision-relevant insights.

### 1. High-intent customer segments

Identify age, occupation, previous-use or other customer groups with substantially higher willingness to purchase. These segments can inform the initial target market.

### 2. Factors associated with purchase intention

Determine whether satisfaction, expected price range, previous use or another variable is strongly associated with willingness to purchase. This can inform product positioning, pricing and marketing decisions.

The actual numerical segments and values should only be reported after analysing the real survey dataset.

---

## (d) One limitation

A major limitation is that **stated willingness to purchase is not necessarily the same as actual purchasing behaviour**.

Respondents may overstate their intention because the survey does not require them to spend real money.

Therefore, survey findings should ideally be supplemented with evidence such as:

- a small pilot or limited launch;
- real transaction data;
- A/B testing of pricing or product messaging;
- competitor and market analysis.

---

# Conclusion

The eight questions collectively demonstrate the major stages of a machine-learning workflow:

\[
\boxed{
\text{Problem Formulation}
\rightarrow
\text{Preprocessing}
\rightarrow
\text{Feature Engineering}
\rightarrow
\text{Data Partitioning}
\rightarrow
\text{Model Training}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Deployment}
}
\]

The most important principles across the assignment are:

1. **Classification predicts categories; regression predicts continuous values.**
2. **Missing values must not automatically be treated as zero.**
3. **Feature encoding must respect the type and meaning of categorical variables.**
4. **Preprocessing must be fitted on training data and consistently reused at deployment.**
5. **Data leakage must be prevented, especially temporal and patient-level leakage.**
6. **Overall accuracy can hide subgroup performance problems.**
7. **Multicollinearity occurs when predictors are strongly correlated with one another.**
8. **Correlation does not establish causation.**
9. **Ridge uses L2 regularisation; Lasso uses L1 regularisation and can perform feature selection.**
10. **MAE and RMSE are lower-is-better metrics; \(R^2\) is generally higher-is-better.**
11. **Survey intention is weaker evidence than observed purchasing behaviour.**




------------


-----------


Yes. You do **not** need to learn the whole module in 10 minutes. You need a mental map of what each question is testing and a few reusable phrases/code patterns.

Here is your **10-minute emergency sheet**. Read it once from top to bottom, then go back to the questions you find hardest.

---

# 1. THE CORE ML IDEA

Almost every question here follows:

**Data → clean → transform → features → split → train model → predict → evaluate → deploy**

And remember:

### Supervised learning

You have **inputs + known target/output**.

Example:

> Attendance, grades, LMS activity → **at-risk / not-at-risk**

The model learns from examples where the correct answer is already known.

Two major types:

**Classification** → predicts a category.

* Yes/no
* Default/not default
* At risk/not at risk
* Renew/not renew

**Regression** → predicts a number.

* GPA = 2.8
* ETA = 35 minutes
* Crime rate = 450 per 100,000
* House price = $100,000

### Unsupervised learning

There is **no known target**.

The model tries to discover patterns/groups by itself.

Example:

> Customer data → discover groups of similar customers.

Common example: **clustering** such as K-Means.

### Memorize this:

> **Classification = class/category**
> **Regression = number**
> **Unsupervised = no target; find patterns**

---

# 2. LINEAR vs LOGISTIC vs RIDGE vs LASSO

This is extremely important because Q1 and Q7 directly test it.

### Linear Regression

Predicts a **continuous number**.

Example:

> Distance + traffic + weather → ETA = **32.5 minutes**

Basic idea:

> Find a line that best fits the data and use it to predict a number.

Use it for:

* ETA
* GPA
* sales
* price
* temperature

---

### Logistic Regression

Despite having "regression" in its name, it is primarily used for **classification**.

Example:

> Attendance + grades + LMS → probability of being at risk = **0.82**

Then:

> 0.82 > threshold → **At Risk**

Use it for:

* default / no default
* sick / healthy
* churn / no churn
* at-risk / not at-risk

### Memorize:

> **Linear → number**
> **Logistic → category/probability**

---

### Ridge Regression

Ridge is basically **Linear Regression + penalty for large coefficients**.

Why?

Suppose predictors are highly correlated:

> journey distance
> journey duration
> previous journey time

They may contain overlapping information.

Ordinary Linear Regression can become unstable when predictors are highly correlated.

Ridge adds a penalty that **shrinks coefficients toward zero**, making the model more stable and reducing overfitting.

**Important:** Ridge normally does **not** make coefficients exactly zero.

---

### Lasso Regression

Lasso also adds a penalty, but it can shrink some coefficients **exactly to zero**.

Therefore Lasso can perform **feature selection**.

Example:

You have 20 predictors.

Lasso might effectively retain:

> distance, traffic, time of day

while making several unhelpful coefficients zero.

### Memorize this:

> **Ridge = shrink coefficients**
> **Lasso = shrink + can remove features**

And:

> **OLS Linear Regression = ordinary baseline**
> **Ridge = useful when predictors are correlated**
> **Lasso = useful when you want feature selection**

---

# 3. QUESTION 1 — UNIVERSITY

The examiner wants you to recognize **classification vs regression**.

### a)

Inputs:

> attendance, previous grades, assessments, LMS engagement, etc.

Target:

> **At risk / Not at risk**

Therefore:

> **Classification**

But regression is also possible if target is:

> predicted GPA / final mark.

### b)

Classification:

> Logistic Regression → probability of being at risk → intervention if probability exceeds threshold.

Regression:

> Linear Regression → predicted GPA → intervention if predicted GPA is below threshold.

### c)

Don't only say accuracy.

Two easy factors:

**Fairness/bias**

> Model may unfairly disadvantage certain student groups because historical data may contain bias.

**Explainability**

> Advisors need to understand why a student was classified as high risk instead of blindly trusting the prediction.

Memorize:

> **Accuracy tells us whether predictions are correct. Fairness tells us whether decisions are equitable. Explainability tells us whether humans can understand and appropriately use the prediction.**

---

# 4. QUESTION 2 — DATA PREPROCESSING

This question is basically:

> "The dataset is messy. What do you do before ML?"

They give you:

* missing ratings
* duplicates
* inconsistent categories
* different numerical scales

Your pipeline:

**Inspect → clean → handle missing values → remove duplicates → standardize categories → encode categorical variables → scale numerical variables → create useful features → train**

### Missing values

Don't automatically replace everything with zero.

For ratings:

> median/mean/imputation depending on situation.

Could also create:

> `rating_missing = 1`

if missingness itself may contain information.

### Duplicates

Remove duplicate records.

### Inconsistent categories

For example:

> "Computer Science"
> "computer science"
> "Comp Sci"

Standardize them into:

> "Computer Science"

### Different scales

Example:

> Age = 20–60
> Number of courses = 1–100
> rating = 1–5

Use **standardization/scaling**, especially for algorithms sensitive to scale.

### Q2(b) — Python

You don't need complicated code. Know this pattern:

```python
import pandas as pd

df = pd.read_csv("data.csv")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numerical values
df["rating"] = df["rating"].fillna(df["rating"].median())

# Standardize category names
df["category"] = df["category"].str.strip().str.lower()

# Feature engineering
df["completed"] = df["completion_status"].map({"Completed": 1, "Not Completed": 0})
```

That's already several operations.

### Q2(c): useful behavioural features

Think:

**How active is the learner?**

Examples:

> `courses_taken`

> `completion_rate`

> `average_rating`

> `number_of_completed_courses`

Why useful?

Because they represent learner behaviour better than simply using learner ID.

### Q2(d): future information

This is **data leakage**.

Memorize:

> **Training data must only contain information that would have been available at the time the prediction was made.**

With timestamps:

> Train on earlier records, test on later records.

---

# 5. QUESTION 3 — MISSING VALUES

This is mostly testing whether you understand why:

> **missing ≠ zero**

Suppose income is missing.

Zero income means:

> Person earns nothing.

Missing income means:

> We don't know their income.

Those are completely different.

If you replace missing income with zero, the model may think:

> missing-income customer = extremely poor customer.

That can produce wrong credit decisions.

### Better strategy

Numerical:

> median/mean imputation depending on distribution.

Categorical:

> mode or `"Unknown"`.

Important variables:

> investigate why they are missing.

For example:

> Missing credit history could mean the customer has no recorded history, not necessarily bad credit.

### Python function

Something like:

```python
def handle_missing(df):
    numerical = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(exclude="number").columns

    for col in numerical:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical:
        df[col] = df[col].fillna("Unknown")

    return df
```

The assumption:

> Numerical missing values can reasonably be represented by their median, while missing categorical information is represented as `"Unknown"`.

### Q3(d)

Bad missing-value treatment can cause:

**Model problem:**

> biased relationships and reduced predictive performance.

**Decision problem:**

> applicants may incorrectly be approved or rejected.

---

# 6. QUESTION 4 — CATEGORICAL + NUMERICAL

This is one you should memorize.

You have:

### Numerical

* age
* expenditure
* transactions
* account duration

### Categorical

* subscription type
* region
* payment method
* status

You need to transform them.

### Numerical → scaling

Usually:

> StandardScaler

### Categorical → encoding

Usually:

> One-Hot Encoding

For example:

Payment:

> Cash / Card / Mobile

becomes something like:

> Cash = [1,0,0]
> Card = [0,1,0]
> Mobile = [0,0,1]

---

# 7. ONE-HOT vs ORDINAL

Very important.

### One-hot encoding

Use when categories have **no natural order**.

Example:

> Rwanda, Kenya, Uganda

There isn't:

> Rwanda < Kenya < Uganda

So one-hot is appropriate.

### Ordinal encoding

Use when categories have a **real order**.

Example:

> Low < Medium < High

Ordinal encoding:

> Low = 1
> Medium = 2
> High = 3

### Bad numerical encoding

Suppose:

> Rwanda = 1
> Kenya = 2
> Uganda = 3

A model may interpret that as:

> Uganda > Kenya > Rwanda

But those numbers are just labels.

That's why arbitrary numerical encoding can introduce a **false mathematical relationship**.

### Memorize:

> **No order → One-hot**
> **Real order → Ordinal**
> **Arbitrary numbers → dangerous because they imply order**

---

### Q4 Python

Know this pattern:

```python
import pandas as pd

df = pd.DataFrame({
    "age": [20, 30, 40],
    "spending": [100, 200, 150],
    "subscription": ["Basic", "Premium", "Basic"],
    "region": ["North", "South", "North"]
})

df_encoded = pd.get_dummies(
    df,
    columns=["subscription", "region"],
    dtype=int
)

print(df_encoded)
```

If asked about scaling:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_encoded[["age", "spending"]] = scaler.fit_transform(
    df_encoded[["age", "spending"]]
)
```

---

# 8. QUESTION 5 — END-TO-END ML

This is basically asking:

> "Tell me the entire ML process."

Memorize this chain:

**1. Acquire data**

↓

**2. Clean data**

↓

**3. Explore data**

↓

**4. Engineer/select features**

↓

**5. Split data**

↓

**6. Train model**

↓

**7. Validate/tune**

↓

**8. Test**

↓

**9. Evaluate**

↓

**10. Deploy + monitor**

That's your answer.

### Why careless splitting is dangerous

Suppose the same patient appears in both training and test.

The model may effectively have already seen information about that patient.

That's **data leakage**.

Result:

> Test performance looks excellent, but real-world performance is worse.

---

# 9. TRAIN / VALIDATION / TEST

Think:

### Training

> Learn the model.

### Validation

> Choose/tune the model.

### Test

> Final unbiased evaluation.

Typical example:

> 70% train
> 15% validation
> 15% test

Or:

> 80% train / 20% test

with cross-validation inside training.

### Q5(c): accuracy isn't enough

Especially because patient groups perform differently.

Use:

> Precision
> Recall
> F1-score
> Confusion matrix
> Sensitivity/specificity
> Performance by demographic/group

For medical risk prediction, **recall/sensitivity** can be particularly important because missing a genuinely high-risk patient can have serious consequences.

---

### Q5(d) Python

Simple:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps preserve class proportions.

---

# 10. QUESTION 6 — CORRELATION + FEATURE SELECTION

Don't overthink this one.

They want:

> Which variables are useful for predicting crime?

### Feature selection

Look at:

* correlation
* domain knowledge
* missingness
* redundancy
* model-based importance
* statistical tests
* validation performance

Don't blindly keep variables just because correlation is high.

### Correlation means:

> Two variables move together.

It **does NOT prove causation**.

Example:

> Higher police presence may correlate with higher crime.

You cannot conclude:

> Police cause crime.

It could be because:

> High-crime areas receive more police.

This is a classic **association vs causation** issue.

---

# 11. CORRELATION MATRIX

You can write:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("crime.csv")

corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
```

If they ask what you are looking for:

> Strong positive correlation → variables tend to increase together.

> Strong negative correlation → one tends to increase when the other decreases.

> Near zero → weak linear relationship.

But:

> Correlation alone is not sufficient for feature selection because it only measures linear association and does not establish causation or necessarily capture predictive usefulness.

---

# 12. MULTICOLLINEARITY

This is extremely important for Q6 and Q7.

Imagine:

> Income and annual income

or:

> Distance and travel distance

are highly correlated.

They contain similar information.

This can create **multicollinearity**.

For Linear Regression, this can make coefficient estimates:

> unstable / difficult to interpret.

Solutions:

> remove one redundant variable

or

> combine them

or

> use Ridge Regression

or

> use feature-selection methods.

Memorize:

> **Highly correlated predictors → multicollinearity → unstable regression coefficients.**

---

# 13. QUESTION 7 — YEGO ETA

This is the most important model comparison question.

Target:

> ETA = a number.

Therefore:

> **Regression problem.**

Compare:

### OLS Linear Regression

Basic model.

### Ridge

Linear regression + coefficient penalty.

Good when predictors are correlated.

### Lasso

Linear regression + penalty that can set coefficients to zero.

Good when some variables may be unnecessary.

---

## Experimental procedure

You can write:

> First clean the dataset, handle missing values and categorical variables, and scale numerical features where appropriate. Then split the data into training and testing sets. Train OLS, Ridge and Lasso using the same training data. Tune Ridge/Lasso hyperparameters using cross-validation. Evaluate all models on the same unseen test set.

Metrics:

**MAE**

> average absolute prediction error.

If MAE = 4:

> predictions are off by about 4 minutes on average.

**RMSE**

> penalizes larger errors more heavily.

**R²**

> indicates how much variation in ETA is explained by the model.

### Memorize:

> **MAE = average error**
> **RMSE = punishes big errors**
> **R² = explained variation**
> Lower MAE/RMSE = better. Higher R² = better.

---

# 14. Q7 MODEL DECISION

Don't say:

> "Ridge is always best."

Wrong.

You choose based on:

> predictive performance + complexity + stability + interpretability.

If Ridge has:

> MAE = 4.1

and Lasso:

> MAE = 4.2

but Lasso uses far fewer predictors, you might prefer Lasso **if the small performance difference is acceptable and simpler deployment is valuable**.

If Ridge has materially better performance and stable coefficients, Ridge may be preferable.

The exam wants you to understand:

> **Don't choose a model from one metric alone.**

---

# 15. OVERFITTING

Overfitting means:

> Model learns the training data too specifically, including noise, and performs poorly on unseen data.

Ways to reduce it:

### 1. Cross-validation

Train/evaluate across multiple folds.

### 2. Regularization

Ridge/Lasso penalize complexity.

### 3. Proper train/test separation

Don't let test information influence training.

### 4. Feature selection

Remove unnecessary variables.

### 5. More data

If possible.

Memorize:

> **Overfitting = great on training data, poor on unseen data.**

---

# 16. QUESTION 8 — EDA

EDA = **Exploratory Data Analysis**.

Don't make it complicated.

EDA asks:

> What is in the data?

> Is it clean?

> What does the distribution look like?

> Which variables are related?

> Are there different customer groups?

Your process:

**Inspect → clean → summarize → visualize → investigate relationships → identify patterns → identify subgroups → generate insights**

---

# 17. WHICH GRAPH DO I USE?

This is worth memorizing.

### One numerical variable

**Histogram**

> age distribution

### Numerical vs numerical

**Scatter plot**

> price vs willingness to purchase

### Categorical vs numerical

**Box plot**

> satisfaction by occupation

### Categorical vs categorical

**Bar chart / stacked bar chart**

> willingness to purchase by occupation

### Time

**Line chart**

> satisfaction over time

---

# 18. Q8 INSIGHTS

Don't just describe:

> "60% said yes."

That's a summary.

They want a **decision-relevant insight**:

> "Customers with previous experience using similar products show higher willingness to purchase, suggesting prior familiarity may be associated with stronger purchase intent."

Then business implication:

> "The company could investigate this segment further during product validation."

Be careful:

> association ≠ causation.

---

# 19. THE BIGGEST CHEAT SHEET

If your brain retains nothing else, memorize this:

```text
SUPERVISED
= target exists

CLASSIFICATION
= predicts category
= yes/no, risk/no risk, default/no default
= Logistic Regression

REGRESSION
= predicts number
= GPA, ETA, price
= Linear Regression

UNSUPERVISED
= no target
= discover patterns/groups
= clustering
```

```text
LINEAR
→ predicts number

LOGISTIC
→ predicts class/probability

RIDGE
→ regression + penalty
→ handles correlated predictors
→ shrinks coefficients

LASSO
→ regression + penalty
→ can make coefficients ZERO
→ feature selection
```

```text
MISSING ≠ ZERO

categorical without order
→ ONE-HOT

categorical with real order
→ ORDINAL

numerical
→ SCALING when appropriate
```

```text
TRAIN
→ learn

VALIDATION
→ tune/choose

TEST
→ final evaluation
```

```text
DATA LEAKAGE
→ future/test information enters training
→ artificially good results
```

```text
OVERFITTING
→ learns training data too well
→ poor unseen performance
```

```text
MULTICOLLINEARITY
→ predictors highly correlated
→ unstable regression coefficients
→ remove/combine variables or use Ridge
```

```text
MAE
→ average error

RMSE
→ punishes large errors

R²
→ explained variation
```

```text
CORRELATION
→ association
≠ causation
```

```text
EDA

Histogram → distribution
Scatter → numerical vs numerical
Boxplot → categorical vs numerical
Bar chart → categorical comparisons
```

---

# FINAL 2-MINUTE MEMORY MAP

Look at each question and immediately translate it:

**Q1:**
University risk → **classification**, possibly regression.

**Q2:**
Messy recommendation data → **preprocessing + feature engineering + leakage**.

**Q3:**
Missing financial data → **missing ≠ zero + imputation + consequences**.

**Q4:**
Mixed data types → **encoding + scaling + consistent pipeline**.

**Q5:**
Healthcare system → **end-to-end ML + train/validation/test + fairness/metrics**.

**Q6:**
Crime dataset → **EDA + correlation + feature selection + multicollinearity + association ≠ causation**.

**Q7:**
ETA → **regression + Linear vs Ridge vs Lasso + MAE/RMSE/R² + overfitting**.

**Q8:**
Survey → **EDA + choose appropriate charts + insights + limitations**.

And one universal answer structure will save you:

> **Identify the problem → identify the data/target → choose the method → explain why → explain how the result is used → mention limitation/risk.**

Don't try to memorize paragraphs. Memorize the **bold concepts and arrows** above, then explain them in your own words. That is enough to construct answers to these questions under exam conditions.
