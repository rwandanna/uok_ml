# MIT91207 — Machine Learning
## Assignment 1 — Final Submission Answer Sheet

> **Student:** Victor Isingizwe Munezero  
> **Course:** MIT91207 — Machine Learning  
> **Assignment:** Assignment 1  
>
> **Note:** The answers below are written for direct submission. Code is included where requested. Numerical results are reported only where they can be supported by the supplied data.

---

# Question 1 — Early-Warning System for At-Risk Students
**[12 Marks]**

## (a) Problem formulation — [5 Marks]

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

## (b) Two modelling approaches — [5 Marks]

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

## (c) Two factors beyond predictive accuracy — [2 Marks]

### 1. Fairness and bias

The model should be evaluated for systematic differences in performance across relevant student groups. Historical data may contain existing inequalities, which can cause the model to reproduce or amplify them.

### 2. Interpretability and actionability

Academic advisors should be able to understand the factors contributing to a student's risk prediction. This helps them decide what intervention is appropriate and supports responsible use of the model.

---

# Question 2 — Course-Recommendation Preprocessing
**[13 Marks]**

## (a) Preprocessing pipeline — [5 Marks]

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

## (b) Python/pandas implementation — [5 Marks]

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

## (c) Two engineered features — [2 Marks]

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
**[13 Marks]**

## (a) Critique of zero-imputation — [5 Marks]

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

## (b) Improved missing-data strategy — [3 Marks]

The strategy should depend on both the variable type and the reason for missingness.

For numerical variables such as income, **median imputation** can be used where appropriate, preferably with the statistic calculated from the training data. A missingness indicator can also be added when the fact that a value is missing may itself contain predictive information.

For categorical variables such as employment information, missing values can be represented using a separate category such as **`Unknown`**.

If the missingness mechanism is systematic, simple imputation should be treated cautiously because the fact that information is missing may be related to the unobserved value.

---

## (c) Python function — [3 Marks]

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

## (d) Consequences for model performance and fairness — [2 Marks]

Inappropriate missing-data treatment can distort the relationships learned by the model and reduce predictive performance, including discrimination and calibration.

It can also affect applicants unfairly. For example, treating missing income as zero may cause an applicant to receive an unnecessarily high estimated default risk.

Therefore, poor missing-data treatment can create both **statistical errors and unfair lending decisions**.

---

# Question 4 — Feature Engineering for Subscription Renewal
**[12 Marks]**

## (a) Feature-engineering pipeline — [3 Marks]

For numerical variables such as age, monthly expenditure, previous transactions and account duration, I would standardise the variables when using scale-sensitive algorithms such as logistic regression, SVM or k-nearest neighbours.

For categorical variables, the encoding method should depend on whether the variable is nominal or genuinely ordinal:

- **Nominal variables**, such as geographical region and payment method, should normally use one-hot encoding.
- A variable such as subscription tier should use ordinal encoding only if its categories have a genuine order, such as Basic < Standard < Premium.

The preprocessing should be fitted on the training data and incorporated into a reproducible pipeline.

---

## (b) Encoding consequences — [4 Marks]

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

## (c) Python implementation — [3 Marks]

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

## (d) Ensuring deployment consistency — [2 Marks]

The preprocessing transformations should be fitted on the training data and saved as part of the same pipeline as the model.

The same fitted scaler and encoder must then be used for validation, testing and newly arriving customer records. They should not be independently refitted on new data.

For categorical variables, `handle_unknown="ignore"` can prevent the pipeline from failing when a previously unseen category appears.

This ensures that the model receives the same feature representation during training and deployment.

---

# Question 5 — Hospital Readmission Prediction
**[13 Marks]**

## (a) End-to-end ML workflow — [4 Marks]

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

## (b) Data partitioning — [3 Marks]

The data should preferably be split **chronologically**, with earlier admissions used for training and later admissions used for validation and testing. This better represents the real deployment situation in which the model predicts future admissions.

Where patients have multiple admissions, records should also be **grouped by patient** so that the same patient does not appear in both training and evaluation sets when evaluating generalisation to unseen patients.

A careless random row-level split can allow information from the same patient to appear in both training and test data, producing overly optimistic performance estimates.

---

## (c) High overall accuracy but subgroup differences — [3 Marks]

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

## (d) Python partitioning — [3 Marks]

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
**[13 Marks]**

## (a) Feature-selection strategy — [3 Marks]

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

## (b) Exploratory analysis and correlation matrix — [6 Marks]

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

## (c) Multicollinearity — [2 Marks]

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

## (d) Government recommendations — [2 Marks]

Variables that are statistically associated with crime can be used as **predictive indicators** for identifying areas that may require further analysis or resource planning.

However, these associations should not automatically be interpreted as causal relationships.

For example, a positive association between police expenditure and crime does not prove that increased police expenditure causes higher crime. Areas experiencing more crime may increase police spending, creating reverse causality. Other factors may also affect both variables.

Therefore, predictive associations can support planning and prioritisation, but causal policy conclusions require stronger evidence such as longitudinal studies, controlled comparisons or other causal research designs.

---

# Question 7 — ETA Prediction for YEGO
**[14 Marks]**

## (a) Experimental procedure — [3 Marks]

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

## (b) Python implementation and results — [6 Marks]

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

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| OLS | 4.380 | 6.237 | 0.800 |
| Ridge | 4.378 | 6.235 | 0.800 |
| Lasso | 4.358 | 6.226 | 0.801 |

The selected regularisation parameters are:

- Ridge: \(\alpha = 1.0\)
- Lasso: \(\alpha = 0.05\)

The three models therefore have very similar predictive performance.

---

## (c) Model choice — [2 Marks]

Lasso is a reasonable model choice because it provides predictive performance comparable to OLS and Ridge while also shrinking coefficients and potentially setting some coefficients exactly to zero.

This can produce a simpler model and provide a form of feature selection, which is useful when some predictors are weak or redundant.

However, the numerical performance difference is very small. Therefore, it would be incorrect to claim that Lasso is dramatically more accurate.

The appropriate conclusion is:

> **OLS, Ridge and Lasso perform almost equally, while Lasso provides an additional simplicity and feature-selection advantage.**

---

## (d) Reducing overfitting — [3 Marks]

### 1. Cross-validation and a strictly held-out test set

Cross-validation should be used on the training data to tune hyperparameters such as \(\alpha\), while the final test set remains untouched until final evaluation.

### 2. Regularisation and evaluation on unseen data

Ridge and Lasso introduce regularisation that constrains model complexity and reduces sensitivity to noise.

The final model should also be evaluated on genuinely new journeys. For time-dependent ETA prediction, chronological or rolling validation may be more realistic than a purely random split because future traffic patterns should not influence evaluation of earlier predictions.

---

# Question 8 — Customer Survey Analysis
**[12 Marks]**

## (a) EDA strategy — [3 Marks]

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

## (b) Python visualisations — [4 Marks]

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

## (c) Two insights for management — [3 Marks]

The analysis should identify at least two decision-relevant insights.

### 1. High-intent customer segments

Identify age, occupation, previous-use or other customer groups with substantially higher willingness to purchase. These segments can inform the initial target market.

### 2. Factors associated with purchase intention

Determine whether satisfaction, expected price range, previous use or another variable is strongly associated with willingness to purchase. This can inform product positioning, pricing and marketing decisions.

The actual numerical segments and values should only be reported after analysing the real survey dataset.

---

## (d) One limitation — [2 Marks]

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

