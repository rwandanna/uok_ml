# MIT91207 – Machine Learning: Assignment 1

---

## Question 1 — Early-Warning System for At-Risk Students

### (a) Problem formulation

The advisory team ultimately needs to decide **who to intervene with**, so the choice between classification and regression should follow the target variable, not the subject area.

- If the target is defined as a **category** — e.g. *at-risk / not-at-risk*, or a risk tier *low/medium/high* — this is a **classification** problem.
- If the target is a **continuous quantity** — e.g. predicted final GPA or next-assessment score — this is a **regression** problem, and "risk" is inferred afterwards by comparing the prediction to a threshold.

**Recommended formulation: classification (with a probability output), not a hard label.** A model that outputs a *risk probability* P(at risk | X) — e.g. logistic regression or a probability-producing classifier — is preferable to a model that outputs only a binary label, because:

1. Advisory capacity is limited, so students must be **ranked**, not just flagged — a probability score supports prioritisation the way a regression output would, while still collapsing to a binary decision (intervene / do not intervene) once management sets a threshold.
2. The organisational decision itself is binary (offer intervention or not), so classification maps most directly onto the decision that has to be made.

Both formulations are technically valid; the justification rests on **what decision the prediction must support**, consistent with the module's framework that a formulation should be chosen by asking what is predicted, what decision it supports, and what the consequence of an error is.

### (b) Two modelling approaches

| Approach | Algorithm | How the prediction supports the decision |
|---|---|---|
| **Classification-based** | Logistic Regression | Outputs a probability of being at risk from attendance, prior grades, assessment scores, extracurricular activity and LMS engagement. Management sets a decision threshold (e.g. 0.6); students above it are routed to advisors. Coefficients are directly interpretable, so an advisor can be told *why* a student was flagged. |
| **Regression-based** | Ridge Regression (or gradient boosting) | Predicts a continuous outcome, e.g. expected end-of-trimester GPA or next assessment score. Risk is then operationalised as a predicted value falling below a pass/attention threshold, or as a large negative deviation from the student's own recent trend. This can flag a *decline* before a hard failure threshold is crossed, enabling earlier intervention than a classification trigger alone. |

Neither approach is intrinsically superior; the choice depends on whether the university wants a direct probability of an already-defined bad outcome (classification) or an early-warning signal based on trajectory (regression).

### (c) Two factors beyond predictive accuracy

1. **Fairness across student subgroups.** A model trained on historical outcomes can encode existing inequities (e.g. unequal access to devices, part-time employment, prior schooling) so that some subgroups are flagged disproportionately even while overall accuracy looks good. This can produce a self-fulfilling stigma if flagged students are treated differently regardless of their true trajectory. This mirrors a well-documented real-world failure in which a widely used healthcare risk algorithm was highly accurate overall but systematically under-flagged Black patients because it relied on a biased proxy label (Obermeyer et al., 2019) — the same risk applies to any proxy label used for "risk" in this system.
2. **Interpretability and actionability.** Advisors need to understand *why* a student was flagged in order to design an appropriate intervention (tutoring, financial aid, counselling). A highly accurate but opaque model reduces advisor trust, complicates accountability when a student is misclassified, and gives no guidance on which lever to pull. A model should therefore also be judged on whether its output can be explained to a non-technical advisor.

---

## Question 2 — Course-Recommendation Preprocessing

### (a) Preprocessing pipeline design

| Problem | Treatment | Effect on the downstream model |
|---|---|---|
| **Missing ratings** | Do *not* fill with the global mean. Missingness here plausibly means "viewed but not rated" rather than "neutral," so add a `rating_missing` indicator and impute conservatively (median or a "not yet rated" state), keeping engagement (views/completions) as separate features. | Prevents an artificial preference signal from being injected into a recommender that would otherwise bias it toward popularity rather than genuine preference. |
| **Duplicated records** | Identify duplicate learner–course interactions with a composite key and drop duplicates, keeping the most recent/complete row. | Prevents a single interaction from being over-weighted in models that aggregate by learner–course pairs. |
| **Inconsistent category names** | Standardise via lower-casing, whitespace trimming, and a controlled mapping (e.g. "Data-Science", "data science" → `data_science`). | Stops one true category fragmenting into several sparse pseudo-categories, which would weaken the category-level signal the recommender relies on. |
| **Different numeric scales** | Scale numerical variables (e.g. courses previously taken, time on platform) with standardisation or min-max scaling. | Needed for any distance- or gradient-based model (k-NN recommenders, matrix factorisation, neural collaborative filtering); otherwise large-range variables would dominate similarity/distance calculations regardless of true importance. |

### (b) Python/pandas implementation (≥3 operations, sample data)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "learner_id": [1, 1, 2, 3, 3],
    "course_title": ["Intro to Python", "Intro to Python", "Data Science", "Data Science", "SQL Basics"],
    "category": ["Programming", "Programming", "Data Science ", " data science", "Databases"],
    "rating": [4.5, 4.5, None, 3.0, None],
    "courses_taken": [3, 3, 12, 12, 1],
})

# 1) Remove duplicate learner-course interactions, keep the latest
df = df.drop_duplicates(subset=["learner_id", "course_title"], keep="last")

# 2) Standardise inconsistent category labels
df["category"] = df["category"].str.strip().str.lower().str.replace(" ", "_")

# 3) Preserve informative missingness in ratings, then impute conservatively
df["rating_missing"] = df["rating"].isna().astype(int)
df["rating"] = df["rating"].fillna(df["rating"].median())

# 4) Scale a numeric engagement feature
df["courses_taken_scaled"] = StandardScaler().fit_transform(df[["courses_taken"]])

print(df)
```

### (c) Two engineered features

1. **Course-completion rate** = courses completed ÷ courses started. Captures genuine follow-through rather than mere exposure, distinguishing committed learners from browsers — a materially different signal than a raw course count.
2. **Recency-weighted engagement** (e.g. days since last activity, or an exponentially decayed interaction count). Learner interest drifts over time, so recent activity is more predictive of *current* preference than lifetime totals; weighting it more heavily stops the model recommending based on interests a learner has already outgrown.

### (d) Preventing temporal (future) leakage

- Any engineered feature (e.g. "average rating for this course") must be computed using **only information available before** the prediction point for that learner–course pair, not a full-dataset average that includes future ratings.
- The train/validation/test split should be **chronological** (train on interactions up to a cut-off date; validate/test on interactions after it), not a random shuffle — a random split would let a learner's future behaviour leak into features used to predict their earlier behaviour, producing an artificially optimistic evaluation that will not hold in production.

---

## Question 3 — Missing Data in Loan-Default Prediction

### (a) Critique of "replace all missing numerical values with zero"

Zero is not a neutral placeholder — it is itself a meaningful value for most financial variables, so equating "missing" with "zero" invents information rather than filling a gap:

- **Income:** a missing value means *unknown*; an imputed zero states the applicant has no income at all, which will push their risk score toward default for the wrong reason (absence of data, not genuine hardship).
- **Repayment history / missed-payments count:** zero can already be a legitimate observed value (perfect repayment record). Imputing zero for *missing* repayment history makes it indistinguishable from a genuinely clean record, destroying a real distinction (e.g. between an established good payer and a first-time borrower with no history at all).
- **Distributional distortion:** forcing many records to exactly zero creates an artificial mass at zero, distorting the variable's distribution and any correlations computed from it.
- **Loss of informative missingness:** in financial data, missingness is frequently *not random* — informal-sector workers, the self-employed, or first-time borrowers are systematically more likely to have incomplete records. Collapsing "missing" into a fixed number erases this pattern, which a well-designed model could otherwise use.

### (b) An improved missing-data strategy

The strategy should depend on **variable type** and the **likely reason** for missingness, not one blanket rule:

- **Numerical variables plausibly Missing at Random** (e.g. income missing but correlated with employment type) → group-wise or overall **median imputation** (robust to the skew typical of financial variables), combined with a `<column>_missing` binary indicator so the model can still use the fact of missingness as a signal.
- **Variables where missingness is itself informative (Missing Not at Random)** — e.g. no credit history because the applicant is a first-time borrower → encode explicitly as a separate category (**"Unknown" / "No prior history"**) rather than imputing a numeric value.
- Removing rows should be considered only when missingness affects a small, non-systematic fraction of records; broad removal in a lending dataset risks discarding exactly the applicants (thin-file, informal-sector) the institution most needs to serve fairly.

### (c) Python function for a preprocessing pipeline

```python
import pandas as pd

def handle_missing_financial(df, numeric_cols, categorical_cols, group_col=None):
    """
    Assumptions:
    - Numerical columns are treated as plausibly Missing at Random and imputed
      with the (group-wise, if group_col given) median, which is robust to the
      skew typical of income/loan data.
    - Categorical columns may be Missing Not at Random; missingness is kept as
      an explicit 'Unknown' category rather than imputed toward the mode, so
      the model can use absence-of-information as a signal.
    - A '<col>_missing' indicator is added for every numeric column so the
      model can still learn from the missingness pattern itself.
    """
    df = df.copy()
    for col in numeric_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        if group_col:
            df[col] = df.groupby(group_col)[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    return df
```

### (d) Consequences of inappropriate treatment

- **Model level:** naive imputation introduces a systematic (not random) distortion that the algorithm can mistake for genuine signal, biasing coefficients or split decisions; because the distortion is systematic, standard cross-validation on the same corrupted data will not reveal the problem, giving false confidence in the deployed model.
- **Applicant level:** applicants whose data happens to be missing — disproportionately informal-sector or first-time borrowers — may be unfairly denied credit (if missingness is read as poor standing) or unfairly approved with mispriced risk (if missingness is read as neutral/favourable), raising both financial-stability and fair-lending concerns.

---

## Question 4 — Feature Engineering for Subscription Renewal

### (a) Feature-engineering pipeline

- **Numerical variables** (age, monthly expenditure, previous transactions, account duration): **standardise** (zero mean, unit variance) for models sensitive to feature magnitude/geometry — logistic regression, SVM, k-NN. Tree-based models do not require scaling but can still benefit from addressing skew (e.g. log-transforming monthly expenditure).
- **Categorical variables** (subscription type, region, payment method, customer status): encoding should match the variable's structure.
  - **Nominal, no natural order** (region, payment method) → **one-hot encoding**.
  - **Genuinely ordinal** (e.g. a tiered subscription basic < standard < premium) → **ordinal encoding** that preserves the tier order — more parsimonious than one-hot and correctly conveys "more/less" relationships.

### (b) Consequences of encoding choices

| Encoding | Appropriate when | Risk if misapplied |
|---|---|---|
| **One-hot encoding** | Nominal categories with no order (region, payment method) | Increases dimensionality with cardinality; for high-cardinality fields this can create a sparse, high-dimensional space that raises overfitting/multicollinearity risk (mitigated by dropping one reference category) |
| **Ordinal encoding** | A genuine order exists (basic/standard/premium) | Applied to an unordered category (e.g. region A=1, B=2, C=3 arbitrarily), it forces a linear/distance-based model to treat categories as though they sit on a meaningful numeric scale, creating spurious relationships that have no basis in the data |
| **Arbitrary numerical encoding** (unplanned integer codes for a nominal field) | Essentially never appropriate | Compounds the ordinal-encoding problem without even the discipline of a deliberately chosen order; should be avoided for nominal variables in any linear or distance-based model |

### (c) Python/pandas implementation

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
    "subscription_type": ["basic", "premium", "standard", "premium"],
    "region": ["Kigali", "Musanze", "Huye", "Kigali"],
    "payment_method": ["mobile_money", "card", "mobile_money", "card"],
})

numeric_cols = ["age", "monthly_expenditure", "previous_transactions", "account_duration_months"]
categorical_cols = ["subscription_type", "region", "payment_method"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
])

pipeline = Pipeline([("prep", preprocessor)])
X_ready = pipeline.fit_transform(df)
```

### (d) Consistency at deployment

The fitted preprocessing (scaler means/variances, encoder category mapping) must be **learned once on training data and saved as part of a single pipeline object** (e.g. `sklearn.pipeline.Pipeline` + `joblib`), then applied unchanged to every new record — never re-derived by hand. The encoder should be configured with `handle_unknown="ignore"` so an unseen category at inference time does not crash the pipeline or silently misalign columns. The pipeline should be version-controlled with the trained model so any change to the feature-engineering logic is deployed together with a corresponding retrain.

---

## Question 5 — Hospital Readmission Prediction

### (a) End-to-end workflow

1. **Data acquisition** — extract admission, diagnosis and treatment records under clinical data-governance/privacy controls (de-identification, access logging), since health data carries regulatory obligations.
2. **Cleaning & integration** — link episodes to one patient identifier, standardise diagnosis coding and length-of-stay units, resolve duplicate/conflicting records.
3. **Exploratory analysis** — readmission base rate, missingness patterns, subgroup composition.
4. **Feature engineering** — e.g. prior-admission count, comorbidity count, time since last discharge — every feature checked for availability at prediction time.
5. **Data partitioning** — train/validation/test (part b).
6. **Model training and tuning** on train/validation only.
7. **Evaluation** on the held-out test set with both overall and subgroup metrics (part c).
8. **Deployment considerations** — integration into the clinical system, a human-in-the-loop review step (prediction supports, not replaces, clinical judgement), and a monitoring/retraining schedule for drift.

Each stage is justified by the same concern: because a readmission prediction can influence real clinical decisions, the workflow must prioritise reliable, auditable, monitored outputs over a one-off accuracy figure.

### (b) Data-partitioning strategy

Partition **chronologically** (train on earlier admissions, validate/test on later ones) **and group by patient**, so every admission belonging to one patient falls entirely inside a single partition (train, validation, *or* test — never split across them).

A careless **random row-level split** would likely place two admissions from the same patient in different partitions. Because those admissions share stable patient-level characteristics (chronic conditions, baseline demographics), the model would effectively be evaluated partly on a patient it has already seen — producing an overly optimistic performance estimate that will not reproduce when the model meets genuinely new patients after deployment.

### (c) Subgroup performance disparity

High **overall** accuracy can hide poor performance for a specific patient group (e.g. by age, sex, or insurance/socioeconomic status) — the aggregate figure can mask a subgroup for which the model is miscalibrated or systematically wrong. Before deployment, evaluate:

- **Subgroup-disaggregated recall/false-negative rate** — a missed high-risk patient is typically more costly than a false alarm, so recall parity across groups matters more than accuracy parity.
- **Subgroup-specific calibration** — does a predicted 30% risk mean the same thing in every group?
- **Precision–recall or ROC-AUC by subgroup**, since accuracy alone is known to be misleading under class imbalance (readmission is typically a minority outcome).

This mirrors a documented real-world case: a widely used healthcare risk algorithm appeared highly accurate overall but systematically under-identified Black patients because it optimised a cost-based proxy label rather than health need (Obermeyer et al., 2019) — a reminder that label choice and subgroup evaluation, not just algorithm choice, determine whether an accurate-looking model is safe to deploy.

### (d) Python partitioning implementation

```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# df has columns: patient_id, admission_date, ...features..., readmitted
df = df.sort_values("admission_date")

# Chronological cut-off separating a held-out test period
cutoff = df["admission_date"].quantile(0.8)
train_val = df[df["admission_date"] <= cutoff]
test = df[df["admission_date"] > cutoff]

# Within train_val, split by patient group to build train/validation (no patient in both)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(train_val, groups=train_val["patient_id"]))
train = train_val.iloc[train_idx]
validation = train_val.iloc[val_idx]

print(train.shape, validation.shape, test.shape)
```

---

## Question 6 — Crime-Rate Prediction (US Crime Dataset, 47 states)

> **Note on the data used.** The Kaggle file itself was not included with the assignment materials. The results below were computed on the public 47-U.S.-state dataset that this Kaggle dataset republishes — the classic dataset compiled by Ehrlich (1973) and widely distributed for teaching (e.g. as `MASS::UScrime` in R), containing the same variables and the same target definition (crime rate per 100,000 population). The figures reported are genuine computed values from this dataset, not invented numbers.

### (a) Feature-selection strategy

With only 47 observations and roughly 15 candidate predictors, the sample is small relative to the number of features, so feature selection must guard against overfitting as much as against irrelevance:

1. Remove identifiers/near-constant variables that carry no predictive information.
2. Examine each predictor's **bivariate correlation with Crime** (part b) to shortlist variables with a non-trivial linear association.
3. Examine **correlations among the candidate predictors themselves** to detect multicollinearity (part c) — two highly correlated predictors add largely redundant information.
4. Decide whether a highly correlated pair should be **combined** (e.g. a composite index) or whether one variable should be **retained in preference to the other**, based on measurement reliability and policy relevance.
5. Treat weakly correlated variables as removal candidates, but confirm with a multivariate method (e.g. regularised regression) rather than the correlation matrix alone, since a variable can still be useful in combination even with a weak marginal correlation.

### (b) Exploratory analysis and correlation matrix

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("us_crime_dataset.csv")
print(df.shape); print(df.isnull().sum().sum()); print(df.duplicated().sum())

corr_matrix = df.corr(numeric_only=True)
print(corr_matrix["Crime"].sort_values(ascending=False))

plt.figure(figsize=(9, 7))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout(); plt.show()
```

**Results (47 states, no missing values, no duplicates).** Correlation of each variable with `Crime`, strongest first:

| Variable | r with Crime | Variable | r with Crime |
|---|---|---|---|
| Po1 (police expenditure, 1960) | **+0.69** | LF (labour-force participation) | +0.19 |
| Po2 (police expenditure, 1959) | +0.67 | Ineq (income inequality) | −0.18 |
| Wealth | +0.44 | U2 (unemployment, older males) | +0.18 |
| Prob (probability of imprisonment) | −0.43 | Time (avg. time served) | +0.15 |
| Pop (population) | +0.34 | So (Southern state) | −0.09 |
| Ed (education level) | +0.32 | M (young-male proportion) | −0.09 |
| M.F (males per 1000 females) | +0.21 | U1, NW | ≈ 0.0–0.05 |

Two clear multicollinear pairs emerge: **Po1 and Po2 correlate at r = 0.99**, and **Wealth and Ineq correlate at r = −0.88**.

**Why correlation alone is not sufficient for feature selection:**

1. It captures only a **linear, bivariate** relationship — a variable with weak correlation could still matter in combination with others (interaction effects), and a variable with strong correlation may be redundant if another predictor already carries the same information (as with Po1/Po2 above).
2. A correlation, however strong, is an **association**, not evidence of causal effect — police expenditure being positively correlated with crime almost certainly reflects that high-crime states allocate more to police, not that police spending causes crime (reverse causality/policy feedback), or that both are driven by an unobserved factor such as urbanisation.

### (c) Multicollinearity: problem and remedy

**Problem:** Po1 and Po2 (r = 0.99) contain almost identical information. In an OLS regression this inflates the variance of the estimated coefficients, makes individual coefficients unstable (small changes in the sample can flip a sign), and can produce a model that predicts well overall while giving misleading guidance about which specific variable drives the outcome — a serious issue for a government audience trying to attribute effects to a specific policy lever.

**Strategy:** retain one variable from the correlated pair (or combine them into a composite, e.g. an average police-expenditure index) based on which is more reliably measured or policy-actionable; alternatively, use a **regularised regression** (Ridge) which shrinks correlated coefficients toward each other and stabilises the estimates without requiring an a-priori choice of which variable to discard.

### (d) Recommendations to government: association vs causation

- Variables strongly and consistently associated with Crime (police expenditure, wealth, imprisonment probability) should be reported as **statistically useful for forecasting and resource-planning purposes**, not as demonstrated causes of crime.
- The positive association between police expenditure and crime should **not** be read as "more police spending causes more crime" — it is far more consistent with reverse causality (high-crime states spend more on policing).
- Purely demographic variables (e.g. the proportion of young males) should be used only for context and targeting, never framed as demographic characteristics "causing" crime — both because cross-sectional correlational data cannot support that claim and because such framing risks stigmatising communities rather than informing constructive policy.
- Any variable proposed as a basis for a major resource-reallocation decision should first be examined with a stronger design (e.g. a piloted intervention with pre/post or control-group comparison) before being treated as a causal lever.

---

## Question 7 — ETA Prediction for YEGO (OLS vs Ridge vs Lasso)

### (a) Experimental procedure

1. **Inspect the data:** `synthetic_ETA.csv` has 1,500 rows, 15 columns, no missing values, no duplicates; one categorical predictor (`road_type`: local/arterial/express/highway) and eleven numerical predictors; `observation_id` is dropped (an identifier, not a predictor).
2. **Split:** 80% train (1,200 rows) / 20% test (300 rows), fixed random seed, test set held out from every model-selection decision.
3. **Preprocess inside a single pipeline:** `StandardScaler` on numerical predictors, `OneHotEncoder(drop="first")` on `road_type` — identical preprocessing for all three models, fitted only on the training fold to avoid leakage.
4. **Train** OLS directly; tune Ridge's and Lasso's regularisation strength (α) using 5-fold cross-validation on the training data only.
5. **Evaluate** all three final models once on the untouched test set using MAE, RMSE and R².
6. **Compare fitted coefficients** across the three models to see how each handles the dataset's built-in multicollinearity (`journey_distance_km`/`distance_proxy`, `traffic_index`/`traffic_proxy`) and the deliberately uninformative `random_noise_feature`.

### (b) Implementation and results

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("synthetic_ETA.csv").drop(columns=["observation_id"])
X = df.drop(columns=["eta_minutes"]); y = df["eta_minutes"]
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([("num", StandardScaler(), num_cols),
                           ("cat", OneHotEncoder(drop="first"), cat_cols)])
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(pipe, name):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {"model": name, "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)}

ols = Pipeline([("prep", prep), ("model", LinearRegression())])
ols_result = evaluate(ols, "OLS")

ridge_grid = GridSearchCV(Pipeline([("prep", prep), ("model", Ridge())]),
                           {"model__alpha": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]},
                           cv=cv, scoring="r2").fit(X_train, y_train)
ridge_result = evaluate(ridge_grid.best_estimator_, "Ridge")

lasso_grid = GridSearchCV(Pipeline([("prep", prep), ("model", Lasso(max_iter=20000))]),
                           {"model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]},
                           cv=cv, scoring="r2").fit(X_train, y_train)
lasso_result = evaluate(lasso_grid.best_estimator_, "Lasso")
```

**Test-set results (verified by direct computation on the supplied dataset):**

| Model | MAE (min) | RMSE (min) | R² | Best α |
|---|---|---|---|---|
| OLS Linear Regression | 4.38 | 6.24 | 0.800 | — |
| Ridge Regression | 4.38 | 6.24 | 0.800 | 1.0 |
| Lasso Regression | **4.36** | **6.23** | **0.801** | 0.05 |

The three models explain roughly 80% of ETA variance and achieve near-identical raw accuracy (differences well under half a minute of MAE). The interesting difference is in the **coefficients**: `journey_distance_km` and `distance_proxy` are correlated at r ≈ 0.93, and `traffic_index`/`traffic_proxy` at r ≈ 0.90. Under OLS, `distance_proxy` receives a counter-intuitive **negative** coefficient (a classic multicollinearity artefact — OLS cannot uniquely apportion shared explanatory power between two near-duplicate variables). Lasso shrinks `traffic_proxy`'s coefficient to (near) zero and substantially reduces `distance_proxy`'s coefficient, effectively resolving the redundancy by leaning on the more directly measured variable; it also shrinks `random_noise_feature` and `time_of_day_hours` toward zero, correctly identifying them as weak predictors.

### (c) Model recommendation for YEGO

Predictive accuracy alone does not separate the three models. **Lasso is the recommended choice**, on grounds of stability and deployment simplicity rather than a marginal accuracy gain:

- It resolves the multicollinearity between the distance/traffic variables and their proxies by shrinking the redundant ones, producing coefficients that are safe to interpret operationally (unlike OLS's sign-inconsistent coefficient on `distance_proxy`).
- It yields a simpler deployed model — a data feed that stops populating `traffic_proxy`, for example, would not degrade a Lasso model that has already learned to disregard it.
- It automatically down-weights the uninformative noise feature, a useful robustness property if new, unvetted signals are added to the feature pipeline later.

Ridge remains a reasonable fallback if YEGO wants to retain all proxy signals as redundancy (e.g. a hedge against one sensor failing) rather than removing them; but for a deployable, interpretable ETA model, **Lasso** is preferred. The choice reflects the module's principle that model comparison should go beyond a single metric, considering stability, complexity and interpretability alongside raw error.

### (d) Reducing the risk of overfitting

1. **Cross-validated hyperparameter tuning with a strictly held-out test set.** The regularisation strength (α) for Ridge and Lasso was tuned only via cross-validation on the training data; the test set was used exactly once, for final reporting. This should be maintained in production, and extended to a **rolling/time-based validation split**, since traffic and journey data arrive sequentially and traffic patterns evolve — a random split can mix future information into training.
2. **Regularisation itself**, which directly constrains model complexity and limits the model's ability to fit noise specific to the training sample (including the noise feature and redundant proxies). In production this should be paired with **periodic re-validation on fresh, held-out journeys** and monitoring of live prediction error against the reported benchmark (MAE ≈ 4.4 min), treating a material rise in live error as a trigger for retraining.

---

## Question 8 — Customer Survey Analysis for a Product Launch

> **Note on the data used.** No raw survey file was supplied with the assignment materials. The strategy and code below are complete and directly executable once the real survey data is attached; where illustrative output is shown, it uses a small representative sample built from the variables named in the question (age group, occupation, previous use, preferred features, price range, satisfaction, willingness to purchase) and is labelled as such — no invented percentages or findings are reported as if they were real results.

### (a) Exploratory data-analysis strategy

1. **Data-quality pass:** number of respondents/variables, data types, missing responses per question (non-response can itself be informative — e.g. skipped price questions may indicate price sensitivity), duplicate submissions, consistency of categorical labels.
2. **Univariate distributions:** chart type matched to variable type — bar/count plots for nominal variables (occupation, preferred feature), ordered bar charts for ordinal variables (satisfaction, willingness to purchase, price range if Likert-scaled), histograms for genuinely continuous variables (e.g. numeric age).
3. **Bivariate/subgroup relationships:** cross-tabulate willingness to purchase against age group, previous use, satisfaction and expected price, using grouped bar charts or boxplots.
4. **Segment-level synthesis:** combine two or more variables (e.g. occupation × previous use) to identify customer segments with distinctly different purchase intent — typically more actionable for a launch decision than any single bivariate relationship.

### (b) Python visualisations (≥3 relationships)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("customer_survey.csv")
print(df.shape); print(df.isnull().sum()); print(df.duplicated().sum())

# 1) Willingness to purchase by age group — two categorical variables → grouped count plot
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="age_group", hue="willingness_to_purchase")
plt.title("Willingness to Purchase by Age Group")
plt.tight_layout(); plt.show()

# 2) Willingness to purchase by previous use — two categorical variables → grouped count plot
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="previous_use", hue="willingness_to_purchase")
plt.title("Willingness to Purchase by Previous Product Use")
plt.tight_layout(); plt.show()

# 3) Satisfaction (numeric/ordinal) across willingness groups → boxplot, shows full distribution
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="willingness_to_purchase", y="satisfaction_score")
plt.title("Satisfaction Score by Willingness to Purchase")
plt.tight_layout(); plt.show()
```

Chart choice follows variable type: two categorical variables are compared with grouped count plots rather than a scatterplot (neither axis is continuous); willingness against a numeric satisfaction score uses a boxplot rather than a bar of means, because it reveals the full within-group distribution — purchase intent may be driven by a distinct sub-population rather than a uniform shift.

### (c) Interpreting the findings

Once run on the real data, this analysis is designed to surface two categories of insight for the launch decision:

1. **Which segments show materially higher purchase intent than the overall average** (e.g. a specific age group or occupation) — this should guide initial target-market selection and marketing spend.
2. **Which factors are most closely associated with willingness to purchase** among the variables collected (e.g. a clear separation in the satisfaction-score boxplot between "willing" and "unwilling" respondents, or a drop in intent at higher price points) — this should guide which product attributes or price tier to emphasise at launch.

Because no genuine survey data was available for this submission, no specific percentage or segment is reported here as an actual finding; the code above will produce them once the dataset is attached.

### (d) Limitation of survey-based conclusions and additional evidence needed

**Limitation:** stated purchase intent is a weak proxy for actual purchasing behaviour. Respondents can over-state willingness to purchase due to social-desirability bias or hypothetical-choice bias (no real financial commitment is attached to a survey answer), and the described product concept typically omits real frictions (competing options, an actual payment step, switching costs) that affect a genuine purchase decision.

**Additional evidence the company should gather before a launch decision:**
- A small-scale pilot or limited release with real transactions (revealed rather than stated preference).
- A/B-tested pricing or landing-page conversion data.
- Competitor and market-sizing analysis.

None of these can be substituted for by a survey alone, however well it is analysed.

---

## References

- Altman, N., & Krzywinski, M. (2015). Association, correlation and causation. *Nature Methods*, 12(10), 899–900.
- Ehrlich, I. (1973). Participation in illegitimate activities: A theoretical and empirical investigation. *Journal of Political Economy*, 81(3), 521–565. (Source of the 47-state crime dataset used in Question 6.)
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
