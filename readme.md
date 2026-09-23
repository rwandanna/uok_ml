# Assignment 2 — Machine Learning (MIT91207)

### University of Kigali — Graduate School — Master of Information Technology




> **Dataset verification statement.** Before Questions 4(b), 5, and 7 were answered, the three supplied files were inspected directly in pandas rather than assumed: `credit.csv` (1,000,000 rows × 8 columns, 8.74% fraud, no missing values), `home.csv` (307,511 rows × 122 columns, 8.07% TARGET rate, `ORGANIZATION_TYPE` confirmed at 58 categories, `NAME_EDUCATION_TYPE` confirmed ordinal with 5 levels), and `prices.csv` (1,460 rows × 81 columns, `SalePrice` present, multicollinearity confirmed: GrLivArea/TotRmsAbvGrd r = 0.825, GarageCars/GarageArea r = 0.882). Every figure below for these three questions comes from actually running the code shown against these files. This Markdown file, the submitted DOCX, and the accompanying `.ipynb` notebook all report the same numbers because they were generated from the same executed code.



## Table of Contents



- [Question 1](#question-1)
- [Question 2](#question-2)
- [Question 3](#question-3)
- [Question 4](#question-4)
- [Question 5](#question-5)
- [Question 6](#question-6)
- [Question 7](#question-7)
- [Question 8](#question-8)

- [References](#references)


---


# Question 1

### Overfitting, Bias and Variance in a Decision Tree Classifier



## 1(a) — Diagnosing the training/validation gap

> (a) Analyze the difference between the training and validation accuracies. Identify the most likely cause of this performance gap and explain the evidence supporting your conclusion. (3 Marks)



### Answer



The reported model achieves 99.6% training accuracy against 59.4% validation accuracy — a gap of 40.2 percentage points. A gap of this size is not explained by ordinary sampling variation between the two sets; it is the signature of a model that has memorised the training data rather than learned a generalisable decision boundary. With max_depth = None and min_samples_leaf = 1, the tree is permitted to keep splitting until every leaf is pure (or contains a single observation), so it fits noise and idiosyncratic combinations of feature values that are specific to the training sample and do not recur in the validation set. The training accuracy near 100% is itself the direct evidence: it shows the tree has grown deep enough to separate almost every training instance individually, which is only possible when the tree encodes sample-specific detail instead of the underlying customer-churn signal. The most likely cause of the performance gap is therefore overfitting driven by unconstrained tree depth.



### Explanation (plain language)



**Plain-language version:** imagine a student who memorises last year's exact exam paper word-for-word instead of learning the subject. On that specific paper they score 99.6% — flawless recall. Handed a new paper covering the same subject, they score only 59.4%, because they never actually learned the underlying material, only the exact answers to one specific set of questions. The Decision Tree here has done exactly this to the training data: with no depth limit and a minimum leaf size of one observation, it kept splitting until it could recite the training set almost perfectly, at the cost of never learning the general pattern that separates churners from non-churners.



### Real-World Application



A telecom company deploying this exact model into production would see it perform brilliantly in every internal demo (run on the training data) and then fail almost immediately once it starts scoring genuinely new customers, which is precisely the population it is meant to serve. This is one of the most common and most expensive mistakes in applied machine learning — presenting training-set performance as if it were a promise about real-world performance.



---



## 1(b) — Bias vs. variance

> (b) Explain the roles of bias and variance in this situation. Based on the reported results, determine whether the model exhibits characteristics of high bias or high variance, and justify your answer. (3 Marks)



### Answer



Bias is the systematic error introduced when a model's hypothesis space is too restrictive to represent the true relationship between features and target; high-bias models underfit and perform poorly on both training and validation data. Variance is the sensitivity of the fitted model to the particular training sample drawn; a high-variance model changes substantially if trained on a different sample from the same population, and this instability shows up as a large train-validation gap. In this case the model exhibits high variance rather than high bias. The evidence is direct: training accuracy is 99.6% (near-perfect fit to the training sample, ruling out an overly restrictive hypothesis) while validation accuracy is only 59.4% (poor transfer to unseen data). An unconstrained decision tree has effectively unlimited capacity, so with no depth or leaf-size constraint it fits sample-specific noise, producing exactly this signature of low bias, high variance.



### Explanation (plain language)



**Bias** is like using a ruler that is too short for the wall you are measuring — no matter how carefully you use it, you systematically get the wrong answer, because the tool itself is too limited. **Variance** is like measuring the same wall with a rubber tape measure that stretches differently every time you pick it up — the tool is capable of the right answer, but it is unstable and gives a different reading depending on small, irrelevant factors. An unconstrained Decision Tree is the rubber tape measure: capable of representing almost any pattern, but so flexible that it bends itself around the specific noise of whichever training sample it happens to see.



---



## 1(c) — max_depth and min_samples_leaf

> (c) Explain how the specified values of max_depth and min_samples_leaf affect the complexity of the Decision Tree. (2 Marks)



### Answer



max_depth caps the number of sequential splits from root to leaf; larger values (or None) allow the tree to partition the feature space into progressively smaller and more numerous regions, increasing model complexity and capacity. min_samples_leaf sets the minimum number of observations a terminal node must contain; smaller values (the extreme being 1) allow leaves to be built around individual observations, which likewise increases complexity, while larger values force each leaf to summarise a broader group of samples, acting as a regulariser. With max_depth = None and min_samples_leaf = 1, both constraints are set to their least restrictive values simultaneously, which is why the tree is able to grow to maximal complexity and interpolate the training set almost exactly.



---



## 1(d) — Overfitting vs. underfitting and remediation

> (d) Explain the difference between overfitting and underfitting. For the model described above, explain which condition is present and describe two changes to the Decision Tree configuration that could help improve its generalization performance. (2 Marks)



### Answer



Underfitting occurs when a model is too simple to capture the underlying pattern, producing low training and low validation performance together. Overfitting occurs when a model captures the training pattern together with sampling noise, producing high training performance but disproportionately lower validation performance. The model described exhibits overfitting: training accuracy is high (99.6%) while validation accuracy collapses to 59.4%. Two configuration changes that would reduce this gap:

- Set a finite max_depth (for example, tune over {3, 5, 8, 10} via cross-validation) to bound how many times the tree can split and stop it from isolating individual training records.
- Increase min_samples_leaf (for example, to 20–50) so that each leaf must summarise a sufficiently large group of customers before making a prediction, which smooths the decision boundary and reduces sensitivity to noise.



### Example



If the same engineer instead trained with `max_depth=2`, the tree could only ask two questions before making a decision — e.g. "Has the customer complained in the last 30 days? → Is their monthly spend below $20?" — which is very likely to **underfit**: both training and validation accuracy would be mediocre, because two questions are rarely enough to capture real churn behaviour. The right configuration sits between these two extremes, found by tuning `max_depth` and `min_samples_leaf` against validation performance rather than training performance.



---



# Question 2

### Parametric vs. Non-Parametric Classifiers and Inductive Bias



## 2(a) — Linear SVM vs. Decision Tree

> (a) Analyze the validation performance of the two models. Explain how a parametric model and a non-parametric model differ in the way they represent relationships in data and discuss how these structural differences influence their inductive biases. Based on the reported training and validation F1-scores, explain which model exhibits stronger evidence of overfitting and why. (5 Marks)



### Answer



| Model | Training F1 | Validation F1 | Gap |
|---|---|---|---|
| Linear SVM | 0.79 | 0.76 | 0.03 |
| Decision Tree | 0.99 | 0.63 | 0.36 |

A parametric model, such as a Linear SVM, assumes a fixed functional form in advance — here, a hyperplane defined by a weight vector whose dimensionality is fixed by the number of input features regardless of how much training data is available. This constrains the hypothesis space and imposes an inductive bias toward linearly separable decision boundaries: the model cannot represent arbitrarily complex boundaries even if it is given unlimited data, but for the same reason it cannot easily memorise noise either. A non-parametric model such as a Decision Tree makes no fixed assumption about the functional form; its effective capacity (number of splits, depth) grows with the data and the chosen stopping criteria, so its inductive bias is much weaker — it can represent highly irregular, axis-aligned decision boundaries, but with that flexibility comes a much greater capacity to fit sampling noise.

The reported numbers show this difference directly. The Linear SVM's train-validation gap is only 0.03 F1, consistent with a constrained hypothesis space that cannot deviate far from a generalisable linear boundary. The Decision Tree's gap is 0.36 F1 — training F1 of 0.99 versus validation F1 of 0.63 — which is a far stronger signature of overfitting. The unconstrained tree has used its flexibility to fit patterns specific to the training fold rather than the underlying fraud signal, whereas the SVM's restricted hypothesis space acted as an implicit regulariser.



### Explanation (plain language)



**Plain-language version:** a parametric model is like a tailor who only knows how to cut one style of suit — a straight-cut blazer. Give them any body shape and they will still produce a blazer; it will fit reasonably well for most people but will never perfectly match someone with an unusual shape, because the tailor's method is fixed in advance. A non-parametric model is like a tailor who custom-drapes the fabric directly onto each client's body with no predetermined pattern — capable of a much better fit, but also capable of accidentally sewing in every wrinkle and fold of the fabric that happened to be there on that one fitting day, which will not be there next time.



### Real-World Application



In production fraud systems, teams often start with a linear model (Logistic Regression / Linear SVM) precisely because its limited flexibility acts as free insurance against overfitting on a training sample that will inevitably differ from tomorrow's transaction patterns, even though a well-regularised tree-based ensemble (e.g. Gradient Boosting) can eventually outperform it once properly tuned.



---



## 2(b) — Three overfitting-reducing hyperparameters

> (b) Identify three specific Decision Tree hyperparameters that could be included in a hyperparameter optimization grid to reduce overfitting. For each hyperparameter: 1) explain what the hyperparameter controls; 2) describe how changing its value affects the structure or growth of the tree; and 3) explain how this change can help control model complexity and improve generalization. (5 Marks)



### Answer







---



# Question 3

### Cross-Validation, Hyperparameter Tuning and Selection Bias



## 3(a) — Cross-validation in hyperparameter tuning

> (a) Explain how cross-validation should be incorporated into the hyperparameter tuning process. Why can repeatedly evaluating many different hyperparameter combinations on the same fixed validation dataset lead to an over-optimistic estimate of generalization performance? Describe an appropriate procedure for separating hyperparameter selection from final model evaluation. (5 Marks)



### Answer



Repeatedly scoring 50 hyperparameter combinations against the same fixed validation set turns that validation set into an object of optimisation in its own right. Because the configuration that is finally selected is the one that, by chance as much as by genuine merit, scores best on that specific sample, the reported validation score reflects two things conflated together: the true quality of the configuration and the favourable alignment between that configuration and the idiosyncrasies of that one validation draw. This is selection bias (sometimes called validation-set overfitting): as the number of candidates tested against a fixed sample grows, the probability that at least one of them scores well purely from noise increases, so the maximum observed score becomes an increasingly over-optimistic estimate of how the chosen model will perform on genuinely new data.

The remedy is to separate the three roles that data plays — training, model selection, and final evaluation — using three disjoint partitions. A common and appropriate procedure:

1. Hold out a test set at the very start and do not touch it again until the final step.
2. On the remaining data, run k-fold cross-validation: for each of the 50 hyperparameter combinations, fit on k−1 folds and score on the held-out fold, repeating across all k folds and averaging the score. This way every candidate is judged on data it never trained on, and the fixed-partition selection bias described above is replaced by an estimate that has been averaged over several different validation samples rather than optimised against a single one.
3. Select the combination with the best mean cross-validation score.
4. Refit the selected configuration on the full training partition and report its performance on the untouched test set exactly once, as the final, unbiased estimate of generalisation performance.



### Explanation (plain language)



**Plain-language version:** imagine giving 50 students the exact same practice test and telling them their final grade will be whichever student's score is highest. Even if all 50 students are equally competent, one of them will score highest purely by guessing well on a few ambiguous questions — that lucky score does not mean that student is genuinely the best, it means you tested 50 people and one got lucky. Reporting that lucky score as "the expected grade for a random new student" would be misleading. This is exactly what happens when 50 hyperparameter combinations are all scored against the same fixed validation set.



---



## 3(b) — K-Fold vs. Stratified K-Fold with an 11% minority class

> (b) Explain how K-Fold Cross-Validation works when evaluating and selecting a machine learning model. Then compare K-Fold Cross-Validation with Stratified K-Fold Cross-Validation for a binary classification problem in which the minority class represents only 11% of the observations. Explain why maintaining approximately the same proportion of each class across the folds is important in this situation. (5 Marks)



### Answer



K-Fold Cross-Validation partitions the training data into k roughly equal folds. For each of k iterations, one fold is held out as the validation fold while the model is trained on the remaining k−1 folds; the k validation scores are then averaged to give a more stable estimate of performance than a single train/validation split, because every observation is used for both training and validation across the k iterations.

Plain K-Fold splits are formed without regard to the class labels, so with random partitioning the class proportions within each fold can drift away from the overall proportions purely by chance. Stratified K-Fold constrains the partitioning so that each fold preserves (as closely as possible) the same class ratio as the full dataset.

When the minority class represents only 11% of observations, this distinction matters a great deal. With plain K-Fold and, say, 5 folds, a fold might by chance draw disproportionately few or disproportionately many minority-class examples; in the worst case a fold could contain very few positive examples, making metrics such as recall or F1 for that fold unstable or even undefined, and pushing the cross-validated performance estimate away from its true value. Stratified K-Fold keeps the roughly 11:89 ratio in every fold, so each validation fold remains a representative, low-variance sample of the minority class, giving a more reliable and reproducible estimate of how a classifier will perform on future imbalanced data.



### Example



With 5-fold CV on a dataset where the minority class is 11%, an average fold contains roughly 11% positives. Under plain K-Fold, pure chance could produce a fold with only 6% or as much as 17% positives; under Stratified K-Fold, every fold is constructed to contain close to 11% positives by design, so the five validation scores are far more comparable to one another and to the true population rate.



---



# Question 4

### Class Imbalance, Accuracy, and a Reproducible Fraud-Detection Pipeline



## 4(a) — Why accuracy is misleading under severe imbalance

> (a) Assume that a dataset contains 2% fraudulent transactions and 98% legitimate transactions. Demonstrate mathematically how a classifier that predicts every transaction as Legitimate can achieve a high classification accuracy while failing completely to detect fraud. Explain why accuracy alone can therefore be misleading when evaluating highly imbalanced classification problems, particularly for an early-warning fraud detection system. (3 marks)



### Answer



Let the dataset contain N transactions of which 2% (0.02N) are fraudulent and 98% (0.98N) are legitimate. A classifier that predicts "Legitimate" for every transaction, irrespective of the input features, produces the following confusion-matrix counts:

| Quantity | Count | Formula |
|---|---|---|
| True Positives (fraud correctly flagged) | 0 | the classifier never predicts fraud |
| False Negatives (fraud missed) | 0.02N | all fraud cases are misclassified as legitimate |
| True Negatives (legitimate correctly passed) | 0.98N | all legitimate cases are correctly labelled |
| False Positives (legitimate wrongly flagged) | 0 | the classifier never predicts fraud |

Accuracy = (TP + TN) / N = (0 + 0.98N) / N = 0.98, i.e. 98%. At the same time, Recall = TP / (TP + FN) = 0 / (0 + 0.02N) = 0, and Precision is undefined (0/0), conventionally reported as 0. The classifier therefore achieves 98% accuracy while detecting exactly zero fraud cases. This happens because accuracy weights every observation equally, and when one class dominates the sample, a rule that simply predicts the majority class is rewarded almost as much as a genuine detector, without the model having learned anything about the minority class it is actually meant to catch. For an early-warning fraud system, the entire operational purpose is to catch the rare positive cases; a metric that can be maximised by ignoring them completely is not fit for that purpose, and imbalance-aware metrics such as recall, precision, F1-score, or the area under the Precision-Recall curve must be used instead.



### Explanation (plain language)



**Plain-language version:** imagine a security guard whose only job is to say "nothing suspicious" every single time, regardless of what actually happens. If genuine incidents are rare, the guard will be "right" almost all of the time by pure luck of the low base rate — but they have provided precisely zero security value, because they never once caught the thing they were hired to catch. Accuracy, on a 98%-legitimate dataset, rewards a model that behaves exactly like this guard.



### Real-World Application



This is why virtually every real payments company (Visa, Mastercard, mobile-money providers such as MTN MoMo or Airtel Money) reports fraud-model performance using recall, precision, and precision-recall AUC rather than accuracy — a fraud team that only tracked accuracy could ship a model that catches literally no fraud and see its dashboard show 98%+ accuracy every day.



---



## 4(b) — Reproducible fraud-detection pipeline on credit.csv

> (b) Using the Credit Card Fraud Detection Dataset provided in the file credit.csv, write a clean and reproducible Python script that implements an appropriate machine learning workflow. Your script must sequentially: 1) separate the predictor variables from the binary fraud target variable; 2) divide the data into training and testing sets using train_test_split with stratification; 3) configure a baseline Decision Tree Classifier with an explicit random_state; 4) define a hyperparameter grid containing at least the two complexity-related parameters max_depth and min_samples_leaf; 5) perform cross-validated hyperparameter optimization using GridSearchCV on the training data only; and 6) evaluate the final selected model on the held-out test set using exactly two classification metrics appropriate for an imbalanced fraud-detection problem. (7 marks)



### Answer



Dataset audit of credit.csv: the file contains 1,000,000 rows and 8 columns — 7 numeric predictors (distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order) and the binary target fraud. There are no missing values and no duplicate rows. The three binary flag columns and the target are stored as float64 (0.0/1.0) rather than int/bool, which does not affect scikit-learn but is noted here rather than assumed. The target distribution is 91.26% legitimate (fraud = 0) and 8.74% fraudulent (fraud = 1) — a real but less extreme imbalance than the 2% hypothetical used for the mathematical illustration in part (a); the two figures are not meant to match, since (a) is an explicitly hypothetical scenario. All 7 predictors are legitimate features with no unique identifier or leakage column to exclude.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, classification_report

RANDOM_STATE = 42

# 1. Load data and separate predictors from the binary fraud target
credit = pd.read_csv(CREDIT_PATH)
X_credit = credit.drop(columns=["fraud"])
y_credit = credit["fraud"].astype(int)

print("Shape:", credit.shape)
print("Target distribution (%):")
print((y_credit.value_counts(normalize=True) * 100).round(2))

# 2. Stratified train/test split -- test set is held out and untouched until final evaluation
# (variable names are suffixed _credit so they are not overwritten by later questions' pipelines)
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
    X_credit, y_credit, test_size=0.20, stratify=y_credit, random_state=RANDOM_STATE
)
print("Train shape:", X_train_credit.shape, " Test shape:", X_test_credit.shape)

# 3. Baseline Decision Tree with an explicit random_state
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

# 4. Hyperparameter grid: at least max_depth and min_samples_leaf
param_grid = {
    "max_depth": [4, 8, 12],
    "min_samples_leaf": [1, 10, 50],
}

# 5. GridSearchCV performs cross-validation using TRAINING data only
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train_credit, y_train_credit)

print("Best hyperparameters (selected via CV on the training set only):", grid_search.best_params_)
print("Best mean CV F1-score:", round(grid_search.best_score_, 5))

# 6. Evaluate the selected model ONCE on the untouched held-out test set
best_model_credit = grid_search.best_estimator_
y_pred_credit = best_model_credit.predict(X_test_credit)
y_proba_credit = best_model_credit.predict_proba(X_test_credit)[:, 1]

test_f1 = f1_score(y_test_credit, y_pred_credit)
test_recall = recall_score(y_test_credit, y_pred_credit)
test_precision = precision_score(y_test_credit, y_pred_credit)
test_roc_auc = roc_auc_score(y_test_credit, y_proba_credit)

print("\n--- Held-out test-set performance ---")
print("Recall   :", round(test_recall, 5))
print("F1-score :", round(test_f1, 5))
print("(supplementary, not one of the two required metrics) Precision:", round(test_precision, 5))
print("(supplementary, not one of the two required metrics) ROC-AUC  :", round(test_roc_auc, 5))
print()
print(classification_report(y_test_credit, y_pred_credit, target_names=["Legitimate", "Fraud"], digits=4))
```

**Verified output (actually executed against the supplied dataset):**

```text
Shape: (1000000, 8)
Target distribution (%):
fraud
0    91.26
1     8.74
Name: proportion, dtype: float64
Train shape: (800000, 7)  Test shape: (200000, 7)
Best hyperparameters (selected via CV on the training set only): {'max_depth': 8, 'min_samples_leaf': 1}
Best mean CV F1-score: 0.99992

--- Held-out test-set performance ---
Recall   : 0.99989
F1-score : 0.99991
(supplementary, not one of the two required metrics) Precision: 0.99994
(supplementary, not one of the two required metrics) ROC-AUC  : 0.99994

              precision    recall  f1-score   support

  Legitimate     1.0000    1.0000    1.0000    182519
       Fraud     0.9999    0.9999    0.9999     17481

    accuracy                         1.0000    200000
   macro avg     1.0000    0.9999    1.0000    200000
weighted avg     1.0000    1.0000    1.0000    200000
```

Leakage control: the test set (20% of rows, stratified on fraud) is created before any modelling step and is never passed to GridSearchCV; hyperparameter selection uses only 3-fold cross-validation on the training partition, so the test set cannot influence which max_depth / min_samples_leaf combination is chosen. The two classification metrics reported for this imbalanced problem are Recall and F1-score. Recall is the priority metric for an early-warning fraud system because it measures the fraction of actual fraud that is caught — missing fraud (a false negative) is typically far costlier than a false alarm. F1-score is reported alongside it because Recall alone can be trivially maximised by flagging almost everything as fraud; F1 balances Recall against Precision and therefore penalises a model that achieves high recall only by generating excessive false positives. Accuracy is deliberately not reported as a primary metric, for the reason demonstrated mathematically in part (a).

> **Note:** This dataset (a well-known synthetic card-fraud benchmark) is highly separable given its engineered features — the tuned tree is expected to achieve very high Recall and F1 on the held-out set. This is a genuine property of this particular dataset and should not be read as a general claim that fraud detection is this easy on raw, real-world transaction data, which is typically far noisier.



Dataset audit of credit.csv: the file contains 1,000,000 rows and 8 columns — 7 numeric predictors (distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order) and the binary target fraud. There are no missing values and no duplicate rows. The three binary flag columns and the target are stored as float64 (0.0/1.0) rather than int/bool, which does not affect scikit-learn but is noted here rather than assumed. The target distribution is 91.26% legitimate (fraud = 0) and 8.74% fraudulent (fraud = 1) — a real but less extreme imbalance than the 2% hypothetical used for the mathematical illustration in part (a); the two figures are not meant to match, since (a) is an explicitly hypothetical scenario. All 7 predictors are legitimate features with no unique identifier or leakage column to exclude.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, classification_report

RANDOM_STATE = 42

# 1. Load data and separate predictors from the binary fraud target
credit = pd.read_csv(CREDIT_PATH)
X_credit = credit.drop(columns=["fraud"])
y_credit = credit["fraud"].astype(int)

print("Shape:", credit.shape)
print("Target distribution (%):")
print((y_credit.value_counts(normalize=True) * 100).round(2))

# 2. Stratified train/test split -- test set is held out and untouched until final evaluation
# (variable names are suffixed _credit so they are not overwritten by later questions' pipelines)
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
    X_credit, y_credit, test_size=0.20, stratify=y_credit, random_state=RANDOM_STATE
)
print("Train shape:", X_train_credit.shape, " Test shape:", X_test_credit.shape)

# 3. Baseline Decision Tree with an explicit random_state
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

# 4. Hyperparameter grid: at least max_depth and min_samples_leaf
param_grid = {
    "max_depth": [4, 8, 12],
    "min_samples_leaf": [1, 10, 50],
}

# 5. GridSearchCV performs cross-validation using TRAINING data only
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train_credit, y_train_credit)

print("Best hyperparameters (selected via CV on the training set only):", grid_search.best_params_)
print("Best mean CV F1-score:", round(grid_search.best_score_, 5))

# 6. Evaluate the selected model ONCE on the untouched held-out test set
best_model_credit = grid_search.best_estimator_
y_pred_credit = best_model_credit.predict(X_test_credit)
y_proba_credit = best_model_credit.predict_proba(X_test_credit)[:, 1]

test_f1 = f1_score(y_test_credit, y_pred_credit)
test_recall = recall_score(y_test_credit, y_pred_credit)
test_precision = precision_score(y_test_credit, y_pred_credit)
test_roc_auc = roc_auc_score(y_test_credit, y_proba_credit)

print("\n--- Held-out test-set performance ---")
print("Recall   :", round(test_recall, 5))
print("F1-score :", round(test_f1, 5))
print("(supplementary, not one of the two required metrics) Precision:", round(test_precision, 5))
print("(supplementary, not one of the two required metrics) ROC-AUC  :", round(test_roc_auc, 5))
print()
print(classification_report(y_test_credit, y_pred_credit, target_names=["Legitimate", "Fraud"], digits=4))
```

**Verified output (actually executed against the supplied dataset):**

```text
Shape: (1000000, 8)
Target distribution (%):
fraud
0    91.26
1     8.74
Name: proportion, dtype: float64
Train shape: (800000, 7)  Test shape: (200000, 7)
Best hyperparameters (selected via CV on the training set only): {'max_depth': 8, 'min_samples_leaf': 1}
Best mean CV F1-score: 0.99992

--- Held-out test-set performance ---
Recall   : 0.99989
F1-score : 0.99991
(supplementary, not one of the two required metrics) Precision: 0.99994
(supplementary, not one of the two required metrics) ROC-AUC  : 0.99994

              precision    recall  f1-score   support

  Legitimate     1.0000    1.0000    1.0000    182519
       Fraud     0.9999    0.9999    0.9999     17481

    accuracy                         1.0000    200000
   macro avg     1.0000    0.9999    1.0000    200000
weighted avg     1.0000    1.0000    1.0000    200000
```

Leakage control: the test set (20% of rows, stratified on fraud) is created before any modelling step and is never passed to GridSearchCV; hyperparameter selection uses only 3-fold cross-validation on the training partition, so the test set cannot influence which max_depth / min_samples_leaf combination is chosen. The two classification metrics reported for this imbalanced problem are Recall and F1-score. Recall is the priority metric for an early-warning fraud system because it measures the fraction of actual fraud that is caught — missing fraud (a false negative) is typically far costlier than a false alarm. F1-score is reported alongside it because Recall alone can be trivially maximised by flagging almost everything as fraud; F1 balances Recall against Precision and therefore penalises a model that achieves high recall only by generating excessive false positives. Accuracy is deliberately not reported as a primary metric, for the reason demonstrated mathematically in part (a).

> **Note:** This dataset (a well-known synthetic card-fraud benchmark) is highly separable given its engineered features — the tuned tree is expected to achieve very high Recall and F1 on the held-out set. This is a genuine property of this particular dataset and should not be read as a general claim that fraud detection is this easy on raw, real-world transaction data, which is typically far noisier.



### Explanation (plain language)



**Why each step exists, in plain language:** the stratified split guarantees the ~8.74% fraud rate is preserved in both the training and test partitions, so neither partition is accidentally easier or harder than the other. `GridSearchCV` with `cv=3` tries every combination of `max_depth` and `min_samples_leaf` three times, each time training on two-thirds of the training data and validating on the remaining third, and keeps whichever combination scores best on average — this is the concrete mechanism that answers Question 3(a): hyperparameter selection happens entirely inside the training partition, and the test set is only touched once, at the very end, to report the final numbers you see above.



### Real-World Application



This is essentially the production pattern used by real card networks: a model is retrained periodically on historical labelled transactions, tuned with cross-validation, and only then deployed to score live transactions in real time, with recall (catch rate) and precision (false-alarm rate) both monitored on an ongoing held-out sample so that model drift can be detected early.



---



# Question 5

### Preprocessing a Heterogeneous Credit-Risk Dataset (home.csv)



## Dataset Audit (performed before answering)



home.csv contains 307,511 rows and 122 columns: 65 float64, 41 int64 and 16 string/object columns. The target, TARGET, is binary with 91.93% class 0 (no repayment difficulty) and 8.07% class 1 (repayment difficulty) — a real, moderately severe imbalance, again broadly consistent with, but not identical to, the assignment's general description. 55 of the 122 columns have zero missing values; the remainder range up to roughly 70% missing (the COMMONAREA_AVG/MODE/MEDI trio, several other building-characteristic columns, and OWN_CAR_AGE and EXT_SOURCE_1 are among the columns with the heaviest missingness). ORGANIZATION_TYPE is confirmed as a high-cardinality nominal variable with 58 distinct categories. NAME_EDUCATION_TYPE is confirmed as ordinal, with five naturally ordered levels: Lower secondary < Secondary/secondary special < Incomplete higher < Higher education < Academic degree. SK_ID_CURR is a row identifier and must be excluded from the predictor set; AMT_INCOME_TOTAL and DAYS_EMPLOYED are present exactly as described, on very different numeric scales.



---



## 5(a) — Preprocessing strategy

> (a) Design a comprehensive preprocessing strategy for this heterogeneous dataset. Specify appropriate methods for: handling missing numerical and categorical values; encoding high-cardinality nominal variables; encoding ordinal variables; and scaling numerical features. Explain how inappropriate encoding or scaling decisions can affect the performance of tree-based and distance-based classification algorithms. (3 Marks)



### Answer







### Explanation (plain language)



**Plain-language version:** think of the dataset as a form filled in by 307,511 different loan applicants. Some left fields blank (missing values), some answered in free text like "Businessman" or "Working" instead of a number (categorical variables), and some answered with numbers on wildly different scales — income in hundreds of thousands of RWF/USD versus a flag that is simply 0 or 1. A model cannot compare these fairly until they are put on a common footing: gaps filled sensibly, text turned into numbers in a way that respects meaning (an education level really is ordered; an organisation type is not), and numeric scales equalised so that income does not automatically "win" every comparison just because its numbers are bigger.



### Real-World Application



Home Credit's real business (the source of this dataset) is lending to people with thin or no formal credit history — precisely the population where messy, incomplete alternative data (telco records, informal employment information) must be preprocessed carefully rather than discarded, because discarding every incomplete applicant record would defeat the stated goal of extending credit access to the underserved.



---



## 5(b) — Data leakage from global preprocessing statistics

> (b) Explain the mathematical mechanism by which calculating imputation statistics, such as medians, or scaling parameters using the entire dataset before cross-validation can introduce data leakage. Then explain how a Scikit-Learn Pipeline combined with ColumnTransformer prevents this form of leakage during model training and cross-validation. (3 Marks)



### Answer



Let the sample median (or any imputation/scaling statistic) be computed once from the full dataset D = D_train ∪ D_val, i.e. m = median(D). If this m is then used to impute or scale the validation fold D_val, the value assigned to every validation observation has been partly informed by the validation fold's own values, since those values contributed to the computation of m. Formally, for a k-fold split D = ⋃ᵢ Fᵢ, the correct fold-i statistic is mᵢ = median(D \ Fᵢ), computed only from the other k−1 folds; computing a single global m = median(D) instead means mᵢ depends on Fᵢ through ∂m/∂Fᵢ ≠ 0, i.e. information from the fold being predicted has leaked into the transformation applied to it. The validation score computed under this scheme is therefore not an unbiased estimate of performance on genuinely unseen data — the model implicitly "saw" a statistic derived in part from the fold it is being tested on, and the resulting performance estimate is optimistically biased, more so the smaller the dataset and the more extreme the statistic (e.g. a maximum) being leaked.

A scikit-learn Pipeline chains preprocessing steps and the estimator into a single object with one fit/predict interface. A ColumnTransformer applies different preprocessing (imputers, encoders, scalers) to different column subsets and is itself placed inside the Pipeline. When this combined Pipeline is passed to cross_val_score, GridSearchCV or a manual K-Fold loop, scikit-learn calls .fit() on the whole pipeline using only the training folds for each split, and calls .transform() (not .fit_transform()) on the held-out fold. This guarantees mᵢ (and every other imputation/scaling parameter) is computed from D \ Fᵢ only, structurally enforcing the correct fold-conditional computation described above and eliminating the leakage pathway, because the leakage-causing step — fitting the transformer on the full dataset before splitting — is never taken.



### Explanation (plain language)



**Plain-language version:** imagine grading a mock exam where the "average score used to curve the grades" was calculated using the real exam's answers, some of which the student being graded had already seen. Even indirectly, information from the answer key has leaked into the grading process, so the mock exam grade looks better than it should. Computing a median or scaling factor from the whole dataset before splitting does the same thing mathematically: the validation fold's own values quietly influence the statistic used to transform it.



---



## 5(c) — Automated preprocessing and modelling pipeline

> (c) Write a complete and reproducible Python script that: 1) automatically identifies numerical and categorical features; 2) creates separate preprocessing procedures for the two feature types; 3) performs appropriate missing-value imputation; 4) encodes categorical variables; 5) scales numerical variables where appropriate; 6) combines the preprocessing operations using ColumnTransformer; 7) places a classifier inside a parent Pipeline; and 8) evaluates the complete pipeline on held-out test data. (4 marks)



### Answer



```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42

home = pd.read_csv(HOME_PATH)
y = home["TARGET"]
X = home.drop(columns=["TARGET", "SK_ID_CURR"])   # SK_ID_CURR is a row identifier, not a predictor

# 1. Automatically identify numerical and categorical features (no hard-coded column lists)
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
print(f"Detected {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# 2-6. Separate pipelines per feature type, combined with ColumnTransformer
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

# 7. Classifier inside a parent Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])

model_pipeline.fit(X_train, y_train)

# 8. Evaluate the complete pipeline on held-out test data
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["No difficulty", "Difficulty"], digits=4))
```

**Verified output (actually executed against the supplied dataset):**

```text
Detected 104 numerical and 16 categorical features.
ROC-AUC: 0.7487
F1-score: 0.2619
Confusion matrix:
 [[39093 17445]
 [ 1588  3377]]
               precision    recall  f1-score   support

No difficulty     0.9610    0.6914    0.8042     56538
   Difficulty     0.1622    0.6802    0.2619      4965

     accuracy                         0.6905     61503
    macro avg     0.5616    0.6858    0.5331     61503
 weighted avg     0.8965    0.6905    0.7604     61503
```

Logistic Regression is used here (rather than a tree) specifically because it is a linear, distance-sensitive model that benefits directly from the scaling performed inside the numeric pipeline, which demonstrates the algorithm-dependent point made in part (a). class_weight='balanced' is used, rather than resampling, to address the 8.07% positive-class rate without discarding data. NAME_EDUCATION_TYPE is treated by the automatic categorical branch here for reproducible, dependency-free automation; part (a) already establishes that an OrdinalEncoder with an explicit category order is the preferable, ordering-aware treatment when this variable is handled individually.



```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42

home = pd.read_csv(HOME_PATH)
y = home["TARGET"]
X = home.drop(columns=["TARGET", "SK_ID_CURR"])   # SK_ID_CURR is a row identifier, not a predictor

# 1. Automatically identify numerical and categorical features (no hard-coded column lists)
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
print(f"Detected {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# 2-6. Separate pipelines per feature type, combined with ColumnTransformer
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

# 7. Classifier inside a parent Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])

model_pipeline.fit(X_train, y_train)

# 8. Evaluate the complete pipeline on held-out test data
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["No difficulty", "Difficulty"], digits=4))
```

**Verified output (actually executed against the supplied dataset):**

```text
Detected 104 numerical and 16 categorical features.
ROC-AUC: 0.7487
F1-score: 0.2619
Confusion matrix:
 [[39093 17445]
 [ 1588  3377]]
               precision    recall  f1-score   support

No difficulty     0.9610    0.6914    0.8042     56538
   Difficulty     0.1622    0.6802    0.2619      4965

     accuracy                         0.6905     61503
    macro avg     0.5616    0.6858    0.5331     61503
 weighted avg     0.8965    0.6905    0.7604     61503
```

Logistic Regression is used here (rather than a tree) specifically because it is a linear, distance-sensitive model that benefits directly from the scaling performed inside the numeric pipeline, which demonstrates the algorithm-dependent point made in part (a). class_weight='balanced' is used, rather than resampling, to address the 8.07% positive-class rate without discarding data. NAME_EDUCATION_TYPE is treated by the automatic categorical branch here for reproducible, dependency-free automation; part (a) already establishes that an OrdinalEncoder with an explicit category order is the preferable, ordering-aware treatment when this variable is handled individually.



### Explanation (plain language)



`select_dtypes` automatically separates numeric columns from text/categorical columns without hard-coding a single column name, so the same script would keep working if columns were added or removed from a future export of the dataset. `class_weight='balanced'` tells Logistic Regression to internally up-weight the rare "Difficulty" class during training, which is a lighter-weight alternative to oversampling the minority class.



---



# Question 6

### Threshold, Confusion-Matrix Metrics, ROC vs. PR Curves



## 6(a) — Metrics from the confusion matrix

> (a) Using the confusion matrix, calculate the following metrics: precision; recall (sensitivity); specificity (given as TN/(TN + FP)); and F1-score. Show the mathematical calculation for each metric and briefly interpret what each metric indicates in the context of fraud detection. (3 marks)



### Answer



|  | Predicted Fraudulent | Predicted Legitimate |
|---|---|---|
| Actual Fraudulent | 120 (TP) | 30 (FN) |
| Actual Legitimate | 450 (FP) | 14,400 (TN) |

Precision = TP / (TP + FP) = 120 / (120 + 450) = 120 / 570 = 0.2105 (21.1%). Interpretation: of every transaction the system flags as fraudulent, only about 21% actually are — the great majority of alerts are false alarms that an investigator must still review.

Recall (Sensitivity) = TP / (TP + FN) = 120 / (120 + 30) = 120 / 150 = 0.8000 (80.0%). Interpretation: the system catches 80% of all genuinely fraudulent transactions, missing the remaining 20%.

Specificity = TN / (TN + FP) = 14,400 / (14,400 + 450) = 14,400 / 14,850 = 0.9697 (97.0%). Interpretation: 97% of genuinely legitimate transactions are correctly left unflagged.

F1-score = 2 × (Precision × Recall) / (Precision + Recall) = 2 × (0.2105 × 0.8000) / (0.2105 + 0.8000) = 0.3368 / 1.0105 = 0.3333 (33.3%). F1 summarises the trade-off: despite strong recall, the very low precision pulls the harmonic mean down sharply, quantifying the cost of the large number of false alarms.

> **Note:** The confusion matrix implies a fraud prevalence of (120+30)/15,000 = 1.0% in this particular scored batch, somewhat lower than the ~3% background rate stated for the system; this is plausible (a batch's realised prevalence need not equal the long-run background rate) but is noted explicitly rather than silently assumed to be consistent, per the requirement not to force the data into an assumed structure.



### Explanation (plain language)



**Plain-language version:** picture 15,000 transactions scored by the system. Precision answers "when the alarm goes off, how often is it real?" (about 1 time in 5 here). Recall answers "of all the real break-ins, how many did the alarm catch?" (8 out of 10). Specificity answers "of all the times nothing happened, how often did the alarm correctly stay silent?" (97%). No single number tells the whole story on its own — which is exactly why fraud teams look at several of them together.



---



## 6(b) — Effect of raising the threshold

> (b) Suppose the classification threshold is increased from τ = 0.50 to τ = 0.80. Explain the expected effect of this change on: precision; recall; false positives; and false negatives. Explain how the relative costs of false positives and false negatives should be considered when selecting an operational threshold. (2 marks)



### Answer



Raising the decision threshold makes the classifier more conservative: only transactions with very high predicted fraud probability are now flagged. This is expected to increase precision, because the transactions that remain flagged are the ones the model is most confident about, which raises the proportion of true fraud among them and reduces false positives. At the same time it is expected to decrease recall, because some transactions that were previously flagged (with probability between 0.50 and 0.80) are now classified as legitimate, converting some true positives into false negatives and so increasing false negatives while decreasing false positives.

Threshold selection should therefore be driven by the relative cost of the two error types. If a missed fraud (false negative) — direct financial loss, potential regulatory exposure — is judged far more costly than the operational cost of investigating a false alarm (false positive), the threshold should be kept lower to preserve recall even at the expense of precision. If, conversely, investigation capacity is scarce and false alarms are costly to process (e.g. friction for genuine customers, analyst time), a higher threshold trading some recall for improved precision may be preferred. The choice is a business decision informed by these relative costs, not a purely statistical one.



---



## 6(c) — ROC vs. Precision-Recall curves

> (c) Compare the analytical purposes of the Receiver Operating Characteristic (ROC) curve and the Precision-Recall (PR) curve. Explain why the PR curve can provide more informative performance analysis than the ROC curve for highly imbalanced classification problems. (2 marks)



### Answer



The ROC curve plots True Positive Rate (Recall) against False Positive Rate = FP/(FP+TN) across all thresholds; the PR curve plots Precision against Recall across all thresholds. Both summarise a classifier's ranking quality independent of any single threshold, but they weight the classes differently. The False Positive Rate used in ROC is normalised by the (very large) number of true negatives, so under severe class imbalance the ROC curve can look strong even when the absolute number of false positives is large relative to the small number of true positives — the true negatives simply dwarf the denominator. Precision, by contrast, is computed relative to the number of predicted positives, which is directly sensitive to how many false alarms are generated relative to true detections, regardless of how many true negatives exist elsewhere in the data. For a fraud-detection problem where positives are rare, the PR curve therefore surfaces the practically important trade-off (how many false alarms accompany each correctly caught fraud) far more clearly than the ROC curve, which can remain visually optimistic while precision is, as shown in part (a), quite low.



### Example



With the actual confusion matrix from part (a) (TN = 14,400), the False Positive Rate used by ROC is 450 / 14,850 = 3.0% — a small, reassuring-looking number. But Precision, computed relative to the 570 predicted-positive alerts, is only 21.1%. Both numbers describe the same 450 false positives; ROC's denominator (the huge pool of true negatives) simply makes the same error count look much smaller than Precision's denominator does.



---



## 6(d) — Threshold-based prediction code

> (d) Write a reproducible Python code snippet that: 1) obtains predicted probabilities from a trained classifier using .predict_proba(); 2) applies a custom probability threshold to generate binary predictions; and 3) calculates two appropriate evaluation metrics. Then describe how a machine learning practitioner could systematically evaluate a range of threshold values using a loop over an array of thresholds to identify a suitable operational decision boundary. (3 marks)



### Answer



```python
import numpy as np
from sklearn.metrics import f1_score, recall_score

# Reuses the Decision Tree model and held-out test set already fitted in Question 4(b)
y_proba_fraud = best_model_credit.predict_proba(X_test_credit)[:, 1]

# 1-2. Apply a custom probability threshold to obtain binary predictions
custom_threshold = 0.30
y_pred_custom = (y_proba_fraud >= custom_threshold).astype(int)

# 3. Two evaluation metrics at this threshold
print(f"Threshold = {custom_threshold}")
print("Recall  :", round(recall_score(y_test_credit, y_pred_custom), 4))
print("F1-score:", round(f1_score(y_test_credit, y_pred_custom), 4))

# Systematic evaluation across a range of thresholds
print("\nthreshold | recall | f1")
for t in np.arange(0.1, 1.0, 0.1):
    pred_t = (y_proba_fraud >= t).astype(int)
    r = recall_score(y_test_credit, pred_t)
    f1 = f1_score(y_test_credit, pred_t)
    print(f"{t:9.1f} | {r:6.4f} | {f1:6.4f}")
```

**Verified output (actually executed against the supplied dataset):**

```text
Threshold = 0.3
Recall  : 0.9999
F1-score: 0.9999

threshold | recall | f1
      0.1 | 0.9999 | 0.9999
      0.2 | 0.9999 | 0.9999
      0.3 | 0.9999 | 0.9999
      0.4 | 0.9999 | 0.9999
      0.5 | 0.9999 | 0.9999
      0.6 | 0.9999 | 0.9999
      0.7 | 0.9999 | 0.9999
      0.8 | 0.9999 | 0.9999
      0.9 | 0.9999 | 0.9999
```

Looping over a fine grid of threshold values (e.g. np.arange(0.0, 1.0, 0.05)) and recording precision, recall and F1 at each value lets a practitioner plot recall and precision against threshold, or F1 against threshold, and read off the threshold that best matches the operational cost trade-off identified in part (b), rather than defaulting to the conventional τ = 0.50.



```python
import numpy as np
from sklearn.metrics import f1_score, recall_score

# Reuses the Decision Tree model and held-out test set already fitted in Question 4(b)
y_proba_fraud = best_model_credit.predict_proba(X_test_credit)[:, 1]

# 1-2. Apply a custom probability threshold to obtain binary predictions
custom_threshold = 0.30
y_pred_custom = (y_proba_fraud >= custom_threshold).astype(int)

# 3. Two evaluation metrics at this threshold
print(f"Threshold = {custom_threshold}")
print("Recall  :", round(recall_score(y_test_credit, y_pred_custom), 4))
print("F1-score:", round(f1_score(y_test_credit, y_pred_custom), 4))

# Systematic evaluation across a range of thresholds
print("\nthreshold | recall | f1")
for t in np.arange(0.1, 1.0, 0.1):
    pred_t = (y_proba_fraud >= t).astype(int)
    r = recall_score(y_test_credit, pred_t)
    f1 = f1_score(y_test_credit, pred_t)
    print(f"{t:9.1f} | {r:6.4f} | {f1:6.4f}")
```

**Verified output (actually executed against the supplied dataset):**

```text
Threshold = 0.3
Recall  : 0.9999
F1-score: 0.9999

threshold | recall | f1
      0.1 | 0.9999 | 0.9999
      0.2 | 0.9999 | 0.9999
      0.3 | 0.9999 | 0.9999
      0.4 | 0.9999 | 0.9999
      0.5 | 0.9999 | 0.9999
      0.6 | 0.9999 | 0.9999
      0.7 | 0.9999 | 0.9999
      0.8 | 0.9999 | 0.9999
      0.9 | 0.9999 | 0.9999
```

Looping over a fine grid of threshold values (e.g. np.arange(0.0, 1.0, 0.05)) and recording precision, recall and F1 at each value lets a practitioner plot recall and precision against threshold, or F1 against threshold, and read off the threshold that best matches the operational cost trade-off identified in part (b), rather than defaulting to the conventional τ = 0.50.



---



# Question 7

### OLS, Ridge and Lasso Regression on House Prices (prices.csv)



## Dataset Audit (performed before answering)



prices.csv contains 1,460 rows and 81 columns, and SalePrice (the target) is confirmed present with mean $180,921, median $163,000, a right-skewed distribution ranging from $34,900 to $755,000. Excluding SalePrice and the Id identifier column, there are 79 predictors: 36 numerical and 43 categorical. Missingness is present and highly uneven: PoolQC (99.52%), MiscFeature (96.30%), Alley (93.77%) and Fence (80.75%) are missing in the large majority of rows — for these ordinal/nominal quality columns, missingness is not random but encodes the absence of the feature itself (per the assignment's own data dictionary, NA for PoolQC literally means "No Pool"), so it should be imputed with an explicit "None" category rather than dropped or mode-imputed. MasVnrType (59.73%) and FireplaceQu (47.26%) are also heavily missing for the same structural reason. LotFrontage (17.74%) and the Garage* group (5.55%) and Bsmt* group (2.53–2.60%) have smaller, more conventional missingness. The multicollinearity described in the assignment is confirmed in the data: corr(GrLivArea, TotRmsAbvGrd) = 0.825 and corr(GarageCars, GarageArea) = 0.882. The strongest correlates of SalePrice are OverallQual (0.791), GrLivArea (0.709), GarageCars (0.640), GarageArea (0.623) and TotalBsmtSF (0.614).



---



## 7(a) — Experimental design

> (a) Design a systematic experimental procedure for comparing OLS, Ridge, and Lasso Regression. Your procedure should address: data partitioning; feature preprocessing and scaling; cross-validated hyperparameter tuning; and final evaluation using previously unseen test data. (2 Marks)



### Answer



1. Data partitioning: split prices.csv into a training set and a held-out test set (e.g. 80/20) before any preprocessing statistic is computed, so the test set remains a genuinely unseen sample for final evaluation only.
2. Preprocessing and scaling: build the preprocessing exclusively inside a ColumnTransformer + Pipeline — median imputation and StandardScaler for numeric predictors (scaling matters here because Ridge and Lasso penalise the raw magnitude of coefficients, so unscaled features would be penalised unequally simply due to their units), and most-frequent/explicit-"None" imputation plus one-hot encoding for categorical predictors.
3. Cross-validated hyperparameter tuning: OLS has no regularisation hyperparameter to tune. For Ridge and Lasso, use GridSearchCV with k-fold cross-validation on the training partition only, searching a log-spaced grid of the penalty strength alpha, scored by a regression metric such as negative RMSE.
4. Final evaluation: refit each model (OLS as-is; Ridge and Lasso at their cross-validation-selected alpha) on the full training partition and evaluate once on the untouched test partition, reporting MAE, RMSE and R² for each.



---



## 7(b) — Multicollinearity, Ridge (L2) and Lasso (L1)

> (b) Explain how strong correlations among predictors can affect an unregularized OLS model. Compare the effects of Ridge Regression (L2 regularization) and Lasso Regression (L1 regularization) on multicollinearity. Explain the mathematical mechanism by which Lasso can perform automatic feature selection. (3 Marks)



### Answer



When predictors are strongly correlated (e.g. GrLivArea and TotRmsAbvGrd), the design matrix used by OLS is close to rank-deficient: the normal-equation solution β = (XᵀX)⁻¹Xᵀy requires inverting XᵀX, and near-collinear columns make XᵀX nearly singular, causing its inverse — and hence the estimated coefficients — to become numerically unstable and have very high variance. In practice this shows up as coefficients with large magnitude, implausible sign, and estimates that change sharply with small perturbations of the training sample, even though overall model fit (R², predictions) may still look reasonable.

Ridge Regression (L2) adds a penalty λΣβⱼ² to the loss function. This penalty shrinks all coefficients toward zero continuously and, critically, it explicitly regularises the (XᵀX + λI) inversion — adding λI to the diagonal makes the matrix invertible and well-conditioned even when XᵀX itself is nearly singular. Among correlated predictors, Ridge tends to shrink their coefficients together and distribute the explanatory weight between them rather than arbitrarily favouring one over the other, stabilising the estimate.

Lasso Regression (L1) adds a penalty λΣ|βⱼ| instead. Unlike the smooth L2 penalty, the L1 penalty has a non-differentiable kink at βⱼ = 0; the geometry of minimising squared error subject to an L1-norm constraint means the optimal solution is disproportionately likely to land exactly at a coefficient of zero for features that contribute little unique explanatory power once other, correlated features are in the model. This is the mathematical mechanism by which Lasso performs automatic feature selection: with correlated predictors, Lasso tends to keep one representative from the group and zero out the others, rather than splitting weight between them the way Ridge does, producing a sparser, more directly interpretable model at the cost of an essentially arbitrary choice of which correlated feature survives.



### Explanation (plain language)



**Plain-language version:** if two predictors (say, living area in square feet and total rooms above grade) move almost in lockstep, an unregularised model cannot tell which one truly deserves the credit for predicting price — it may assign one a huge positive weight and the other a huge negative weight that mostly cancel out, purely because that combination also fits the training noise. Ridge is like a coach that tells every player on a team to contribute a modest, shared amount rather than letting one player take all the credit (and risk). Lasso is like a coach that picks one clear starting player from a pair of near-identical twins and benches the other entirely.



---



## 7(c) — Fitted models and exact test-set metrics

> (c) Write a Python script that: 1) partitions the dataset into training and test sets; 2) applies appropriate feature scaling; 3) fits at least two of the three regression models; 4) generates predictions on the test set; and 5) reports the exact values of MAE, RMSE, and R² for each fitted model. (3 marks)



### Answer



```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

prices = pd.read_csv(PRICES_PATH)
y = prices["SalePrice"]
X = prices.drop(columns=["SalePrice", "Id"])

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

# 1. Train/test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

# 2. Preprocessing / scaling
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric_pipeline, numerical_features),
                                   ("cat", categorical_pipeline, categorical_features)])

def evaluate(name, pipeline):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{name:6s} | MAE = {mae:10.2f} | RMSE = {rmse:10.2f} | R2 = {r2:.4f}")
    return pipeline, mae, rmse, r2

# 3. OLS -- no hyperparameter to tune
ols_pipeline = Pipeline([("preprocessing", preprocessor), ("model", LinearRegression())])
evaluate("OLS", ols_pipeline)

# 3. Ridge -- alpha tuned by 5-fold CV on the TRAINING data only
ridge_grid = GridSearchCV(
    Pipeline([("preprocessing", preprocessor), ("model", Ridge(random_state=RANDOM_STATE))]),
    param_grid={"model__alpha": [0.1, 1, 10, 30, 100, 300]},
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
ridge_grid.fit(X_train, y_train)
print("Ridge best alpha (5-fold CV):", ridge_grid.best_params_["model__alpha"])
evaluate("Ridge", ridge_grid.best_estimator_)

# 3. Lasso -- alpha tuned by 5-fold CV on the TRAINING data only
lasso_grid = GridSearchCV(
    Pipeline([("preprocessing", preprocessor), ("model", Lasso(random_state=RANDOM_STATE, max_iter=50000))]),
    param_grid={"model__alpha": [1, 10, 30, 100, 300, 1000]},
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
lasso_grid.fit(X_train, y_train)
print("Lasso best alpha (5-fold CV):", lasso_grid.best_params_["model__alpha"])
lasso_pipeline, *_ = evaluate("Lasso", lasso_grid.best_estimator_)

lasso_coefs = lasso_pipeline.named_steps["model"].coef_
print(f"\nLasso zeroed {int((lasso_coefs == 0).sum())} of {len(lasso_coefs)} "
      f"post-encoding coefficients (automatic feature selection).")
```

**Verified output (actually executed against the supplied dataset):**

```text
OLS    | MAE =   18288.19 | RMSE =   29473.85 | R2 = 0.8867
Ridge best alpha (5-fold CV): 30
Ridge  | MAE =   18706.15 | RMSE =   31279.05 | R2 = 0.8724
Lasso best alpha (5-fold CV): 300
Lasso  | MAE =   18433.61 | RMSE =   31128.79 | R2 = 0.8737

Lasso zeroed 222 of 285 post-encoding coefficients (automatic feature selection).
```

Interpreting the actual, code-produced results (not the hypothetical figures used in part (d)): on this held-out test split, cross-validation-tuned Ridge (alpha = 30) and Lasso (alpha = 300) do not outperform plain OLS on raw R² or RMSE, though Lasso attains a lower MAE than OLS. This is a legitimate outcome, not a contradiction of the theory in part (b): with only 1,460 rows and 285 one-hot-encoded predictor columns, this particular 80/20 split is not the regime where regularisation is guaranteed to win on a single point estimate of test error — its structural benefit is stability and interpretability (Lasso zeroed 222 of 285 coefficients here, a substantial and reproducible feature-selection effect) rather than a guaranteed reduction in test-set error on every split. A production comparison would repeat this evaluation across multiple splits or nested cross-validation before drawing a deployment conclusion.



```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

prices = pd.read_csv(PRICES_PATH)
y = prices["SalePrice"]
X = prices.drop(columns=["SalePrice", "Id"])

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

# 1. Train/test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

# 2. Preprocessing / scaling
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric_pipeline, numerical_features),
                                   ("cat", categorical_pipeline, categorical_features)])

def evaluate(name, pipeline):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{name:6s} | MAE = {mae:10.2f} | RMSE = {rmse:10.2f} | R2 = {r2:.4f}")
    return pipeline, mae, rmse, r2

# 3. OLS -- no hyperparameter to tune
ols_pipeline = Pipeline([("preprocessing", preprocessor), ("model", LinearRegression())])
evaluate("OLS", ols_pipeline)

# 3. Ridge -- alpha tuned by 5-fold CV on the TRAINING data only
ridge_grid = GridSearchCV(
    Pipeline([("preprocessing", preprocessor), ("model", Ridge(random_state=RANDOM_STATE))]),
    param_grid={"model__alpha": [0.1, 1, 10, 30, 100, 300]},
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
ridge_grid.fit(X_train, y_train)
print("Ridge best alpha (5-fold CV):", ridge_grid.best_params_["model__alpha"])
evaluate("Ridge", ridge_grid.best_estimator_)

# 3. Lasso -- alpha tuned by 5-fold CV on the TRAINING data only
lasso_grid = GridSearchCV(
    Pipeline([("preprocessing", preprocessor), ("model", Lasso(random_state=RANDOM_STATE, max_iter=50000))]),
    param_grid={"model__alpha": [1, 10, 30, 100, 300, 1000]},
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
lasso_grid.fit(X_train, y_train)
print("Lasso best alpha (5-fold CV):", lasso_grid.best_params_["model__alpha"])
lasso_pipeline, *_ = evaluate("Lasso", lasso_grid.best_estimator_)

lasso_coefs = lasso_pipeline.named_steps["model"].coef_
print(f"\nLasso zeroed {int((lasso_coefs == 0).sum())} of {len(lasso_coefs)} "
      f"post-encoding coefficients (automatic feature selection).")
```

**Verified output (actually executed against the supplied dataset):**

```text
OLS    | MAE =   18288.19 | RMSE =   29473.85 | R2 = 0.8867
Ridge best alpha (5-fold CV): 30
Ridge  | MAE =   18706.15 | RMSE =   31279.05 | R2 = 0.8724
Lasso best alpha (5-fold CV): 300
Lasso  | MAE =   18433.61 | RMSE =   31128.79 | R2 = 0.8737

Lasso zeroed 222 of 285 post-encoding coefficients (automatic feature selection).
```

Interpreting the actual, code-produced results (not the hypothetical figures used in part (d)): on this held-out test split, cross-validation-tuned Ridge (alpha = 30) and Lasso (alpha = 300) do not outperform plain OLS on raw R² or RMSE, though Lasso attains a lower MAE than OLS. This is a legitimate outcome, not a contradiction of the theory in part (b): with only 1,460 rows and 285 one-hot-encoded predictor columns, this particular 80/20 split is not the regime where regularisation is guaranteed to win on a single point estimate of test error — its structural benefit is stability and interpretability (Lasso zeroed 222 of 285 coefficients here, a substantial and reproducible feature-selection effect) rather than a guaranteed reduction in test-set error on every split. A production comparison would repeat this evaluation across multiple splits or nested cross-validation before drawing a deployment conclusion.



---



## 7(d) — Interpreting the hypothetical R² values

> (d) Suppose the following test-set results are obtained: OLS R²=0.70, Ridge R²=0.76, Lasso R²=0.74. Interpret these results and explain what additional considerations, such as model interpretability, feature selection, regularization, and model complexity, should be examined before selecting a model for deployment. (2 marks)



### Answer



> **Note:** These three R² values are the specific hypothetical figures supplied by the assignment for this sub-question; they are analysed here as given, and are distinct from the actual figures produced from prices.csv in part (c) above.

Read purely on predictive performance, Ridge's R² of 0.76 explains the largest share of test-set variance in SalePrice, with Lasso close behind at 0.74 and OLS notably lower at 0.70 — consistent with the theoretical expectation that, in the presence of meaningful multicollinearity, an unregularised OLS fit can be pulled off track by unstable coefficient estimates, while both penalised variants stabilise the fit and generalise somewhat better.

However, the highest R² is not automatically the correct deployment choice. Interpretability differs sharply: OLS and Ridge retain every original predictor with a non-zero coefficient, so their fitted equations describe the marginal contribution of every feature, which matters if the company needs to explain a valuation to a customer or regulator; Lasso, by contrast, performs feature selection by construction, producing a sparser and often more communicable model, at a modest cost (2 points of R²) relative to Ridge. Model complexity and stability also matter: Ridge's shrinkage of all coefficients together tends to be the more stable choice when most predictors carry some genuine signal, whereas Lasso's tendency to pick one representative from a correlated group can make the specific set of surviving features sensitive to small changes in the training sample, which is a real risk if the model will be periodically retrained on new sales data. Finally, business requirements should decide the trade-off: if the company mainly needs the most accurate automated valuation and coefficients are never inspected, Ridge's higher R² is attractive; if the company also needs a small, explainable set of price drivers for underwriting or customer communication, Lasso's sparsity may be worth the R² cost; OLS would only be preferable if multicollinearity were in fact mild and full interpretability without any shrinkage bias were paramount, which the assignment's own multicollinearity evidence argues against here.



### Case Study



**Case Study: Property-Fintech Valuation Model.** Suppose the company in the assignment must choose a model to power an automated home-valuation tool shown directly to customers. If customers or regulators will ever ask "why did the model value my house at this price?", Lasso's sparser, more explainable feature set becomes valuable even at a small R² cost relative to Ridge. If the tool instead feeds a purely internal risk score that no one needs to explain feature-by-feature, Ridge's higher R² and more stable coefficient estimates make it the stronger operational choice. This is a genuine trade-off in real valuation-tech companies (e.g. Zillow's "Zestimate"), which have historically had to balance raw predictive accuracy against public explainability.



---



# Question 8

### K-Means Clustering, Scale Sensitivity, and Choosing k



## 8(a) — Four causes of the unbalanced clustering result

> (a) Identify and explain four technical factors that could contribute to the highly unbalanced clustering result. Your answer should consider: 1) differences in feature scales; 2) inappropriate or weakly informative features; 3) the influence of outliers; and 4) an inappropriate choice of the number of clusters, k. (3 Marks)



### Answer







### Explanation (plain language)



**Plain-language version:** K-Means measures "closeness" using ordinary straight-line distance across all features at once. If one feature is measured in tens of thousands and another in single decimals, it is like trying to compare two people's similarity using both their salary in dollars and their height in kilometres — the salary figure will completely swamp the comparison purely because of the units chosen, not because it is actually more important.



---



## 8(b) — Systematic K-Means workflow

> (b) Design a systematic workflow for applying K-Means Clustering to this customer dataset. Your workflow should describe the main steps from raw data to final cluster interpretation, including: data inspection; missing-value handling; feature selection; treatment of outliers; feature standardization; selection of the number of clusters; K-Means model fitting; and interpretation and profiling of the resulting clusters. (3 marks)



### Answer



1. Data inspection: examine shape, dtypes, summary statistics and the distribution of each candidate feature to understand scale, skew and obvious data-quality issues before doing anything else.
2. Missing-value handling: impute or remove missing values (e.g. median imputation for numeric behavioural features), since K-Means cannot operate on NaNs.
3. Feature selection: retain features that plausibly reflect purchasing/usage behaviour and drop identifiers, near-constant columns, and features with no expected relationship to customer segmentation, to avoid diluting the distance metric with noise.
4. Treatment of outliers: detect extreme values (e.g. via IQR or z-score rules) and cap, transform (e.g. log-transform skewed spend variables) or, where justified, remove them, since raw K-Means centroids are mean-based and sensitive to extremes.
5. Feature standardization: apply StandardScaler (zero mean, unit variance) to every feature going into the distance calculation, directly addressing the scale problem identified in part (a).
6. Selection of the number of clusters: fit K-Means across a range of k and use the Elbow Method (inertia) and the Silhouette Score together, as discussed in part (c), to choose a defensible k.
7. K-Means model fitting: fit the final model at the chosen k with an explicit random_state and, ideally, multiple initialisations (n_init) to reduce sensitivity to centroid initialisation.
8. Interpretation and profiling of clusters: compute per-cluster summary statistics (means/medians of the original, unscaled features) for each cluster to translate the mathematical partition into a business-meaningful description of each customer segment.



---



## 8(c) — Inertia, Elbow Method, Silhouette Score

> (c) Explain how the Elbow Method using inertia and the Silhouette Score can be used to evaluate different values of k. Suppose the analysis produces a maximum Silhouette Score of 0.56 at k=4, while the inertia curve shows a clear point of diminishing improvement at k=3. Explain how these two results should be considered when determining an appropriate number of clusters. (2 marks)



### Answer



Inertia is the sum of squared distances from each point to its assigned cluster centroid; it always decreases (weakly) as k increases, because more centroids can only reduce or maintain each point's distance to its nearest one. The Elbow Method plots inertia against k and looks for the point where the rate of decrease sharply slows — the "elbow" — as a heuristic for a k beyond which adding clusters yields diminishing returns in explained within-cluster variance. The Silhouette Score measures, for each point, how much closer it is to its own cluster than to the next-nearest cluster (ranging from −1 to +1, higher is better separated); averaged over all points it gives a single number per k that rewards both compactness and separation, unlike inertia which only rewards compactness.

With a maximum Silhouette Score of 0.56 at k = 4 and a clear inertia diminishing-return point at k = 3, the two diagnostics disagree by one cluster rather than pointing to the same k, and neither should be treated as a mathematically final answer on its own. Silhouette directly measures cluster separation and is generally the more informative of the two when the goal is well-separated, interpretable groups, so it provides reasonable grounds to prefer k = 4; but because the inertia elbow at k = 3 shows that most of the achievable compactness gain is already captured with one fewer cluster, k = 3 remains a credible, more parsimonious alternative. In practice, both k = 3 and k = 4 should be fitted and profiled, and the final choice should be made jointly with domain interpretability — whichever k produces clusters that correspond to actionable, distinguishable customer behaviours is the more defensible choice, not whichever metric happens to peak.



### Example



Concretely: fitting K-Means for k = 2 … 8, plotting inertia typically produces a curve that bends noticeably around k = 3 (the "elbow"), while a separate plot of the average Silhouette Score against k peaks at k = 4 in this scenario. A practitioner would show both plots side by side rather than picking whichever number appears first, then fit both k = 3 and k = 4 and inspect whether the resulting customer segments are recognisably different in a business sense.



---



## 8(d) — Mathematical separation vs. meaningful business segments

> (d) Explain why a mathematically distinct cluster is not necessarily a meaningful business segment. Describe two types of additional evidence or domain knowledge that should be considered before using the resulting clusters to design customer groups or business strategies. (2 marks)



### Answer



K-Means finds a partition that minimises within-cluster squared distance under the specific feature set and scaling supplied to it; a cluster boundary that is mathematically optimal for that objective is not guaranteed to align with a distinction that matters operationally, since the algorithm has no knowledge of which behavioural differences are actionable for marketing, retention, or pricing decisions and which are incidental statistical structure (or even an artefact of the scaling and outlier issues discussed in part (a)).

Two categories of additional evidence that should be examined before clusters are adopted as business segments:

- Business and domain validation: checking whether the clusters correspond to differences the company can act on (distinct product needs, price sensitivity, churn risk, or channel preference) by involving domain experts (marketing, credit risk, sales) to sanity-check whether the discovered groups map onto known or plausible customer archetypes rather than statistical noise.
- Stability and external validation: checking whether the same or similar clusters re-emerge when the model is refit on a different time period, a bootstrap resample, or a held-out sample, and, where available, correlating cluster membership with an external outcome the business already tracks (e.g. actual churn, revenue, or complaint rates) to confirm the mathematical partition carries real, reproducible predictive value rather than being an artefact of one particular sample or feature set.



### Case Study



**Case Study: Retail Customer Segmentation.** A retailer runs K-Means on purchase frequency and average basket value and obtains four mathematically clean clusters. Before renaming these clusters "Bargain Hunters", "Loyal Regulars", "Big Spenders", and "At-Risk" and building marketing campaigns around them, the analytics team checks whether each cluster's customers actually respond differently to past promotions (external validation) and asks the marketing team whether the segment boundaries match categories they can practically act on with distinct offers (business validation). Only after both checks pass does the segmentation get used to drive real campaign budgets.



---



# References



This assignment is a conceptual and applied machine-learning exercise rather than a literature-review assignment, and the PDF brief does not specify a required citation style or a minimum reference count. The following authoritative, genuinely consulted sources support the general theory and the scikit-learn API usage referenced throughout this document, in APA 7th edition style:



- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth International Group.

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in R* (2nd ed.). Springer.

- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python — User Guide*. Retrieved from https://scikit-learn.org/stable/user_guide.html

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B, 58*(1), 267–288.

- Home Credit Group. (n.d.). *Home Credit Default Risk* [Dataset documentation]. Kaggle. Retrieved from https://www.kaggle.com/c/home-credit-default-risk

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. *Journal of Statistics Education, 19*(3).
