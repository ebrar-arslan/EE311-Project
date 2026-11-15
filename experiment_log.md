# Experiment Log

What we tried, why we tried it, and what we found. This log is the source
for the "Methods" and "Results" sections of the midterm and final reports.
All results below come from the **training set only** (320 rows); the test
set (80 rows) has not been used yet.

---

## Step 1 – Loading and first look at the data
**Script:** `src/01_load_and_explore.py`

**What:** Loaded the ARFF file and inspected it with `head()`, `info()`,
`value_counts()`, `describe()` and histograms (Ch2, "Take a Look at the Data Structure").

**Why:** To learn the type, range and number of missing values of each attribute
before deciding on any preprocessing.

**Notes / findings**
- `scipy.io.arff` could not read the file: some rows contain stray TAB characters
  and three rows have an extra comma (two at the end of the line, one in the middle).
  We wrote our own parser (`load_ckd_data` in `src/ckd_data.py`) that strips
  whitespace, drops the empty fields and maps `?` to missing (NaN).
  All 400 rows are read correctly (250 ckd, 150 notckd).
- 11 numerical, 13 categorical attributes + class.
- Many missing values: rbc 152, rbcc 131, wbcc 106, pot 88, sod 87 (of 400).
- Numerical attributes have very different scales and several are right-skewed
  (e.g. bgr, bu, sc) → feature scaling is needed (Ch2).

## Step 2 – Test set and exploration
**Script:** `src/02_split_and_visualize.py`

**What:** Stratified 80/20 split on the class label
(`train_test_split(..., stratify=y, random_state=42)`), then exploration of the
training set only: missing-value table, class balance, box plots by class,
Pearson correlation matrix and scatter matrix.

**Why:**
- The test set is set aside *before* any cleaning or deeper exploration to avoid
  data snooping bias (Ch2).
- Stratified sampling because the classes are imbalanced (62.5% / 37.5%).

**Findings**
- Stratified split keeps the exact class ratio (62.5% ckd in the full data and in
  the test set); a purely random split gives 65.0% ckd in the test set.
- Only 123 of the 320 training rows have no missing value at all.
- Highest missing rates in the training set: rbc 39.7%, rbcc 33.8%, wbcc 27.8%,
  sod and pot 22.5%.
- Strongest correlations with the target (ckd = 1): hemo −0.77, pcv −0.75,
  rbcc −0.68, sod −0.47, bgr +0.41, bu +0.36, sc +0.34.
  (Pearson's r only measures linear relationships.)

## Step 3 – Missing values and preprocessing pipeline
**Script:** `src/03_preprocessing.py` (pipeline defined in `build_preprocessing`
in `src/ckd_data.py`)

**Preprocessing pipeline (Ch2, `ColumnTransformer`)**
| Attribute group | Columns | Steps |
|---|---|---|
| Numerical | age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc | impute → `StandardScaler` |
| Ordinal categorical | sg, al, su | `SimpleImputer(most_frequent)` → `StandardScaler` |
| Nominal categorical | rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

- sg, al and su are kept as numbers because their levels have a natural order.
- Scaling is needed because k-NN (distance based) and SGD (gradient based) are
  sensitive to feature scales.
- Every step is inside a pipeline, so it is fitted on the training folds only
  and applied unchanged to validation/test data (no leakage).
- Output: 24 attributes → 34 features, no missing values.

**What we compared:** the three options for missing data in Ch2.
| Set-up | Description | Why |
|---|---|---|
| A | median (numerical) + most frequent (categorical) | least destructive option, baseline (Ch2) |
| B | drop `rbc`, then as A | rbc has the most missing values (~40%) |
| C | `KNNImputer` (k = 5, after scaling) for numerical + most frequent | uses the other attributes, may give more realistic values |
| D | drop every row with a missing value (`dropna`) | reference: how much does deleting data hurt? |

**Evaluation:** k-NN (k = 5) and `SGDClassifier` (default, linear SVM), stratified
10-fold cross-validation on the training set; accuracy, precision, recall, F1
(positive class = ckd). Recall is the most important metric here because a
missed CKD patient (false negative) is worse than a false alarm (Ch3).

**Outputs for the report** (all produced by running `python src/03_preprocessing.py`):
- `results/missing_value_strategies.csv` – full table (mean and std of each metric)
- `results/missing_value_strategies.tex` – the same table in LaTeX (`\input{...}`)
- `images/missing_value_strategies.png` – bar chart of recall and F1 with error bars
- The console output also shows the three Ch2 options with pandas
  (`dropna()`, `drop()`, `fillna()`) and their effect on the training set.

**Results (mean ± std over 10 folds)**
| Set-up | Model | Rows | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A | k-NN | 320 | 0.972 ± 0.022 | 1.000 ± 0.000 | 0.955 ± 0.035 | 0.977 ± 0.018 |
| A | SGD | 320 | 0.994 ± 0.013 | 1.000 ± 0.000 | 0.990 ± 0.020 | 0.995 ± 0.010 |
| B | k-NN | 320 | 0.975 ± 0.023 | 1.000 ± 0.000 | 0.960 ± 0.037 | 0.979 ± 0.020 |
| B | SGD | 320 | 0.997 ± 0.009 | 1.000 ± 0.000 | 0.995 ± 0.015 | 0.997 ± 0.008 |
| C | k-NN | 320 | 0.959 ± 0.031 | 1.000 ± 0.000 | 0.935 ± 0.050 | 0.966 ± 0.027 |
| C | SGD | 320 | 0.991 ± 0.014 | 1.000 ± 0.000 | 0.985 ± 0.023 | 0.992 ± 0.012 |
| D | k-NN | 123 | 0.967 ± 0.040 | 1.000 ± 0.000 | 0.850 ± 0.189 | 0.907 ± 0.120 |
| D | SGD | 123 | 0.984 ± 0.032 | 1.000 ± 0.000 | 0.917 ± 0.171 | 0.947 ± 0.111 |

**Interpretation**
- Imputation (A, B, C) clearly beats deleting rows (D). Dropping rows keeps only
  123 rows and, more importantly, **biases the data**: 94 of them are notckd and
  only 29 are ckd, because CKD patients have more missing test results. Recall
  drops and becomes very unstable (std ≈ 0.17–0.19). Option D also could not
  handle new patients with missing values, so it is rejected.
- Median imputation (A) and dropping rbc (B) give almost the same scores; the
  differences are smaller than one standard deviation, so they are not
  significant.
- KNNImputer (C) did not improve the results over the simpler median imputation.
- **Decision:** use set-up A (median + most frequent, all attributes) as the
  default preprocessing, because it is the simplest and loses no information.
  B and C can be revisited during model fine-tuning (e.g. with `GridSearchCV`).
- Scores are already very high with simple models, which agrees with the
  literature on this dataset; the dataset is small, so final conclusions will
  be based on cross-validation mean ± std and the held-out test set.

## Step 4 – Shortlist promising models
**Script:** `src/04_compare_models.py`

**What:** Compared the classifiers covered in the lecture notes so far, all with
**default hyperparameters** and the preprocessing pipeline of step 3 (set-up A):
| Model | Lecture notes | Why |
|---|---|---|
| `DummyClassifier(strategy="most_frequent")` | Evaluating Hypotheses | baseline; shows why accuracy alone is misleading on imbalanced data |
| `SGDClassifier` | Ch3 | linear classifier (linear SVM by default) trained with SGD |
| `KNeighborsClassifier` (k = 5) | k-NN | instance-based, distance-based model |
| `LogisticRegression` | Ch4 | linear model that outputs class probabilities |
| `RandomForestClassifier` | Ch3 (ROC / PR example) | non-linear model; also gives feature importances |

Naive Bayes, SVM (kernel), decision tree and XGBoost from the project plan are
postponed until they are covered in the lectures (remaining work).

**Evaluation:** stratified 10-fold CV on the training set (320 rows).
Metrics: accuracy, precision, recall, F1, UAR (unweighted average recall =
balanced accuracy), ROC AUC; confusion matrices (raw and normalized by the true
class) from `cross_val_predict`; ROC and precision–recall curves; learning curves
(Ch4) to check for overfitting; feature importance.

**Outputs:** `results/model_comparison.csv/.tex`, `images/model_comparison.png`,
`images/confusion_matrices.png`, `images/confusion_matrices_normalized.png`,
`images/roc_pr_curves.png`, `images/learning_curves.png`,
`images/feature_importance.png`.

**Results (mean over 10 folds)**
| Model | Accuracy | Recall | F1 | UAR | ROC AUC | Missed CKD (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|
| Dummy | 0.625 | 1.000 | 0.769 | 0.500 | 0.500 | 0 | 120 |
| SGD | 0.994 | 0.990 | 0.995 | 0.995 | 1.000 | 2 | 0 |
| k-NN | 0.972 | 0.955 | 0.977 | 0.978 | 1.000 | 9 | 0 |
| Logistic Regression | 0.994 | 0.995 | 0.995 | 0.993 | 1.000 | 1 | 1 |
| Random Forest | 0.991 | 0.995 | 0.993 | 0.989 | 0.999 | 1 | 2 |
(FN / FP counted from `cross_val_predict` over the 320 training rows.)

**Interpretation**
- The dummy model reaches 62.5% accuracy and perfect recall by predicting "ckd"
  for everyone, but its UAR is 0.5 and it gives 120 false alarms. This confirms
  the lecture point that accuracy (and recall alone) can be misleading on
  imbalanced data; UAR and the confusion matrix reveal the problem.
- All real models are far above the baseline (accuracy 0.97–0.99, ROC AUC ≈ 1).
  The differences between SGD, logistic regression and random forest are within
  one standard deviation, so they are not significant on this small dataset.
- k-NN has the lowest recall (0.955): it misses 9 CKD patients, the most costly
  error here. Its default k = 5 has not been tuned yet.
- Learning curves: training and validation F1 are close for all models
  (e.g. logistic regression 1.000 vs 0.995), so there is no strong overfitting.
- Most important attributes (random forest and logistic regression agree):
  hemoglobin (hemo), packed cell volume (pcv), specific gravity (sg), serum
  creatinine (sc), red blood cell count (rbcc) and albumin (al). Low hemo, pcv and
  sg push towards CKD, high al and sc push towards CKD. This matches the
  correlation analysis in step 2 and medical knowledge (anemia and kidney
  function markers).
- **Shortlist:** logistic regression, SGD and random forest (highest recall and
  UAR, fewest missed CKD cases). k-NN is kept for the fine-tuning step only if
  tuning k improves its recall.

**Remaining work:** fine-tune the shortlisted models (`GridSearchCV`, e.g. C for
logistic regression, alpha for SGD, n_estimators / max_depth for random forest,
k for k-NN), add the models from the plan as they are covered in the lectures,
then evaluate the final model once on the test set.
