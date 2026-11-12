"""
Step 3: Prepare the data for machine learning algorithms (Ch2) and compare
different ways of handling the missing values.

Every experiment we tried is kept in this script, so that all results in the
report can be reproduced by running it:

    python src/03_preprocessing.py

Ch2 lists three options for missing values:
    1. Get rid of the rows with missing values           -> dropna()
    2. Get rid of the whole attribute                    -> drop()
    3. Set the missing values to some value (imputation) -> fillna() / SimpleImputer

Compared set-ups:
    A. Option 3: median (numerical) + most frequent (categorical)   [baseline]
    B. Option 2 + 3: drop 'rbc' (about 40% missing), impute the rest
    C. Option 3 with KNNImputer for the numerical attributes
    D. Option 1: drop every row that has a missing value

Each set-up is evaluated with two classifiers from the lecture notes
(k-NN and SGDClassifier) using stratified 10-fold cross-validation (Ch3)
on the TRAINING SET only. The test set is not touched here.

Outputs (used in the report):
    results/missing_value_strategies.csv   table with mean and std
    results/missing_value_strategies.tex   same table in LaTeX format
    images/missing_value_strategies.png    bar chart of recall and F1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from ckd_data import (load_ckd_data, split_train_test, split_features_labels,
                      build_preprocessing, save_fig, PROJECT_ROOT, TARGET)

RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)

ckd = load_ckd_data()
strat_train_set, strat_test_set = split_train_test(ckd)
X_train, y_train = split_features_labels(strat_train_set)


print("=== Part 1: the three options for missing values (Ch2) ===")
print(f"Training set: {strat_train_set.shape[0]} rows, "
      f"{strat_train_set.shape[1] - 1} attributes, "
      f"{int(strat_train_set.isnull().sum().sum())} missing values")

option1 = strat_train_set.dropna()
print(f"\nOption 1 - dropna(): {len(option1)} rows left "
      f"({len(strat_train_set) - len(option1)} rows deleted)")
print("Class counts before:", strat_train_set[TARGET].value_counts().to_dict())
print("Class counts after: ", option1[TARGET].value_counts().to_dict())
print("-> CKD patients have more missing test results, so deleting rows "
      "removes mostly CKD patients and biases the data.")

option2 = strat_train_set.drop("rbc", axis=1)
print(f"\nOption 2 - drop('rbc'): {option2.shape[1] - 1} attributes left, "
      f"{int(option2.isnull().sum().sum())} missing values left")

median = strat_train_set["rbcc"].median()
option3 = strat_train_set["rbcc"].fillna(median)
print(f"\nOption 3 - fillna(median): rbcc missing values "
      f"{strat_train_set['rbcc'].isnull().sum()} -> {option3.isnull().sum()} "
      f"(median = {median})")


print("\n=== Part 2: preprocessing pipeline (set-up A) ===")
preprocessing = build_preprocessing()
X_train_prepared = preprocessing.fit_transform(X_train)
df_prepared = pd.DataFrame(X_train_prepared,
                           columns=preprocessing.get_feature_names_out(),
                           index=X_train.index)
print("Shape before preprocessing:", X_train.shape)
print("Shape after preprocessing: ", df_prepared.shape)
print("Missing values after preprocessing:",
      int(df_prepared.isnull().sum().sum()))
print(df_prepared.head())

print("\n=== Part 3: comparison of the set-ups (stratified 10-fold CV) ===")
setups = {
    "A": ("Median + most frequent", dict(numeric_imputer="median")),
    "B": ("Drop rbc + median", dict(numeric_imputer="median",
                                     drop_cols=["rbc"])),
    "C": ("KNNImputer", dict(numeric_imputer="knn")),
}
classifiers = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "SGD": SGDClassifier(random_state=42),
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = ["accuracy", "precision", "recall", "f1"]

rows = []


def evaluate(setup_id, description, clf_name, model, X, y):
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    row = {"setup": setup_id, "description": description,
           "model": clf_name, "n_rows": len(X)}
    for metric in scoring:
        row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_std"] = scores[f"test_{metric}"].std()
    rows.append(row)
    print(f"{setup_id} | {description:<24} | {clf_name:<4} | "
          f"recall = {row['recall_mean']:.3f} ± {row['recall_std']:.3f} | "
          f"F1 = {row['f1_mean']:.3f} ± {row['f1_std']:.3f}")


for setup_id, (description, kwargs) in setups.items():
    for clf_name, clf in classifiers.items():
        model = make_pipeline(build_preprocessing(**kwargs), clf)
        evaluate(setup_id, description, clf_name, model, X_train, y_train)


X_complete, y_complete = split_features_labels(option1)
for clf_name, clf in classifiers.items():
    model = make_pipeline(build_preprocessing(), clf)
    evaluate("D", "Drop rows (dropna)", clf_name, model,
             X_complete, y_complete)

results = pd.DataFrame(rows)


results.to_csv(RESULTS_PATH / "missing_value_strategies.csv",
               index=False, float_format="%.3f")

table = results[["setup", "description", "model", "n_rows"]].copy()
for metric in scoring:
    table[metric.capitalize()] = [
        f"{m:.3f} $\\pm$ {s:.3f}"
        for m, s in zip(results[f"{metric}_mean"], results[f"{metric}_std"])]
table.columns = ["Set-up", "Description", "Model", "Rows",
                 "Accuracy", "Precision", "Recall", "F1"]
table.to_latex(RESULTS_PATH / "missing_value_strategies.tex", index=False,
               caption="Comparison of missing-value strategies "
                       "(stratified 10-fold CV on the training set, "
                       "mean $\\pm$ std).",
               label="tab:missing_values")
print(f"\nResults saved: {RESULTS_PATH / 'missing_value_strategies.csv'}")
print(f"Results saved: {RESULTS_PATH / 'missing_value_strategies.tex'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
setup_ids = list(results["setup"].unique())
x = np.arange(len(setup_ids))
width = 0.38
for ax, metric in zip(axes, ["recall", "f1"]):
    for i, clf_name in enumerate(classifiers):
        res = results[results["model"] == clf_name]
        ax.bar(x + (i - 0.5) * width, res[f"{metric}_mean"], width,
               yerr=res[f"{metric}_std"], capsize=4, label=clf_name)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s}\n{results[results['setup'] == s]['description'].iloc[0]}"
         for s in setup_ids], fontsize=8)
    ax.set_title(metric.capitalize() if metric == "recall" else "F1 score")
    ax.set_ylim(0.6, 1.05)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Score (mean ± std, 10 folds)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2)
plt.suptitle("Missing-value strategies (training set)")
save_fig("missing_value_strategies")
