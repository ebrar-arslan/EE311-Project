"""
Step 4: Shortlist promising models (Ch3, Ch4, k-NN, Evaluating Hypotheses).

We only use models covered in the lecture notes so far:
    - DummyClassifier          baseline: always predicts the most frequent class
    - SGDClassifier            linear classifier trained with SGD (Ch3)
    - KNeighborsClassifier     k-nearest neighbours (k-NN lecture)
    - LogisticRegression       logistic regression (Ch4)
    - RandomForestClassifier   used as an example in Ch3 (ROC / PR comparison)

All models use the same preprocessing pipeline as step 3 (set-up A) and the
DEFAULT hyperparameters. Fine-tuning comes later (Ch2 "Fine-Tune Your Model").
Everything is evaluated with stratified 10-fold cross-validation on the
training set only; the test set is still not touched.

Metrics (positive class = ckd):
    accuracy (classification rate), precision, recall, F1,
    UAR (unweighted average recall = balanced accuracy), ROC AUC
Why several metrics: our classes are imbalanced (62.5% / 37.5%), so accuracy
alone can be misleading (Evaluating Hypotheses, "Imbalanced dataset").

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict, learning_curve)
from sklearn.metrics import (ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, roc_auc_score)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from ckd_data import (load_ckd_data, split_train_test, split_features_labels,
                      build_preprocessing, save_fig, PROJECT_ROOT)

RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)

ckd = load_ckd_data()
strat_train_set, strat_test_set = split_train_test(ckd)
X_train, y_train = split_features_labels(strat_train_set)


classifiers = {
    "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
    "SGD": SGDClassifier(random_state=42),
    "k-NN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}
models = {name: make_pipeline(build_preprocessing(), clf)
          for name, clf in classifiers.items()}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "UAR": "balanced_accuracy",   
    "ROC AUC": "roc_auc",
}


print("=== Part 1: stratified 10-fold CV on the training set ===")
rows = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                            return_train_score=True)
    row = {"Model": name}
    for metric in scoring:
        row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_std"] = scores[f"test_{metric}"].std()
    row["Train accuracy"] = scores["train_Accuracy"].mean()
    rows.append(row)
    print(f"{name:<22} acc = {row['Accuracy_mean']:.3f}  "
          f"recall = {row['Recall_mean']:.3f} ± {row['Recall_std']:.3f}  "
          f"F1 = {row['F1_mean']:.3f}  UAR = {row['UAR_mean']:.3f}  "
          f"AUC = {row['ROC AUC_mean']:.3f}  "
          f"(train acc = {row['Train accuracy']:.3f})")

results = pd.DataFrame(rows)
results.to_csv(RESULTS_PATH / "model_comparison.csv", index=False,
               float_format="%.3f")

table = pd.DataFrame({"Model": results["Model"]})
for metric in scoring:
    table[metric] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in
                     zip(results[f"{metric}_mean"], results[f"{metric}_std"])]
table.to_latex(RESULTS_PATH / "model_comparison.tex", index=False,
               caption="Comparison of classifiers with default hyperparameters "
                       "(stratified 10-fold CV on the training set, "
                       "mean $\\pm$ std, positive class = CKD).",
               label="tab:model_comparison")
print(f"\nResults saved: {RESULTS_PATH / 'model_comparison.csv'}")
print(f"Results saved: {RESULTS_PATH / 'model_comparison.tex'}")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(scoring))
width = 0.8 / len(models)
for i, name in enumerate(models):
    r = results[results["Model"] == name].iloc[0]
    ax.bar(x + (i - (len(models) - 1) / 2) * width,
           [r[f"{m}_mean"] for m in scoring], width,
           yerr=[r[f"{m}_std"] for m in scoring], capsize=3, label=name)
ax.set_xticks(x)
ax.set_xticklabels(list(scoring))
ax.set_ylim(0, 1.2)
ax.set_ylabel("Score (mean ± std, 10 folds)")
ax.set_title("Classifier comparison (training set, default hyperparameters)")
ax.legend(loc="upper center", ncol=len(models), fontsize=8)
ax.grid(axis="y", alpha=0.3)
save_fig("model_comparison")


print("\n=== Part 2: confusion matrices (cross_val_predict) ===")
y_preds = {name: cross_val_predict(model, X_train, y_train, cv=cv)
           for name, model in models.items()}

for normalize, fig_id in [(None, "confusion_matrices"),
                          ("true", "confusion_matrices_normalized")]:
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for ax, (name, y_pred) in zip(axes, y_preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_train, y_pred, display_labels=["notckd", "ckd"],
            normalize=normalize, values_format=".2f" if normalize else "d",
            ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    title = "Confusion matrices"
    if normalize:
        title += " (normalized by true class)"
    plt.suptitle(title + " - training set, 10-fold CV")
    save_fig(fig_id)

for name, y_pred in y_preds.items():
    fn = int(((y_train == 1) & (y_pred == 0)).sum())
    fp = int(((y_train == 0) & (y_pred == 1)).sum())
    print(f"{name:<22} false negatives (missed CKD) = {fn:>3}, "
          f"false positives = {fp:>3}")


print("\n=== Part 3: ROC and PR curves ===")
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
for name, model in models.items():
    if name.startswith("Dummy"):
        continue
    method = "decision_function" if name == "SGD" else "predict_proba"
    y_scores = cross_val_predict(model, X_train, y_train, cv=cv,
                                 method=method)
    if method == "predict_proba":
        y_scores = y_scores[:, 1]     
    fpr, tpr, _ = roc_curve(y_train, y_scores)
    precisions, recalls, _ = precision_recall_curve(y_train, y_scores)
    auc = roc_auc_score(y_train, y_scores)
    ax_roc.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax_pr.plot(recalls, precisions, linewidth=2, label=name)
    print(f"{name:<22} ROC AUC = {auc:.3f}")

ax_roc.plot([0, 1], [0, 1], "k:", label="Random classifier")
ax_roc.set_xlabel("False positive rate (1 - specificity)")
ax_roc.set_ylabel("True positive rate (recall)")
ax_roc.set_title("ROC curves")
ax_roc.legend(loc="lower right", fontsize=8)
ax_roc.grid(alpha=0.3)
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-recall curves")
ax_pr.legend(loc="lower left", fontsize=8)
ax_pr.grid(alpha=0.3)
save_fig("roc_pr_curves")

print("\n=== Part 4: learning curves ===")
names = [n for n in models if not n.startswith("Dummy")]
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4),
                         sharey=True)
for ax, name in zip(axes, names):
    train_sizes, train_scores, valid_scores = learning_curve(
        models[name], X_train, y_train, cv=cv, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8))
    ax.plot(train_sizes, train_scores.mean(axis=1), "r-+", linewidth=2,
            label="train")
    ax.plot(train_sizes, valid_scores.mean(axis=1), "b-", linewidth=2,
            label="valid")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Training set size")
    ax.grid(alpha=0.3)
    print(f"{name:<22} F1 train = {train_scores.mean(axis=1)[-1]:.3f}, "
          f"valid = {valid_scores.mean(axis=1)[-1]:.3f}")
axes[0].set_ylabel("F1 score")
axes[0].legend(loc="lower right")
plt.suptitle("Learning curves (stratified 10-fold CV)")
save_fig("learning_curves")


print("\n=== Part 5: feature importance ===")
rf = models["Random Forest"].fit(X_train, y_train)
feature_names = rf[0].get_feature_names_out()
rf_importance = pd.Series(rf[-1].feature_importances_, index=feature_names)

log_reg = models["Logistic Regression"].fit(X_train, y_train)

lr_coef = pd.Series(log_reg[-1].coef_[0], index=feature_names)

print("Random forest, top 10:")
print(rf_importance.sort_values(ascending=False).head(10).round(3))
print("\nLogistic regression, top 10 by |coefficient|:")
print(lr_coef.reindex(lr_coef.abs().sort_values(ascending=False).index)
      .head(10).round(3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
rf_importance.sort_values().tail(15).plot(kind="barh", ax=ax1)
ax1.set_title("Random forest feature importance (top 15)")
top_lr = lr_coef.reindex(lr_coef.abs().sort_values().index).tail(15)
top_lr.plot(kind="barh", ax=ax2,
            color=["tab:red" if c > 0 else "tab:blue" for c in top_lr])
ax2.set_title("Logistic regression coefficients (top 15)\n"
              "red = towards ckd, blue = towards notckd")
save_fig("feature_importance")
