"""
Step 2: Create a test set and explore the training set
(Ch2 - "Create a Test Set" and "Explore and Visualize the Data to Gain Insights").

Important (Ch2): put the test set aside BEFORE looking deeper into the data
or filling in missing values, and explore only the training set.
Otherwise we may "snoop" on the test data (data snooping bias).
"""
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from ckd_data import (load_ckd_data, split_train_test, save_fig,
                      NUMERIC_COLS, TARGET)

ckd = load_ckd_data()


strat_train_set, strat_test_set = split_train_test(ckd)
print(f"Training set: {len(strat_train_set)} rows, "
      f"test set: {len(strat_test_set)} rows")

from sklearn.model_selection import train_test_split
_, random_test_set = train_test_split(ckd, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall %": ckd[TARGET].value_counts(normalize=True),
    "Stratified %": strat_test_set[TARGET].value_counts(normalize=True),
    "Random %": random_test_set[TARGET].value_counts(normalize=True),
}) * 100
print("\n=== Class proportions ===")
print(compare_props.round(2))

ckd_train = strat_train_set.copy()


missing = pd.DataFrame({
    "missing": ckd_train.isnull().sum(),
    "missing %": (ckd_train.isnull().mean() * 100).round(1),
}).sort_values("missing", ascending=False)
print("\n=== Missing values in the training set ===")
print(missing)

rows_complete = ckd_train.dropna().shape[0]
print(f"\nRows with no missing value at all: {rows_complete} "
      f"of {len(ckd_train)}")


ckd_train[TARGET].value_counts().plot(kind="bar", figsize=(5, 4))
plt.ylabel("Number of patients")
plt.title("Class balance (training set)")
save_fig("class_balance")

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for ax, col in zip(axes.flat, NUMERIC_COLS):
    ckd_train.boxplot(column=col, by=TARGET, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
axes.flat[-1].set_visible(False)
plt.suptitle("Numerical attributes by class (training set)")
save_fig("boxplots_by_class")


ckd_num = ckd_train[NUMERIC_COLS].copy()
ckd_num["ckd"] = (ckd_train[TARGET] == "ckd").astype(int)

corr_matrix = ckd_num.corr()
print("\n=== Correlation with the target (ckd = 1) ===")
print(corr_matrix["ckd"].sort_values(ascending=False))

plt.figure(figsize=(9, 7))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson's r")
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title("Correlation matrix (training set)")
save_fig("correlation_matrix")

attributes = ["hemo", "pcv", "rbcc", "sc", "bu"]
scatter_matrix(ckd_train[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
