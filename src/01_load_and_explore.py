"""
Step 1: Load and explore the data (Ch2 - "Take a Look at the Data Structure").

Methods used (as in the lecture notes):
    head(), info(), value_counts(), describe(), hist()
"""
from ckd_data import load_ckd_data, save_fig, NUMERIC_COLS, TARGET

ckd = load_ckd_data()

print("\n=== head() ===")
print(ckd.head())

print("\n=== info() ===")
ckd.info()

print("\n=== Missing values per attribute ===")
print(ckd.isnull().sum().sort_values(ascending=False))

print("\n=== value_counts(): target attribute (class) ===")
print(ckd[TARGET].value_counts())

print("\n=== value_counts(): categorical attributes ===")
for col in ckd.columns:
    if col not in NUMERIC_COLS and col != TARGET:
        print(f"\n{col}:\n{ckd[col].value_counts(dropna=False)}")

print("\n=== describe() ===")
print(ckd.describe())

ckd.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")
