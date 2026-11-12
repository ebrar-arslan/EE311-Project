"""
Shared helper functions used by all scripts in this project.

As suggested in Ch2 ("Prepare the Data..."), we write reusable functions
instead of repeating the same code in every script.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files instead of opening a window
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Paths are relative to the project root, so the scripts work from any folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "chronic_kidney_disease_full.arff"
IMAGES_PATH = PROJECT_ROOT / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
                "hemo", "pcv", "wbcc", "rbcc"]
TARGET = "class"

ORDINAL_COLS = ["sg", "al", "su"]
NOMINAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
                "appet", "pe", "ane"]
CATEGORICAL_COLS = ORDINAL_COLS + NOMINAL_COLS


def load_ckd_data(path=DATA_PATH):

    columns, rows = [], []
    in_data = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                columns.append(line.split()[1].strip("'"))
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data:
                values = [v.strip() for v in line.split(",")]
                values = [v for v in values if v != ""]
                if len(values) == len(columns):
                    rows.append(values)
                else:
                    print("Skipped line:", line)

    df = pd.DataFrame(rows, columns=columns).replace("?", np.nan)

    df[NUMERIC_COLS + ORDINAL_COLS] = df[NUMERIC_COLS + ORDINAL_COLS].apply(
        pd.to_numeric)
    return df


def split_features_labels(df):

    X = df.drop(TARGET, axis=1)
    y = (df[TARGET] == "ckd").astype(int)
    return X, y


def split_train_test(df, test_size=0.2, random_state=42):
  
    strat_train_set, strat_test_set = train_test_split(
        df, test_size=test_size, stratify=df[TARGET],
        random_state=random_state)
    return strat_train_set, strat_test_set


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()
    print(f"Figure saved: {path}")


def build_preprocessing(numeric_imputer="median", drop_cols=()):
    
    num_cols = [c for c in NUMERIC_COLS if c not in drop_cols]
    ord_cols = [c for c in ORDINAL_COLS if c not in drop_cols]
    nom_cols = [c for c in NOMINAL_COLS if c not in drop_cols]

    if numeric_imputer == "median":
        num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
    elif numeric_imputer == "knn":
        # KNN uses distances, so we scale before imputing
        num_pipeline = make_pipeline(StandardScaler(),
                                     KNNImputer(n_neighbors=5))
    else:
        raise ValueError("numeric_imputer must be 'median' or 'knn'")

    ord_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                 StandardScaler())
    nom_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                 OneHotEncoder(handle_unknown="ignore"))

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("ord", ord_pipeline, ord_cols),
        ("nom", nom_pipeline, nom_cols),
    ])
