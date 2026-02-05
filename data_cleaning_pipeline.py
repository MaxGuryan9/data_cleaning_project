# %%
"""
Data cleaning pipeline utilities.

Functions operate on copies of DataFrames and return modified copies.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# %%
# df = name of data set
# cat_cols = []
def cast_to_category(df, cat_cols):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

# %%
# df = name of data set
# true_map = {column_name: true_value}

def make_binary(df, true_map):
    df = df.copy()

    for col, true_value in true_map.items():
        if col in df.columns:
            df[col] = (df[col] == true_value).astype(int)

    return df

# %%
# df = name of data set
# one_hot_cols = []

def one_hot_encode(df, one_hot_cols, drop_first=False):
    """One-hot encode specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
    one_hot_cols : list
        Columns to one-hot encode (only those present are encoded).
    drop_first : bool
        If True, drop the first level to avoid collinearity.
    """

    df = df.copy()
    cols_to_encode = [c for c in one_hot_cols if c in df.columns]

    if cols_to_encode:
        df = pd.get_dummies(
            df, columns=cols_to_encode, dtype=int, drop_first=drop_first
        )

    return df

# %%
# df = name of data set
# col = name of numeric column to threshold
# quantile = float (e.g., 0.5 for median, 0.75 for Q3)
# new_col = Name of output binary column
def quantile_to_binary(df, col, quantile, new_col=None):
    df = df.copy()

    if col not in df.columns:
        return df

    threshold = df[col].quantile(quantile)

    if new_col is None:
        q_str = str(quantile).replace(".", "")
        new_col = f"{col}_above_q{q_str}"

    df[new_col] = (df[col] > threshold).astype(int)

    return df


# %%
# df = name of data set
# scale_cols = list of numeric columns to scale
# scaler = sklearn-like scaler object (e.g., MinMaxScaler, StandardScaler)
def scale_columns(df, scale_cols, scaler):
    df = df.copy()

    for col in scale_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[[col]] = scaler.fit_transform(df[[col]])

    return df

# %%
# df = name of data set
# cols_to_drop = list of columns to drop

def drop_columns(df, cols_to_drop):
    df = df.copy()

    cols_present = [c for c in cols_to_drop if c in df.columns]

    if cols_present:
        df = df.drop(columns=cols_present)

    return df

# %%
# df = name of data set
# binary_col = name of binary column to calculate prevalence for
def calculate_prevalence(df, binary_col):
    if binary_col not in df.columns:
        raise ValueError(f"Column '{binary_col}' not found in DataFrame")

    total_rows = len(df)

    if total_rows == 0:
        return 0.0

    prevalence = df[binary_col].sum() / total_rows

    return prevalence


# %%
# df = name of data set
# target_col = name of binary target column for stratification
# train_size = number of rows (int) or fraction (float) for training set
# tune_frac = fraction of remaining data to allocate to tuning set
def split_train_tune_test(df, target_col, train_size, tune_frac):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # First split: Train vs remainder
    Train, remainder = train_test_split(
        df, train_size=train_size, stratify=df[target_col]
    )

    # Second split: Tune vs Test
    Tune, Test = train_test_split(
        remainder, train_size=tune_frac, stratify=remainder[target_col]
    )

    return Train, Tune, Test
# %%
# Train, Test, Tune
# target_col = name of binary target column for stratification

def print_prevalence_splits(Train, Tune, Test, target_col):
    def prevalence(df):
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found")

        return df[target_col].value_counts(normalize=True).get(1, 0)

    train_prev = prevalence(Train)
    tune_prev = prevalence(Tune)
    test_prev = prevalence(Test)

    print(
        f"""Training Prevalence: {train_prev:.2f}
Tuning Prevalence: {tune_prev:.2f}
Test Prevalence: {test_prev:.2f}
"""
    )

