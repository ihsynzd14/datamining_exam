import pandas as pd
import numpy as np


def basic_clean(df):
    """
    Full cleaning pipeline for the Board Games dataset.
    Matches the logic in 01_data_understanding.ipynb exactly.

    Steps:
        1. Drop text/useless/high-missing columns
        2. Fill missing numeric values with median
        3. Fix semantic errors (YearPublished, MaxPlayers)
        4. Drop highly correlated columns
        5. Drop identifier columns (BGGId)
        6. Add log-transformed columns for skewed features
    """
    clean_df = df.copy()

    # 1. Drop text, high-missing, and useless columns
    cols_to_drop = [
        "Family",        # 69.6 % missing
        "ImagePath",     # text / URL
        "Description",   # free text
        "Name",          # identifier text
        "GoodPlayers",   # string list, not numeric
        "NumComments",   # all zeros — useless
        "BGGId",         # identifier — no analytic value
    ]
    clean_df = clean_df.drop(
        columns=[c for c in cols_to_drop if c in clean_df.columns]
    )

    # 2. Fill missing values with median
    for col in ("ComAgeRec", "LanguageEase"):
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # 3. Fix semantic errors
    if "YearPublished" in clean_df.columns:
        clean_df["YearPublished"] = clean_df["YearPublished"].clip(lower=1800)
    if "MaxPlayers" in clean_df.columns:
        clean_df["MaxPlayers"] = clean_df["MaxPlayers"].clip(upper=100)

    # 4. Drop highly correlated columns (|r| > 0.85 redundancy)
    highly_correlated = [
        "ComMinPlaytime",   # ≈ MfgPlaytime
        "ComMaxPlaytime",   # ≈ MfgPlaytime
        "ComWeight",        # ≈ GameWeight
        "NumUserRatings",   # ≈ NumOwned
        "NumWant",          # ≈ NumWish
    ]
    clean_df = clean_df.drop(
        columns=[c for c in highly_correlated if c in clean_df.columns]
    )

    # 5. Add log-transformed columns for heavily skewed features
    for col in ("NumOwned", "NumWish", "MfgPlaytime"):
        if col in clean_df.columns:
            log_col = f"log_{col}"
            if log_col not in clean_df.columns:
                clean_df[log_col] = np.log1p(clean_df[col])

    return clean_df


def load_clean_dataset(path: str = None) -> pd.DataFrame:
    """Load the pre-cleaned CSV produced by notebook 01."""
    import os

    if path is None:
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "dataset", "processed", "DM1_game_dataset_clean.csv",
        )
    return pd.read_csv(path)
