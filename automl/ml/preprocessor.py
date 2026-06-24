"""
preprocessor.py
---------------
Builds a fully configurable sklearn preprocessing pipeline from user
configuration. No Flask imports — pure ML logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, FunctionTransformer,
)
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColumnPreprocessConfig:
    """Per-column preprocessing settings."""
    col_type: str  # "numerical" | "categorical" | "datetime" | "ignore"

    # Numerical options
    missing_num: str = "median"     # mean | median | mode | drop
    scaling: str = "standard"       # standard | minmax | robust | none
    transform: str = "none"         # log | none
    outlier: str = "none"           # iqr | zscore | none

    # Categorical options
    missing_cat: str = "mode"       # mode | drop
    encoding: str = "onehot"        # label | onehot | frequency


@dataclass
class PreprocessConfig:
    """Global preprocessing configuration collected from the UI."""
    columns: Dict[str, ColumnPreprocessConfig] = field(default_factory=dict)
    remove_duplicates: bool = True
    remove_constant: bool = True
    remove_correlated: bool = False
    correlation_threshold: float = 0.95
    remove_low_variance: bool = False
    variance_threshold: float = 0.01
    extract_datetime: bool = True


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categories by their frequency in the training data."""

    def __init__(self):
        self.freq_maps_: Dict[int, Dict] = {}

    def fit(self, X, y=None):
        self.freq_maps_ = {}
        arr = np.array(X)
        for i in range(arr.shape[1]):
            col = arr[:, i].astype(str)
            vals, counts = np.unique(col, return_counts=True)
            total = len(col)
            self.freq_maps_[i] = {v: c / total for v, c in zip(vals, counts)}
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=object)
        result = np.zeros_like(arr, dtype=float)
        for i in range(arr.shape[1]):
            col = arr[:, i].astype(str)
            result[:, i] = [self.freq_maps_[i].get(v, 0.0) for v in col]
        return result


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Clip outliers using IQR or Z-score (does NOT drop rows — clips instead)."""

    def __init__(self, method: str = "iqr"):
        self.method = method
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=float)
        if self.method == "iqr":
            Q1 = np.nanpercentile(X, 25, axis=0)
            Q3 = np.nanpercentile(X, 75, axis=0)
            IQR = Q3 - Q1
            self.lower_ = Q1 - 1.5 * IQR
            self.upper_ = Q3 + 1.5 * IQR
        else:  # zscore
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            std[std == 0] = 1
            self.lower_ = mean - 3 * std
            self.upper_ = mean + 3 * std
        return self

    def transform(self, X, y=None):
        X = np.array(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


# ---------------------------------------------------------------------------
# Global preprocessing (applied to entire DataFrame before ColumnTransformer)
# ---------------------------------------------------------------------------

def apply_global_preprocessing(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Apply row-level / column-level global operations to the DataFrame."""
    steps_log = []

    if config.remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        steps_log.append(f"Removed {removed} duplicate rows")

    # Drop rows with missing values where explicitly configured as 'drop'
    for col in list(df.columns):
        if col in config.columns:
            col_cfg = config.columns[col]
            if col_cfg.col_type == "numerical" and col_cfg.missing_num == "drop":
                before = len(df)
                df = df.dropna(subset=[col])
                removed = before - len(df)
                if removed > 0:
                    steps_log.append(f"Dropped {removed} rows with missing values in numerical column '{col}'")
            elif col_cfg.col_type == "categorical" and col_cfg.missing_cat == "drop":
                before = len(df)
                df = df.dropna(subset=[col])
                removed = before - len(df)
                if removed > 0:
                    steps_log.append(f"Dropped {removed} rows with missing values in categorical column '{col}'")

    if config.extract_datetime:
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                    df[f"{col}_year"] = parsed.dt.year
                    df[f"{col}_month"] = parsed.dt.month
                    df[f"{col}_day"] = parsed.dt.day
                    df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                    df = df.drop(columns=[col])
                    steps_log.append(f"Extracted datetime features from '{col}'")
                except Exception:
                    pass

    if config.remove_constant:
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            steps_log.append(f"Removed {len(constant_cols)} constant columns: {constant_cols}")

    return df, steps_log


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_column_transformer(df: pd.DataFrame, config: PreprocessConfig) -> ColumnTransformer:
    """
    Build a ColumnTransformer from the PreprocessConfig.
    Each column gets its own tailored pipeline.
    """
    transformers = []

    # Group columns by their settings for efficiency
    num_groups: Dict[str, List[str]] = {}
    cat_groups: Dict[str, List[str]] = {}

    for col, cfg in config.columns.items():
        if col not in df.columns:
            continue
        if cfg.col_type == "ignore":
            continue
        key = f"{cfg.col_type}|{cfg.missing_num}|{cfg.scaling}|{cfg.transform}|{cfg.outlier}|{cfg.missing_cat}|{cfg.encoding}"
        if cfg.col_type == "numerical":
            num_groups.setdefault(key, []).append(col)
        elif cfg.col_type == "categorical":
            cat_groups.setdefault(key, []).append(col)

    idx = 0
    for key, cols in num_groups.items():
        parts = key.split("|")
        _, missing_num, scaling, transform, outlier, _, _ = parts

        steps = []

        # Imputation
        strategy_map = {"mean": "mean", "median": "median", "mode": "most_frequent"}
        if missing_num in strategy_map:
            steps.append(("imputer", SimpleImputer(strategy=strategy_map[missing_num])))

        # Log transform
        if transform == "log":
            steps.append(("log", FunctionTransformer(
                func=lambda X: np.log1p(np.clip(X, 0, None)),
                validate=False
            )))

        # Outlier removal
        if outlier in ("iqr", "zscore"):
            steps.append(("outlier", OutlierRemover(method=outlier)))

        # Scaling
        if scaling == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaling == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaling == "robust":
            steps.append(("scaler", RobustScaler()))

        if steps:
            pipe = Pipeline(steps)
            transformers.append((f"num_{idx}", pipe, cols))
            idx += 1

    for key, cols in cat_groups.items():
        parts = key.split("|")
        _, _, _, _, _, missing_cat, encoding = parts

        steps = []

        # Categorical imputation
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

        # Encoding
        if encoding == "onehot":
            steps.append(("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, max_categories=30
            )))
        elif encoding == "label":
            # LabelEncoder doesn't work in pipelines directly; use OrdinalEncoder
            from sklearn.preprocessing import OrdinalEncoder
            steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        elif encoding == "frequency":
            steps.append(("encoder", FrequencyEncoder()))

        pipe = Pipeline(steps)
        transformers.append((f"cat_{idx}", pipe, cols))
        idx += 1

    if not transformers:
        # Fallback: passthrough all numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        transformers.append(("passthrough", "passthrough", num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_output_feature_names(ct: ColumnTransformer, df_cols) -> List[str]:
    """Get feature names after ColumnTransformer (handles OHE expansion)."""
    names = []
    for name, transformer, cols in ct.transformers_:
        if transformer == "drop" or transformer == "passthrough":
            names.extend(cols)
            continue
        try:
            step_names = transformer.get_feature_names_out()
            names.extend(step_names)
        except Exception:
            if isinstance(cols, list):
                names.extend(cols)
            else:
                names.extend(list(cols))
    return names


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect column types for each column.
    Returns a dict mapping col_name -> "numerical" | "categorical" | "datetime" | "ignore"
    """
    type_map = {}
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64, np.int32, np.float32, "int64", "float64"]:
            type_map[col] = "numerical"
        elif df[col].dtype == "object":
            # Try datetime detection
            sample = df[col].dropna().head(50)
            try:
                pd.to_datetime(sample)
                type_map[col] = "datetime"
            except Exception:
                type_map[col] = "categorical"
        elif str(df[col].dtype).startswith("datetime"):
            type_map[col] = "datetime"
        else:
            type_map[col] = "categorical"
    return type_map
