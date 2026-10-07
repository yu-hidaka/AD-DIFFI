"""
Data loading, validation, preprocessing, and output utilities for AD-DIFFI.

This module does not:
- generate synthetic fallback datasets;
- fit Isolation Forests;
- calculate DIFFI or AD-DIFFI scores;
- perform train/test splitting;
- fit downstream models.

Training-dependent preprocessing is implemented through fit/transform helpers.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths and dataset URLs
# ============================================================


DEFAULT_DATA_DIR = Path(
    "/tmp/ad_diffi_data"
)

DEFAULT_DATASET_URLS = {
    "annthyroid": [
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/data/annthyroid.csv"
        ),
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/datasets/Classical/"
            "21_annthyroid.csv"
        ),
    ],
    "stroke": [
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/data/stroke.csv"
        ),
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/datasets/Classical/"
            "21_stroke.csv"
        ),
    ],
    "hepatitis": [
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/data/hepatitis.csv"
        ),
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/datasets/Classical/"
            "21_hepatitis.csv"
        ),
    ],
    "breast_cancer": [
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/data/breast_cancer.csv"
        ),
        (
            "https://raw.githubusercontent.com/Minqi824/"
            "ADBench/main/datasets/Classical/"
            "21_breast_cancer.csv"
        ),
    ],
}


# ============================================================
# Dataset loading
# ============================================================


def _validate_csv_file(
    csv_path: Path,
) -> None:
    """
    Confirm that a downloaded file can be read as a non-empty CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: {csv_path}"
        )

    if csv_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset file is empty: {csv_path}"
        )

    sample = pd.read_csv(
        csv_path,
        nrows=5,
    )

    if sample.shape[1] == 0:
        raise ValueError(
            f"Dataset has no columns: {csv_path}"
        )


def download_dataset(
    dataset_name: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    urls: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> str:
    """
    Download a real benchmark dataset.

    No synthetic fallback is used. If all download attempts fail, an exception
    is raised.
    """
    dataset_name = dataset_name.lower()
    data_dir = Path(data_dir)

    data_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    csv_path = data_dir / f"{dataset_name}.csv"

    if csv_path.exists() and not overwrite:
        _validate_csv_file(csv_path)
        return str(csv_path)

    if urls is None:
        if dataset_name not in DEFAULT_DATASET_URLS:
            raise ValueError(
                f"No default URLs registered for {dataset_name!r}."
            )

        urls = DEFAULT_DATASET_URLS[
            dataset_name
        ]

    errors = []

    for url in urls:
        try:
            if csv_path.exists():
                csv_path.unlink()

            urllib.request.urlretrieve(
                url,
                str(csv_path),
            )

            _validate_csv_file(csv_path)

            return str(csv_path)

        except Exception as exc:
            errors.append(
                f"{url}: {exc}"
            )

            if csv_path.exists():
                csv_path.unlink()

    error_message = "\n".join(errors)

    raise RuntimeError(
        f"Could not download real dataset {dataset_name!r}. "
        "Synthetic fallback is disabled.\n"
        f"{error_message}"
    )


def load_csv_dataset(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"CSV file contains no rows: {csv_path}"
        )

    return df


# Backward-compatible name used by earlier notebooks.
def download_adbench_dataset(
    dataset_name: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> str:
    """
    Backward-compatible wrapper for download_dataset.
    """
    return download_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
    )


# ============================================================
# Dataset validation
# ============================================================


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """
    Verify that required columns are present.
    """
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def validate_target(
    y: np.ndarray,
    target_name: str = "target",
) -> None:
    """
    Verify that the target contains exactly two classes.
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(
            f"{target_name} must be one-dimensional."
        )

    if len(y) == 0:
        raise ValueError(
            f"{target_name} is empty."
        )

    unique = np.unique(y)

    if len(unique) != 2:
        raise ValueError(
            f"{target_name} must contain two classes; "
            f"found {unique.tolist()}."
        )


def dataset_summary(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return basic dataset metadata.
    """
    summary = {
        "n_observations": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values_total": int(
            df.isna().sum().sum()
        ),
    }

    if label_col is not None:
        validate_required_columns(
            df,
            [label_col],
        )

        summary["label_column"] = label_col
        summary["label_values"] = (
            df[label_col]
            .value_counts(dropna=False)
            .to_dict()
        )

    return summary


# ============================================================
# Label conversion
# ============================================================


def convert_binary_label(
    series: pd.Series,
    positive_values: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """
    Convert a binary label series to {0,1}.

    If positive_values is supplied, matching values are mapped to 1.
    Otherwise, the larger of the two sorted unique values is mapped to 1.
    """
    values = series.copy()

    if positive_values is not None:
        positive_set = {
            str(value).strip().lower()
            for value in positive_values
        }

        y = values.map(
            lambda value: int(
                str(value).strip().lower()
                in positive_set
            )
        ).to_numpy()

        validate_target(
            y,
            target_name=series.name or "target",
        )

        return y

    nonmissing = values.dropna()
    unique_values = list(
        pd.unique(nonmissing)
    )

    if len(unique_values) != 2:
        raise ValueError(
            f"Expected binary label, found "
            f"{len(unique_values)} values: {unique_values}"
        )

    try:
        ordered = sorted(
            unique_values
        )
    except TypeError:
        ordered = sorted(
            unique_values,
            key=lambda value: str(value),
        )

    negative_value = ordered[0]
    positive_value = ordered[1]

    y = values.map(
        {
            negative_value: 0,
            positive_value: 1,
        }
    )

    if y.isna().any():
        raise ValueError(
            "Label conversion produced missing values."
        )

    y = y.astype(int).to_numpy()

    validate_target(
        y,
        target_name=series.name or "target",
    )

    return y


def convert_adbench_outlier_label(
    series: pd.Series,
) -> np.ndarray:
    """
    Convert common ADBench labels to binary labels.

    Common positive labels include:
    - o
    - outlier
    - anomaly
    - 1
    - true
    """
    positive_values = {
        "o",
        "outlier",
        "anomaly",
        "1",
        "true",
        "yes",
        "positive",
    }

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y = normalized.isin(
        positive_values
    ).astype(int).to_numpy()

    validate_target(
        y,
        target_name=series.name or "Outlier_label",
    )

    return y


# ============================================================
# Feature type detection
# ============================================================


def detect_feature_types(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[int, str]:
    """
    Detect continuous and binary features.

    Features with at most two observed values are classified as binary.
    Features with more than two observed values are classified as continuous.
    """
    feature_types = {}

    for index, feature_name in enumerate(
        feature_names
    ):
        values = pd.to_numeric(
            df[feature_name],
            errors="coerce",
        ).dropna()

        n_unique = values.nunique()

        if n_unique <= 2:
            feature_types[index] = "bin"
        else:
            feature_types[index] = "cont"

    return feature_types


def get_feature_metadata(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
) -> Tuple[List[str], Dict[int, str]]:
    """
    Return feature names and feature types.
    """
    validate_required_columns(
        df,
        [label_col],
    )

    feature_names = [
        column
        for column in df.columns
        if column != label_col
    ]

    feature_types = detect_feature_types(
        df=df,
        feature_names=feature_names,
    )

    return feature_names, feature_types


# ============================================================
# Type-specific preprocessing
# ============================================================


def coerce_numeric_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Convert selected feature columns to numeric values.
    """
    output = df[
        list(feature_names)
    ].copy()

    for feature_name in feature_names:
        output[feature_name] = pd.to_numeric(
            output[feature_name],
            errors="coerce",
        )

    return output


def fit_feature_imputation(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
):
    """
    Fit training-only imputation values.

    Continuous features use the training median.
    Binary features use the training mode.
    """
    imputation_values = {}

    for index, feature_name in enumerate(
        feature_names
    ):
        series = X_train[feature_name]

        if feature_types[index] == "cont":
            value = series.median()
        else:
            mode = series.mode(dropna=True)

            if mode.empty:
                value = 0.0
            else:
                value = mode.iloc[0]

        if pd.isna(value):
            value = 0.0

        imputation_values[
            feature_name
        ] = float(value)

    return imputation_values


def apply_feature_imputation(
    X: pd.DataFrame,
    imputation_values: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply previously fitted imputation values.
    """
    output = X.copy()

    for feature_name, value in (
        imputation_values.items()
    ):
        output[feature_name] = (
            output[feature_name]
            .fillna(value)
        )

    return output


def fit_standardizer(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
) -> StandardScaler:
    """
    Fit StandardScaler on training data only.
    """
    scaler = StandardScaler()

    scaler.fit(
        X_train[
            list(feature_names)
        ]
    )

    return scaler


def apply_standardizer(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply a training-fitted scaler.
    """
    return scaler.transform(
        X[
            list(feature_names)
        ]
    )


def prepare_benchmark_data(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
    positive_values: Optional[Sequence[Any]] = None,
    convert_adbench_labels: bool = True,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """
    Prepare a benchmark dataset for downstream analysis.

    This function performs deterministic type conversion and label conversion.
    It does not fit training-dependent imputation or scaling. Those operations
    should be performed within each benchmark repetition.
    """
    validate_required_columns(
        df,
        [label_col],
    )

    feature_names = [
        column
        for column in df.columns
        if column != label_col
    ]

    X = coerce_numeric_features(
        df=df,
        feature_names=feature_names,
    )

    feature_types = detect_feature_types(
        df=X,
        feature_names=feature_names,
    )

    raw_label = df[label_col]

    if (
        convert_adbench_labels
        and raw_label.dtype == object
    ):
        normalized = (
            raw_label
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if set(normalized.unique()).issubset(
            {
                "normal",
                "o",
                "outlier",
                "anomaly",
            }
        ):
            y = convert_adbench_outlier_label(
                raw_label
            )
        else:
            y = convert_binary_label(
                raw_label,
                positive_values=positive_values,
            )
    else:
        y = convert_binary_label(
            raw_label,
            positive_values=positive_values,
        )

    validate_target(
        y,
        target_name=label_col,
    )

    return X, y, feature_names, feature_types


# ============================================================
# Fixed Thyroid-subset creation
# ============================================================


def make_stratified_subset(
    df: pd.DataFrame,
    label_col: str,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a fixed stratified subset.

    The same saved subset should be reused for all subsequent analyses.
    """
    validate_required_columns(
        df,
        [label_col],
    )

    if n_samples >= len(df):
        return df.copy().reset_index(
            drop=True
        )

    rng = np.random.default_rng(
        random_state
    )

    groups = []

    label_values = df[
        label_col
    ].dropna().unique()

    for label_value in label_values:
        group = df.loc[
            df[label_col] == label_value
        ]

        proportion = len(group) / len(df)
        n_group = int(
            round(
                n_samples * proportion
            )
        )

        n_group = min(
            n_group,
            len(group),
        )

        if n_group > 0:
            selected = group.sample(
                n=n_group,
                random_state=int(
                    rng.integers(
                        0,
                        2**32 - 1,
                    )
                ),
            )

            groups.append(selected)

    subset = pd.concat(
        groups,
        axis=0,
    )

    if len(subset) < n_samples:
        remaining = df.drop(
            subset.index
        )

        extra = remaining.sample(
            n=n_samples - len(subset),
            random_state=int(
                rng.integers(
                    0,
                    2**32 - 1,
                )
            ),
        )

        subset = pd.concat(
            [subset, extra],
            axis=0,
        )

    if len(subset) > n_samples:
        subset = subset.sample(
            n=n_samples,
            random_state=int(
                rng.integers(
                    0,
                    2**32 - 1,
                )
            ),
        )

    return subset.sample(
        frac=1.0,
        random_state=int(
            rng.integers(
                0,
                2**32 - 1,
            )
        ),
    ).reset_index(
        drop=True
    )


def save_fixed_subset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save a fixed benchmark subset.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        exist_ok=True,
        parents=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )


def load_fixed_subset(
    input_path: str | Path,
) -> pd.DataFrame:
    """
    Load a previously saved fixed subset.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fixed subset not found: {input_path}"
        )

    return pd.read_csv(input_path)


# ============================================================
# Survival-data utilities
# ============================================================


def validate_survival_schema(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    time_col: str,
    event_col: str,
) -> None:
    """
    Validate survival dataset columns.
    """
    required = list(feature_names) + [
        time_col,
        event_col,
    ]

    validate_required_columns(
        df,
        required,
    )

    time_values = pd.to_numeric(
        df[time_col],
        errors="coerce",
    )

    event_values = pd.to_numeric(
        df[event_col],
        errors="coerce",
    )

    if time_values.isna().all():
        raise ValueError(
            f"No valid survival times in {time_col!r}."
        )

    if event_values.isna().all():
        raise ValueError(
            f"No valid event values in {event_col!r}."
        )

    event_values = event_values.dropna()

    if not set(
        event_values.astype(int).unique()
    ).issubset({0, 1}):
        raise ValueError(
            f"{event_col!r} must contain only 0/1 values."
        )


def prepare_survival_data(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """
    Convert survival dataset columns to numeric values and remove rows lacking
    valid time/event values.

    Training-dependent imputation and scaling should still be performed inside
    each training/test repetition.
    """
    validate_survival_schema(
        df=df,
        feature_names=feature_names,
        time_col=time_col,
        event_col=event_col,
    )

    columns = list(feature_names) + [
        time_col,
        event_col,
    ]

    output = df[
        columns
    ].copy()

    for column in columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output = output.dropna(
        subset=[
            time_col,
            event_col,
        ]
    ).copy()

    output[event_col] = (
        output[event_col]
        .astype(int)
        .clip(0, 1)
    )

    output = output.loc[
        output[time_col] > 0
    ].reset_index(
        drop=True
    )

    return output


# ============================================================
# Output utilities
# ============================================================


def save_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save a DataFrame as CSV.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        exist_ok=True,
        parents=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )


def save_json(
    payload: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save JSON metadata.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        exist_ok=True,
        parents=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )


def save_benchmark_outputs(
    results: Dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
) -> None:
    """
    Save outputs from benchmark.run_benchmark.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    if "rankings_by_repeat" in results:
        save_dataframe(
            results[
                "rankings_by_repeat"
            ],
            output_dir
            / f"{dataset_name}_rankings_by_repeat.csv",
        )

    if "rank_summary" in results:
        save_dataframe(
            results[
                "rank_summary"
            ],
            output_dir
            / f"{dataset_name}_rank_summary.csv",
        )

    if "repeat_results" in results:
        save_dataframe(
            results[
                "repeat_results"
            ],
            output_dir
            / f"{dataset_name}_repeat_results.csv",
        )

    if "metric_summary" in results:
        save_dataframe(
            results[
                "metric_summary"
            ],
            output_dir
            / f"{dataset_name}_metric_summary.csv",
        )

    metadata = {
        "dataset": dataset_name,
    }

    for key in [
        "config",
        "spearman_rank_correlation",
        "spearman_rank_pvalue",
        "mean_abs_rank_change",
    ]:
        if key in results:
            metadata[key] = results[key]

    save_json(
        metadata,
        output_dir
        / f"{dataset_name}_metadata.json",
    )


def create_analysis_metadata(
    df: pd.DataFrame,
    dataset_name: str,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
    label: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create reproducibility metadata for one analysis.
    """
    metadata = {
        "dataset": dataset_name,
        "n_observations": int(
            df.shape[0]
        ),
        "n_features": int(
            len(feature_names)
        ),
        "feature_names": list(
            feature_names
        ),
        "feature_types": {
            str(key): value
            for key, value in feature_types.items()
        },
        "n_continuous": int(
            sum(
                value == "cont"
                for value in feature_types.values()
            )
        ),
        "n_binary": int(
            sum(
                value == "bin"
                for value in feature_types.values()
            )
        ),
    }

    if label is not None:
        label = np.asarray(label)

        metadata["n_positive"] = int(
            label.sum()
        )

        metadata["positive_rate"] = float(
            label.mean()
        )

    if config is not None:
        metadata["config"] = config

    return metadata


def save_analysis_metadata(
    metadata: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save reproducibility metadata.
    """
    save_json(
        metadata,
        output_path,
    )


# ============================================================
# Compatibility aliases
# ============================================================


def create_features_thyroid(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
):
    """
    Compatibility helper for earlier Annthyroid notebooks.

    For new analyses, use prepare_benchmark_data instead.
    """
    X, y, feature_names, feature_types = (
        prepare_benchmark_data(
            df=df,
            label_col=label_col,
        )
    )

    processed = X.copy()
    processed[label_col] = y

    return (
        processed,
        feature_names,
        label_col,
    )
