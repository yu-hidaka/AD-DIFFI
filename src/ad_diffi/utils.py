"""
Data loading, validation, preprocessing, and output utilities for AD-DIFFI.

This module supports real-world benchmark datasets and raw UCI thyroid data.

This module does not:
- generate synthetic fallback datasets;
- fit Isolation Forests;
- calculate DIFFI or AD-DIFFI scores;
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths and dataset URLs
# ============================================================

DEFAULT_DATA_DIR = Path(
    "/tmp/ad_diffi_data"
)

DEFAULT_DATASET_URLS = {
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

RAW_THYROID_DATA_URLS = [
    (
        "https://rioultf.users.greyc.fr/"
        "uci/files/thyroid-disease/"
        "thyroid0387.data"
    ),
]


# ============================================================
# Raw UCI Thyroid schema
# ============================================================

RAW_THYROID_LABEL_COLUMN = (
    "thyroid_class"
)

RAW_THYROID_FEATURE_NAMES = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "referral_source",
]

RAW_THYROID_BINARY_COLUMNS = [
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
]

RAW_THYROID_CONTINUOUS_COLUMNS = [
    "age",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
]

RAW_THYROID_CATEGORICAL_COLUMNS = [
    "referral_source",
]


# ============================================================
# General validation
# ============================================================


def validate_feature_metadata(
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
) -> None:
    """
    Validate complete feature-type metadata.
    """
    feature_names = list(feature_names)

    if not feature_names:
        raise ValueError(
            "feature_names must not be empty."
        )

    expected_indices = set(
        range(len(feature_names))
    )

    observed_indices = set(
        feature_types
    )

    missing_indices = (
        expected_indices
        - observed_indices
    )

    extra_indices = (
        observed_indices
        - expected_indices
    )

    if missing_indices:
        raise ValueError(
            "Missing feature types: "
            f"{sorted(missing_indices)}"
        )

    if extra_indices:
        raise ValueError(
            "Invalid feature indices: "
            f"{sorted(extra_indices)}"
        )

    invalid_types = {
        index: feature_types[index]
        for index in expected_indices
        if feature_types[index]
        not in {"cont", "bin"}
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )


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
    y = np.asarray(
        y
    )

    if y.ndim != 1:
        raise ValueError(
            f"{target_name} must be one-dimensional."
        )

    if len(y) == 0:
        raise ValueError(
            f"{target_name} is empty."
        )

    unique = np.unique(
        y
    )

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
        "n_observations": int(
            df.shape[0]
        ),
        "n_columns": int(
            df.shape[1]
        ),
        "columns": list(
            df.columns
        ),
        "missing_values_total": int(
            df.isna().sum().sum()
        ),
    }

    if label_col is not None:
        validate_required_columns(
            df,
            [label_col],
        )

        summary["label_column"] = (
            label_col
        )

        summary["label_values"] = (
            df[label_col]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    return summary


# ============================================================
# Download and CSV utilities
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

    No synthetic fallback is used.
    """
    dataset_name = dataset_name.lower()
    data_dir = Path(
        data_dir
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        data_dir
        / f"{dataset_name}.csv"
    )

    if (
        csv_path.exists()
        and not overwrite
    ):
        _validate_csv_file(
            csv_path
        )

        return str(csv_path)

    if urls is None:
        if dataset_name not in (
            DEFAULT_DATASET_URLS
        ):
            raise ValueError(
                "No default URLs registered "
                f"for {dataset_name!r}."
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

            _validate_csv_file(
                csv_path
            )

            return str(csv_path)

        except Exception as exc:
            errors.append(
                f"{url}: {exc}"
            )

            if csv_path.exists():
                csv_path.unlink()

    raise RuntimeError(
        f"Could not download dataset "
        f"{dataset_name!r}.\n"
        + "\n".join(errors)
    )


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


def load_csv_dataset(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    """
    csv_path = Path(
        csv_path
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    if df.empty:
        raise ValueError(
            f"CSV file contains no rows: {csv_path}"
        )

    return df


# ============================================================
# Raw UCI Thyroid loading
# ============================================================


def download_raw_thyroid_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    urls: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> str:
    """
    Download the raw UCI thyroid0387.data file.
    """
    data_dir = Path(
        data_dir
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        data_dir
        / "thyroid0387.data"
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        if output_path.stat().st_size == 0:
            raise ValueError(
                f"Raw thyroid file is empty: "
                f"{output_path}"
            )

        return str(output_path)

    if urls is None:
        urls = RAW_THYROID_DATA_URLS

    errors = []

    for url in urls:
        try:
            if output_path.exists():
                output_path.unlink()

            urllib.request.urlretrieve(
                url,
                str(output_path),
            )

            if not output_path.exists():
                raise FileNotFoundError(
                    "Downloaded raw thyroid file "
                    "was not created."
                )

            if output_path.stat().st_size == 0:
                raise ValueError(
                    "Downloaded raw thyroid file "
                    "is empty."
                )

            return str(output_path)

        except Exception as exc:
            errors.append(
                f"{url}: {exc}"
            )

            if output_path.exists():
                output_path.unlink()

    raise RuntimeError(
        "Could not download raw UCI "
        "thyroid0387.data.\n"
        + "\n".join(errors)
    )


def load_raw_thyroid_table(
    data_path: str | Path,
) -> pd.DataFrame:
    """
    Load the raw thyroid file without assuming a header row.

    The raw file schema must be checked before assigning final column names.
    """
    data_path = Path(
        data_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw thyroid file not found: "
            f"{data_path}"
        )

    df = pd.read_csv(
        data_path,
        header=None,
        na_values=[
            "?",
            "NA",
            "nan",
            "",
        ],
    )

    if df.empty:
        raise ValueError(
            "Raw thyroid file contains no rows."
        )

    return df


def assign_raw_thyroid_columns(
    df: pd.DataFrame,
    feature_names: Optional[
        Sequence[str]
    ] = None,
    label_col: str = RAW_THYROID_LABEL_COLUMN,
) -> pd.DataFrame:
    """
    Assign stable names to a raw thyroid table.

    By default, generic names are used for the feature columns. This avoids
    silently assuming that a particular mirror has a specific raw schema.
    """
    df = df.copy()

    n_columns = df.shape[1]

    if n_columns < 2:
        raise ValueError(
            "Raw thyroid table must contain "
            "features and one label column."
        )

    if feature_names is None:
        feature_names = [
            f"thyroid_feature_{index + 1}"
            for index in range(
                n_columns - 1
            )
        ]
    else:
        feature_names = list(
            feature_names
        )

        if len(feature_names) != (
            n_columns - 1
        ):
            raise ValueError(
                "feature_names length must equal "
                "the number of columns minus one."
            )

    columns = list(
        feature_names
    ) + [
        label_col
    ]

    df.columns = columns

    return df


# ============================================================
# Label conversion
# ============================================================


def convert_binary_label(
    series: pd.Series,
    positive_values: Optional[
        Sequence[Any]
    ] = None,
) -> np.ndarray:
    """
    Convert a binary label series to {0,1}.
    """
    values = series.copy()

    if positive_values is not None:
        positive_set = {
            str(value)
            .strip()
            .lower()
            for value in positive_values
        }

        y = values.map(
            lambda value: int(
                str(value)
                .strip()
                .lower()
                in positive_set
            )
        ).to_numpy()

        validate_target(
            y,
            target_name=(
                series.name
                or "target"
            ),
        )

        return y

    nonmissing = values.dropna()

    unique_values = list(
        pd.unique(
            nonmissing
        )
    )

    if len(unique_values) != 2:
        raise ValueError(
            "Expected binary label, found "
            f"{len(unique_values)} values: "
            f"{unique_values}"
        )

    try:
        ordered = sorted(
            unique_values
        )
    except TypeError:
        ordered = sorted(
            unique_values,
            key=lambda value: str(
                value
            ),
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
            "Label conversion produced "
            "missing values."
        )

    y = y.astype(
        int
    ).to_numpy()

    validate_target(
        y,
        target_name=(
            series.name
            or "target"
        ),
    )

    return y


def convert_adbench_outlier_label(
    series: pd.Series,
) -> np.ndarray:
    """
    Convert common anomaly labels to binary labels.
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
    ).astype(
        int
    ).to_numpy()

    validate_target(
        y,
        target_name=(
            series.name
            or "Outlier_label"
        ),
    )

    return y


def convert_raw_thyroid_label(
    series: pd.Series,
    normal_values: Optional[
        Sequence[Any]
    ] = None,
) -> np.ndarray:
    """
    Convert raw thyroid class labels to anomaly labels.

    The default mapping treats normal/negative/n/1 as inliers and all other
    non-missing classes as anomalies. The mapping must be checked against the
    actual raw file before publication.
    """
    if normal_values is None:
        normal_values = [
            "normal",
            "negative",
            "n",
            "1",
            "normal.",
        ]

    normal_tokens = {
        str(value)
        .strip()
        .lower()
        for value in normal_values
    }

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y = (
        ~normalized.isin(
            normal_tokens
        )
    ).astype(
        int
    ).to_numpy()

    validate_target(
        y,
        target_name=(
            series.name
            or RAW_THYROID_LABEL_COLUMN
        ),
    )

    return y


# ============================================================
# Feature encoding and type detection
# ============================================================


def normalize_binary_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common binary encodings to numeric 0/1 values.
    """
    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "f": 0.0,
        "false": 0.0,
        "n": 0.0,
        "no": 0.0,
        "0": 0.0,
        "t": 1.0,
        "true": 1.0,
        "y": 1.0,
        "yes": 1.0,
        "1": 1.0,
    }

    return normalized.map(
        mapping
    )


def detect_feature_types(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[int, str]:
    """
    Detect binary and continuous features from numeric values.

    Features with at most two observed values are classified as binary.
    """
    feature_types = {}

    for index, feature_name in enumerate(
        feature_names
    ):
        if feature_name not in df.columns:
            raise ValueError(
                f"Feature not found: "
                f"{feature_name}"
            )

        values = pd.to_numeric(
            df[feature_name],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise ValueError(
                f"Feature has no numeric values: "
                f"{feature_name}"
            )

        feature_types[index] = (
            "bin"
            if values.nunique() <= 2
            else "cont"
        )

    validate_feature_metadata(
        feature_names=list(feature_names),
        feature_types=feature_types,
    )

    return feature_types


def get_feature_metadata(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
) -> Tuple[
    List[str],
    Dict[int, str],
]:
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

    return (
        feature_names,
        feature_types,
    )


# ============================================================
# Raw UCI thyroid preparation
# ============================================================


def prepare_raw_thyroid_data(
    df: pd.DataFrame,
    label_col: str = RAW_THYROID_LABEL_COLUMN,
    normal_values: Optional[
        Sequence[Any]
    ] = None,
    include_categorical: bool = False,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """
    Prepare raw UCI thyroid data for benchmark analysis.

    The default path keeps the six continuous variables and binary clinical
    indicators. Multi-level categorical variables such as referral_source are
    excluded unless include_categorical is implemented explicitly in a
    dataset-specific preprocessing pipeline.

    This function does not fit training-dependent imputation or scaling.
    """
    validate_required_columns(
        df,
        [label_col],
    )

    y = convert_raw_thyroid_label(
        df[label_col],
        normal_values=normal_values,
    )

    available_binary = [
        column
        for column in RAW_THYROID_BINARY_COLUMNS
        if column in df.columns
    ]

    available_continuous = [
        column
        for column in RAW_THYROID_CONTINUOUS_COLUMNS
        if column in df.columns
    ]

    if not available_binary:
        raise ValueError(
            "No raw thyroid binary columns "
            "were found."
        )

    if not available_continuous:
        raise ValueError(
            "No raw thyroid continuous columns "
            "were found."
        )

    if include_categorical:
        raise NotImplementedError(
            "Multi-level categorical encoding "
            "must be fitted inside each training "
            "repetition. Use a dataset-specific "
            "ColumnTransformer pipeline."
        )

    feature_names = (
        available_continuous
        + available_binary
    )

    X = pd.DataFrame(
        index=df.index
    )

    feature_types = {}

    for feature_name in available_continuous:
        X[feature_name] = pd.to_numeric(
            df[feature_name],
            errors="coerce",
        )

        feature_types[
            len(feature_types)
        ] = "cont"

    for feature_name in available_binary:
        X[feature_name] = (
            normalize_binary_series(
                df[feature_name]
            )
        )

        feature_types[
            len(feature_types)
        ] = "bin"

    X = X[
        feature_names
    ].copy()

    if X.shape[1] == 0:
        raise ValueError(
            "No usable raw thyroid features "
            "were created."
        )

    validate_feature_metadata(
        feature_names=feature_names,
        feature_types=feature_types,
    )

    return (
        X,
        y,
        feature_names,
        feature_types,
    )


# ============================================================
# Generic benchmark preparation
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


def prepare_benchmark_data(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
    positive_values: Optional[
        Sequence[Any]
    ] = None,
    convert_adbench_labels: bool = True,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """
    Prepare a numeric benchmark dataset.

    This function is appropriate for already encoded benchmark CSV files.
    It is not appropriate for raw categorical thyroid data.
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

    raw_label = df[
        label_col
    ]

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

        if set(
            normalized.unique()
        ).issubset(
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

    return (
        X,
        y,
        feature_names,
        feature_types,
    )


# ============================================================
# Training-dependent preprocessing
# ============================================================


def fit_feature_imputation(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
) -> Dict[str, float]:
    """
    Fit training-only imputation values.

    Continuous features use the training median.
    Binary features use the training mode.
    """
    validate_feature_metadata(
        feature_names=list(feature_names),
        feature_types=feature_types,
    )

    values = {}

    for index, feature_name in enumerate(
        feature_names
    ):
        if feature_name not in X_train.columns:
            raise ValueError(
                f"Feature not found: "
                f"{feature_name}"
            )

        series = pd.to_numeric(
            X_train[feature_name],
            errors="coerce",
        )

        if feature_types[index] == "cont":
            value = series.median()
        else:
            mode = series.mode(
                dropna=True
            )

            value = (
                0.0
                if mode.empty
                else mode.iloc[0]
            )

        if pd.isna(value):
            value = 0.0

        values[feature_name] = float(
            value
        )

    return values


def apply_feature_imputation(
    X: pd.DataFrame,
    imputation_values: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply training-fitted imputation values.
    """
    output = X.copy()

    for feature_name, value in (
        imputation_values.items()
    ):
        if feature_name not in output.columns:
            raise ValueError(
                f"Feature not found: "
                f"{feature_name}"
            )

        output[feature_name] = (
            pd.to_numeric(
                output[feature_name],
                errors="coerce",
            )
            .fillna(value)
        )

    return output


def fit_mixed_type_preprocessor(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
) -> Tuple[
    Dict[str, float],
    Optional[StandardScaler],
]:
    """
    Fit training-only imputation and continuous scaling.

    Binary features are not standardized.
    """
    imputation_values = (
        fit_feature_imputation(
            X_train=X_train,
            feature_names=feature_names,
            feature_types=feature_types,
        )
    )

    X_train_imputed = (
        apply_feature_imputation(
            X=X_train[
                list(feature_names)
            ],
            imputation_values=imputation_values,
        )
    )

    continuous_names = [
        feature_names[index]
        for index, feature_type
        in feature_types.items()
        if feature_type == "cont"
    ]

    scaler = None

    if continuous_names:
        scaler = StandardScaler()

        scaler.fit(
            X_train_imputed[
                continuous_names
            ]
        )

    return (
        imputation_values,
        scaler,
    )


def transform_mixed_type_data(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
    imputation_values: Dict[str, float],
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    """
    Transform data with a training-fitted preprocessor.

    Continuous features are standardized.
    Binary features remain on their 0/1 scale.
    """
    X_selected = X[
        list(feature_names)
    ].copy()

    X_imputed = (
        apply_feature_imputation(
            X=X_selected,
            imputation_values=imputation_values,
        )
    )

    output = X_imputed.to_numpy(
        dtype=float,
        copy=True,
    )

    continuous_indices = [
        index
        for index, feature_type
        in feature_types.items()
        if feature_type == "cont"
    ]

    continuous_names = [
        feature_names[index]
        for index in continuous_indices
    ]

    if scaler is not None and continuous_names:
        output[
            :,
            continuous_indices
        ] = scaler.transform(
            X_imputed[
                continuous_names
            ]
        )

    return output


def train_test_split_mixed_type(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_types: Dict[int, str],
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split data and fit preprocessing using training data only.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    y = np.asarray(
        y,
        dtype=int,
    )

    if len(X) != len(y):
        raise ValueError(
            "X and y must have the same "
            "number of rows."
        )

    feature_names = list(
        X.columns
    )

    validate_feature_metadata(
        feature_names=feature_names,
        feature_types=feature_types,
    )

    validate_target(
        y,
        target_name="y",
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    )

    (
        imputation_values,
        scaler,
    ) = fit_mixed_type_preprocessor(
        X_train=X_train,
        feature_names=feature_names,
        feature_types=feature_types,
    )

    X_train_processed = (
        transform_mixed_type_data(
            X=X_train,
            feature_names=feature_names,
            feature_types=feature_types,
            imputation_values=imputation_values,
            scaler=scaler,
        )
    )

    X_test_processed = (
        transform_mixed_type_data(
            X=X_test,
            feature_names=feature_names,
            feature_types=feature_types,
            imputation_values=imputation_values,
            scaler=scaler,
        )
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
    )


# ============================================================
# Fixed subset utilities
# ============================================================


def make_stratified_subset(
    df: pd.DataFrame,
    label_col: str,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a fixed stratified subset.
    """
    validate_required_columns(
        df,
        [label_col],
    )

    if n_samples < 1:
        raise ValueError(
            "n_samples must be positive."
        )

    if n_samples >= len(df):
        return df.copy().reset_index(
            drop=True
        )

    rng = np.random.default_rng(
        random_state
    )

    groups = []

    for label_value in (
        df[label_col]
        .dropna()
        .unique()
    ):
        group = df.loc[
            df[label_col] == label_value
        ]

        n_group = min(
            len(group),
            int(
                round(
                    n_samples
                    * len(group)
                    / len(df)
                )
            ),
        )

        if n_group > 0:
            groups.append(
                group.sample(
                    n=n_group,
                    random_state=int(
                        rng.integers(
                            0,
                            2**32 - 1,
                        )
                    ),
                )
            )

    if not groups:
        raise ValueError(
            "Could not create a stratified subset."
        )

    subset = pd.concat(
        groups,
        axis=0,
    )

    if len(subset) < n_samples:
        remaining = df.drop(
            subset.index
        )

        n_extra = min(
            n_samples - len(subset),
            len(remaining),
        )

        if n_extra > 0:
            subset = pd.concat(
                [
                    subset,
                    remaining.sample(
                        n=n_extra,
                        random_state=int(
                            rng.integers(
                                0,
                                2**32 - 1,
                            )
                        ),
                    ),
                ],
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
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fixed subset not found: "
            f"{input_path}"
        )

    return pd.read_csv(
        input_path
    )


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
    required = list(
        feature_names
    ) + [
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
            f"No valid survival times in "
            f"{time_col!r}."
        )

    if event_values.isna().all():
        raise ValueError(
            f"No valid event values in "
            f"{event_col!r}."
        )

    event_values = event_values.dropna()

    if not set(
        event_values.astype(int).unique()
    ).issubset({0, 1}):
        raise ValueError(
            f"{event_col!r} must contain "
            "only 0/1 values."
        )


def prepare_survival_data(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """
    Prepare survival data.
    """
    validate_survival_schema(
        df=df,
        feature_names=feature_names,
        time_col=time_col,
        event_col=event_col,
    )

    columns = list(
        feature_names
    ) + [
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
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def create_analysis_metadata(
    df: pd.DataFrame,
    dataset_name: str,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
    label: Optional[np.ndarray] = None,
    config: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Create reproducibility metadata.
    """
    validate_feature_metadata(
        feature_names=list(feature_names),
        feature_types=feature_types,
    )

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
            str(index): value
            for index, value
            in feature_types.items()
        },
        "n_continuous": int(
            sum(
                value == "cont"
                for value
                in feature_types.values()
            )
        ),
        "n_binary": int(
            sum(
                value == "bin"
                for value
                in feature_types.values()
            )
        ),
    }

    if label is not None:
        label = np.asarray(
            label,
            dtype=int,
        )

        validate_target(
            label,
            target_name="label",
        )

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


def save_benchmark_outputs(
    results: Dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
) -> None:
    """
    Save dictionary-style benchmark results.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mapping = {
        "rankings_by_repeat": (
            f"{dataset_name}_rankings_by_repeat.csv"
        ),
        "rank_summary": (
            f"{dataset_name}_rank_summary.csv"
        ),
        "repeat_results": (
            f"{dataset_name}_repeat_results.csv"
        ),
        "metric_summary": (
            f"{dataset_name}_metric_summary.csv"
        ),
    }

    for key, filename in (
        output_mapping.items()
    ):
        if key in results:
            save_dataframe(
                results[key],
                output_dir / filename,
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


# ============================================================
# Compatibility helper
# ============================================================


def create_features_thyroid(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
):
    """
    Backward-compatible helper for already encoded thyroid-like data.

    For raw UCI thyroid data, use prepare_raw_thyroid_data().
    """
    X, y, feature_names, feature_types = (
        prepare_benchmark_data(
            df=df,
            label_col=label_col,
        )
    )

    processed = X.copy()

    processed[label_col] = np.where(
        y == 1,
        "o",
        "normal",
    )

    return (
        processed,
        feature_names,
        label_col,
    )
