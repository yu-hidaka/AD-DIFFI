"""
Data loading, validation, preprocessing, synthetic data generation,
and output utilities for AD-DIFFI.

This module does not:
- fit Isolation Forests;
- calculate DIFFI or AD-DIFFI scores;
- fit downstream models.

Synthetic data generation is explicit and separate from real-data download.
Training-dependent preprocessing is implemented through fit/transform helpers.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
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
# Synthetic mixed-type dataset
# ============================================================

SYNTHETIC_DATASET_NAME = (
    "synthetic_thyroid_like"
)

SYNTHETIC_LABEL_COLUMN = (
    "Outlier_label"
)

SYNTHETIC_FEATURE_NAMES = [
    "X1_Cont_Signal_Strong",
    "X2_Cont_Signal_Moderate",
    "X3_Cont_Signal_Weak",
    "X4_Cont_Noise_1",
    "X5_Cont_Noise_2",
    "X6_Cont_Noise_3",
    "X7_Bin_Signal_Strong",
    "X8_Bin_Signal_Weak",
] + [
    f"X{i}_Bin_Noise"
    for i in range(9, 28)
]

SYNTHETIC_FEATURE_TYPES: Dict[int, str] = {
    index: (
        "cont"
        if index < 6
        else "bin"
    )
    for index in range(27)
}

SYNTHETIC_SIGNAL_MASK = np.array(
    [
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    + [False] * 19,
    dtype=bool,
)

SYNTHETIC_SIGNAL_INDICES = np.flatnonzero(
    SYNTHETIC_SIGNAL_MASK
)

SYNTHETIC_NOISE_INDICES = np.flatnonzero(
    ~SYNTHETIC_SIGNAL_MASK
)

SYNTHETIC_CONTINUOUS_INDICES = np.arange(
    0,
    6,
    dtype=int,
)

SYNTHETIC_BINARY_INDICES = np.arange(
    6,
    27,
    dtype=int,
)


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """
    Configuration for the 27-feature synthetic mixed-type dataset.

    Feature structure
    -----------------
    X1--X3:
        Continuous signal features.

    X4--X6:
        Continuous noise features.

    X7--X8:
        Binary signal features.

    X9--X27:
        Binary noise features.
    """

    n_samples: int = 7200
    contamination: float = 0.0742
    seed: int = 42

    normal_cont_mean: float = 0.0
    normal_cont_sd: float = 1.0

    cont_shift_strong: float = 5.0
    cont_shift_moderate: float = 3.0
    cont_shift_weak: float = 2.0

    normal_binary_probability: float = 0.05
    binary_probability_strong: float = 0.80
    binary_probability_weak: float = 0.50

    noise_binary_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(
                "n_samples must be at least 2."
            )

        if not (
            0.0 < self.contamination < 1.0
        ):
            raise ValueError(
                "contamination must be in (0, 1)."
            )

        if self.normal_cont_sd <= 0:
            raise ValueError(
                "normal_cont_sd must be positive."
            )

        probabilities = [
            self.normal_binary_probability,
            self.binary_probability_strong,
            self.binary_probability_weak,
            self.noise_binary_probability,
        ]

        if any(
            not 0.0 <= probability <= 1.0
            for probability in probabilities
        ):
            raise ValueError(
                "Binary probabilities must be in [0, 1]."
            )


def validate_feature_metadata(
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
) -> None:
    """
    Validate complete feature-type metadata.
    """
    feature_names = list(feature_names)

    if len(feature_names) == 0:
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


def generate_synthetic_thyroid_like(
    config: SyntheticDatasetConfig | None = None,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """
    Generate a reproducible 27-feature mixed-type dataset.

    The returned dataframe includes the label column
    ``Outlier_label``. The returned y array contains 0/1 labels.

    This is synthetic data. It must not be described as a real clinical
    Annthyroid cohort or as a downloaded ADBench dataset.
    """
    if config is None:
        config = SyntheticDatasetConfig()

    rng = np.random.default_rng(
        config.seed
    )

    n_anomaly = int(
        round(
            config.n_samples
            * config.contamination
        )
    )

    n_anomaly = max(
        1,
        min(
            config.n_samples - 1,
            n_anomaly,
        ),
    )

    anomaly_indices = rng.choice(
        config.n_samples,
        size=n_anomaly,
        replace=False,
    )

    y = np.zeros(
        config.n_samples,
        dtype=int,
    )

    y[anomaly_indices] = 1

    X = np.empty(
        (
            config.n_samples,
            len(SYNTHETIC_FEATURE_NAMES),
        ),
        dtype=float,
    )

    # --------------------------------------------------------
    # Continuous features
    # --------------------------------------------------------

    # X1--X6: continuous features.
    X[:, 0:6] = rng.normal(
        loc=config.normal_cont_mean,
        scale=config.normal_cont_sd,
        size=(
            config.n_samples,
            6,
        ),
    )

    # Continuous signal features.
    X[anomaly_indices, 0] += (
        config.cont_shift_strong
    )

    X[anomaly_indices, 1] += (
        config.cont_shift_moderate
    )

    X[anomaly_indices, 2] += (
        config.cont_shift_weak
    )

    # X4--X6 remain continuous noise.

    # --------------------------------------------------------
    # Binary features
    # --------------------------------------------------------

    # X9--X27: binary noise features.
    X[:, 8:27] = rng.binomial(
        n=1,
        p=config.noise_binary_probability,
        size=(
            config.n_samples,
            19,
        ),
    )

    # X7--X8: binary signal features.
    X[:, 6] = rng.binomial(
        n=1,
        p=config.normal_binary_probability,
        size=config.n_samples,
    )

    X[:, 7] = rng.binomial(
        n=1,
        p=config.normal_binary_probability,
        size=config.n_samples,
    )

    X[anomaly_indices, 6] = rng.binomial(
        n=1,
        p=config.binary_probability_strong,
        size=n_anomaly,
    )

    X[anomaly_indices, 7] = rng.binomial(
        n=1,
        p=config.binary_probability_weak,
        size=n_anomaly,
    )

    # Shuffle observations so that anomalies are not ordered.
    permutation = rng.permutation(
        config.n_samples
    )

    X = X[
        permutation
    ]

    y = y[
        permutation
    ]

    df = pd.DataFrame(
        X,
        columns=SYNTHETIC_FEATURE_NAMES,
    )

    df[SYNTHETIC_LABEL_COLUMN] = np.where(
        y == 1,
        "o",
        "normal",
    )

    validate_feature_metadata(
        feature_names=SYNTHETIC_FEATURE_NAMES,
        feature_types=SYNTHETIC_FEATURE_TYPES,
    )

    return (
        df,
        y,
        SYNTHETIC_FEATURE_NAMES.copy(),
        SYNTHETIC_FEATURE_TYPES.copy(),
    )


def save_synthetic_thyroid_like(
    output_path: str | Path,
    config: SyntheticDatasetConfig | None = None,
) -> Dict[str, Any]:
    """
    Generate and save the reproducible synthetic dataset.
    """
    if config is None:
        config = SyntheticDatasetConfig()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        exist_ok=True,
        parents=True,
    )

    df, y, feature_names, feature_types = (
        generate_synthetic_thyroid_like(
            config=config
        )
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return {
        "dataset_name": (
            SYNTHETIC_DATASET_NAME
        ),
        "path": str(output_path),
        "n_samples": int(len(df)),
        "n_features": int(
            len(feature_names)
        ),
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
        "n_anomaly": int(y.sum()),
        "contamination": float(y.mean()),
        "seed": int(config.seed),
        "feature_names": feature_names,
        "feature_types": {
            str(index): feature_type
            for index, feature_type
            in feature_types.items()
        },
    }


def load_synthetic_thyroid_like(
    input_path: str | Path,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """
    Load a previously saved synthetic mixed-type dataset.
    """
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    expected_columns = (
        SYNTHETIC_FEATURE_NAMES
        + [SYNTHETIC_LABEL_COLUMN]
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            f"{missing_columns}"
        )

    df = df[
        expected_columns
    ].copy()

    X = df[
        SYNTHETIC_FEATURE_NAMES
    ].copy()

    for column in SYNTHETIC_FEATURE_NAMES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    if X.isna().any().any():
        raise ValueError(
            "Synthetic dataset contains invalid "
            "feature values."
        )

    labels = (
        df[SYNTHETIC_LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_labels = set(labels) - {
        "normal",
        "o",
    }

    if invalid_labels:
        raise ValueError(
            "Unexpected synthetic label values: "
            f"{sorted(invalid_labels)}"
        )

    y = labels.eq(
        "o"
    ).astype(int).to_numpy()

    return (
        X,
        y,
        SYNTHETIC_FEATURE_NAMES.copy(),
        SYNTHETIC_FEATURE_TYPES.copy(),
    )


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
    data_dir = Path(
        data_dir
    )

    data_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    csv_path = (
        data_dir
        / f"{dataset_name}.csv"
    )

    if csv_path.exists() and not overwrite:
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

    error_message = "\n".join(
        errors
    )

    raise RuntimeError(
        f"Could not download real dataset "
        f"{dataset_name!r}. "
        "Synthetic generation is separate and "
        "must be called explicitly.\n"
        f"{error_message}"
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
    Convert common ADBench labels to binary labels.
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


# ============================================================
# Feature type detection
# ============================================================


def detect_feature_types(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[int, str]:
    """
    Detect continuous and binary features.

    Features with at most two observed numeric values are classified as binary.
    Features with more than two observed values are classified as continuous.
    """
    feature_types = {}

    for index, feature_name in enumerate(
        feature_names
    ):
        if feature_name not in df.columns:
            raise ValueError(
                f"Feature not found: {feature_name}"
            )

        values = pd.to_numeric(
            df[feature_name],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise ValueError(
                f"Feature contains no numeric values: "
                f"{feature_name}"
            )

        n_unique = values.nunique()

        if n_unique <= 2:
            feature_types[index] = "bin"
        else:
            feature_types[index] = "cont"

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

    imputation_values = {}

    for index, feature_name in enumerate(
        feature_names
    ):
        if feature_name not in X_train.columns:
            raise ValueError(
                f"Feature not found in X_train: "
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


def fit_mixed_type_preprocessor(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
) -> Tuple[
    Dict[str, float],
    Optional[StandardScaler],
    List[int],
]:
    """
    Fit imputation and continuous-feature standardization on training data.

    Binary features are not standardized.
    """
    validate_feature_metadata(
        feature_names=list(feature_names),
        feature_types=feature_types,
    )

    imputation_values = fit_feature_imputation(
        X_train=X_train,
        feature_names=feature_names,
        feature_types=feature_types,
    )

    X_train_imputed = (
        apply_feature_imputation(
            X=X_train[
                list(feature_names)
            ],
            imputation_values=imputation_values,
        )
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

    scaler = None

    if continuous_names:
        scaler = fit_standardizer(
            X_train=X_train_imputed,
            feature_names=continuous_names,
        )

    return (
        imputation_values,
        scaler,
        continuous_indices,
    )


def transform_mixed_type_data(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    feature_types: Dict[int, str],
    imputation_values: Dict[str, float],
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    """
    Apply a training-fitted mixed-type preprocessor.

    Continuous features are standardized when a scaler is supplied.
    Binary features remain on their original 0/1 scale.
    """
    validate_feature_metadata(
        feature_names=list(feature_names),
        feature_types=feature_types,
    )

    X_selected = X[
        list(feature_names)
    ].copy()

    X_imputed = apply_feature_imputation(
        X=X_selected,
        imputation_values=imputation_values,
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
            continuous_indices,
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
    Split mixed-type data and apply training-only preprocessing.

    Processing order
    ----------------
    1. Stratified train/test split.
    2. Fit imputation on X_train.
    3. Fit continuous standardizer on X_train.
    4. Transform X_train and X_test.
    5. Leave binary features unstandardized.
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
        _,
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
    Prepare a benchmark dataset.

    This function performs deterministic numeric conversion and label conversion.
    It does not fit training-dependent preprocessing.
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
# Fixed subset creation
# ============================================================


def make_stratified_subset(
    df: pd.DataFrame,
    label_col: str,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a fixed stratified subset.

    The saved subset should be reused for all subsequent analyses.
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

    label_values = (
        df[label_col]
        .dropna()
        .unique()
    )

    groups = []

    for label_value in label_values:
        group = df.loc[
            df[label_col] == label_value
        ]

        proportion = (
            len(group) / len(df)
        )

        n_group = int(
            round(
                n_samples
                * proportion
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

            groups.append(
                selected
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
            extra = remaining.sample(
                n=n_extra,
                random_state=int(
                    rng.integers(
                        0,
                        2**32 - 1,
                    )
                ),
            )

            subset = pd.concat(
                [
                    subset,
                    extra,
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
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fixed subset not found: {input_path}"
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
    output_path = Path(
        output_path
    )

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
    Save outputs from benchmark-style result dictionaries.

    This function supports the older dictionary-based notebook output format.
    The current benchmark.py also provides its own output saver.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
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
    Create reproducibility metadata for one analysis.
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
            str(key): value
            for key, value
            in feature_types.items()
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


# ============================================================
# Compatibility aliases
# ============================================================


def create_features_thyroid(
    df: pd.DataFrame,
    label_col: str = "Outlier_label",
):
    """
    Compatibility helper for earlier Annthyroid notebooks.

    Returns
    -------
    processed:
        Numeric feature dataframe with a binary label column appended.

    feature_names:
        Feature column names.

    label_col:
        Name of the appended binary label column.

    For new analyses, use prepare_benchmark_data() and
    train_test_split_mixed_type().
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
