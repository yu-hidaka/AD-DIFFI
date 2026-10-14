"""Reusable missing-data handling for mixed-type tabular data.

The module implements a leakage-safe imputation strategy:

1. Create missingness indicators before imputation.
2. Split data into training and test folds.
3. Fit imputers using the training fold only.
4. Transform both the training and test folds with the fitted imputers.

This design preserves the information that a value was missing while
preventing test-fold information from influencing imputation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

MISSING_INDICATOR_SUFFIX = "_missing"


@dataclass(frozen=True)
class MissingDataConfig:
    """Configuration for fold-wise missing-data handling.

    Parameters
    ----------
    continuous_columns:
        Columns treated as continuous numeric variables.
    binary_columns:
        Columns treated as binary or categorical variables encoded as
        numeric values.
    create_missing_indicators:
        Whether to append `{column}_missing` indicator columns.
    continuous_strategy:
        Imputation strategy for continuous columns.
    binary_strategy:
        Imputation strategy for binary or categorical columns.
    """

    continuous_columns: Sequence[str]
    binary_columns: Sequence[str]
    create_missing_indicators: bool = True
    continuous_strategy: str = "median"
    binary_strategy: str = "most_frequent"

    def validate_columns(self, data: pd.DataFrame) -> None:
        """Raise an error if configured columns are absent."""

        required_columns = list(self.continuous_columns) + list(
            self.binary_columns
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing configured columns: "
                f"{missing_columns}"
            )


def add_missing_indicators(
    data: pd.DataFrame,
    columns: Iterable[str],
    suffix: str = MISSING_INDICATOR_SUFFIX,
) -> tuple[pd.DataFrame, list[str]]:
    """Add binary missingness indicators for the specified columns.

    Parameters
    ----------
    data:
        Input data frame containing missing values.
    columns:
        Columns for which missingness indicators are created.
    suffix:
        Suffix appended to each original column name.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A copy of `data` with indicator columns and the names of the
        newly created indicator columns.
    """

    output = data.copy()
    indicator_columns: list[str] = []

    for column in columns:
        if column not in output.columns:
            raise ValueError(f"Column not found: {column}")

        indicator_column = f"{column}{suffix}"

        output[indicator_column] = (
            output[column]
            .isna()
            .astype("int8")
        )

        indicator_columns.append(indicator_column)

    return output, indicator_columns


class FoldWiseImputer:
    """Fit imputers on a training fold and apply them to both folds.

    The imputer is fitted separately for continuous and binary columns.
    It must be fitted independently in every train/test split or
    cross-validation fold.
    """

    def __init__(
        self,
        config: MissingDataConfig,
    ) -> None:
        self.config = config

        self.continuous_imputer = SimpleImputer(
            strategy=config.continuous_strategy,
            keep_empty_features=True,
        )

        self.binary_imputer = SimpleImputer(
            strategy=config.binary_strategy,
            keep_empty_features=True,
        )

        self._is_fitted = False

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit imputers on `X_train` and transform both folds.

        Parameters
        ----------
        X_train:
            Training-fold feature matrix. Missing values are allowed.
        X_test:
            Test-fold feature matrix. Missing values are allowed.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Imputed training and test feature matrices.
        """

        self.config.validate_columns(X_train)
        self.config.validate_columns(X_test)

        X_train = X_train.copy()
        X_test = X_test.copy()

        continuous_columns = list(
            self.config.continuous_columns
        )
        binary_columns = list(
            self.config.binary_columns
        )

        X_train_continuous = (
            self.continuous_imputer.fit_transform(
                X_train[continuous_columns]
            )
        )

        X_test_continuous = (
            self.continuous_imputer.transform(
                X_test[continuous_columns]
            )
        )

        X_train_binary = (
            self.binary_imputer.fit_transform(
                X_train[binary_columns]
            )
        )

        X_test_binary = (
            self.binary_imputer.transform(
                X_test[binary_columns]
            )
        )

        X_train.loc[:, continuous_columns] = (
            X_train_continuous
        )

        X_test.loc[:, continuous_columns] = (
            X_test_continuous
        )

        X_train.loc[:, binary_columns] = (
            X_train_binary
        )

        X_test.loc[:, binary_columns] = (
            X_test_binary
        )

        self._is_fitted = True

        return X_train, X_test

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform new data using previously fitted imputers."""

        if not self._is_fitted:
            raise RuntimeError(
                "FoldWiseImputer must be fitted before "
                "calling transform()."
            )

        self.config.validate_columns(X)

        X = X.copy()

        continuous_columns = list(
            self.config.continuous_columns
        )
        binary_columns = list(
            self.config.binary_columns
        )

        X.loc[:, continuous_columns] = (
            self.continuous_imputer.transform(
                X[continuous_columns]
            )
        )

        X.loc[:, binary_columns] = (
            self.binary_imputer.transform(
                X[binary_columns]
            )
        )

        return X


def make_fold_preprocess_fn(
    config: MissingDataConfig,
) -> callable:
    """Create a preprocessing function for use inside a benchmark.

    The returned function is intended to be called after each
    train/test split. It fits imputers only on the training fold.

    Returns
    -------
    callable
        A function with signature
        `(X_train, X_test, y_train, y_test)` that returns imputed
        `(X_train, X_test, y_train, y_test)`.
    """

    def preprocess_fold(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        imputer = FoldWiseImputer(config)

        X_train_imputed, X_test_imputed = (
            imputer.fit_transform(
                X_train=X_train,
                X_test=X_test,
            )
        )

        return (
            X_train_imputed,
            X_test_imputed,
            y_train,
            y_test,
        )

    return preprocess_fold
