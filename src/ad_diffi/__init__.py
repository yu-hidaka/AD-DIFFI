"""
AD-DIFFI package.

The package exposes:
- AD-DIFFI core scoring;
- real-world benchmark utilities;
- repeated benchmark evaluation.
"""

from .core import (
    compute_enrichment_ad_diffi,
)

from .benchmark import (
    AD_DIFFI_METHOD,
    ORIGINAL_METHOD,
    BenchmarkConfig,
    compute_ad_diffi_importance,
    fit_downstream_models,
    original_diffi_importance,
    run_benchmark,
    run_one_repeat,
    save_benchmark_outputs,
    select_top_features,
    summarize_benchmark,
    summarize_feature_scores,
)

from .utils import (
    RAW_THYROID_BINARY_COLUMNS,
    RAW_THYROID_CONTINUOUS_COLUMNS,
    RAW_THYROID_CATEGORICAL_COLUMNS,
    RAW_THYROID_DATA_URLS,
    RAW_THYROID_FEATURE_NAMES,
    RAW_THYROID_LABEL_COLUMN,
    assign_raw_thyroid_columns,
    create_analysis_metadata,
    convert_raw_thyroid_label,
    detect_feature_types,
    download_dataset,
    download_raw_thyroid_dataset,
    fit_mixed_type_preprocessor,
    load_csv_dataset,
    load_raw_thyroid_table,
    prepare_benchmark_data,
    prepare_raw_thyroid_data,
    save_analysis_metadata,
    save_dataframe,
    train_test_split_mixed_type,
    transform_mixed_type_data,
)

__all__ = [
    "compute_enrichment_ad_diffi",
    "AD_DIFFI_METHOD",
    "ORIGINAL_METHOD",
    "BenchmarkConfig",
    "compute_ad_diffi_importance",
    "fit_downstream_models",
    "original_diffi_importance",
    "run_benchmark",
    "run_one_repeat",
    "save_benchmark_outputs",
    "select_top_features",
    "summarize_benchmark",
    "summarize_feature_scores",
    "RAW_THYROID_BINARY_COLUMNS",
    "RAW_THYROID_CONTINUOUS_COLUMNS",
    "RAW_THYROID_CATEGORICAL_COLUMNS",
    "RAW_THYROID_DATA_URLS",
    "RAW_THYROID_FEATURE_NAMES",
    "RAW_THYROID_LABEL_COLUMN",
    "assign_raw_thyroid_columns",
    "create_analysis_metadata",
    "convert_raw_thyroid_label",
    "detect_feature_types",
    "download_dataset",
    "download_raw_thyroid_dataset",
    "fit_mixed_type_preprocessor",
    "load_csv_dataset",
    "load_raw_thyroid_table",
    "prepare_benchmark_data",
    "prepare_raw_thyroid_data",
    "save_analysis_metadata",
    "save_dataframe",
    "train_test_split_mixed_type",
    "transform_mixed_type_data",
]
