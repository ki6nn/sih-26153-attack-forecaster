"""Data ingestion, validation, feature engineering, and temporal windowing pipeline."""

from src.pipeline.features import (
    CICIDS2017_FEATURES,
    UNSW_NB15_FEATURES,
    encode_categorical_features,
    extract_features,
    get_feature_columns,
    map_stage_labels,
)
from src.pipeline.ingestion import (
    load_csv_chunks,
    load_dataset,
    normalize_column_names,
    parse_timestamps,
    record_source_provenance,
)
from src.pipeline.preprocessor import (
    NetworkDataPreprocessor,
    ResearchPipelineData,
    prepare_research_pipeline,
    preprocess,
)
from src.pipeline.validation import (
    clean_missing_and_infinite,
    detect_duplicate_records,
    detect_missing_and_infinite,
    generate_data_quality_report,
    validate_labels,
    validate_required_columns,
)
from src.pipeline.windowing import (
    chronological_split,
    create_network_windows,
    validate_and_extract_durations,
    verify_temporal_isolation,
)

__all__ = [
    "load_csv_chunks",
    "normalize_column_names",
    "parse_timestamps",
    "record_source_provenance",
    "load_dataset",
    "validate_required_columns",
    "detect_missing_and_infinite",
    "clean_missing_and_infinite",
    "validate_labels",
    "detect_duplicate_records",
    "generate_data_quality_report",
    "CICIDS2017_FEATURES",
    "UNSW_NB15_FEATURES",
    "get_feature_columns",
    "extract_features",
    "map_stage_labels",
    "encode_categorical_features",
    "NetworkDataPreprocessor",
    "ResearchPipelineData",
    "prepare_research_pipeline",
    "preprocess",
    "chronological_split",
    "create_network_windows",
    "validate_and_extract_durations",
    "verify_temporal_isolation",
]
