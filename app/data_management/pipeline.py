from typing import List
import joblib
from sklearn.pipeline import Pipeline
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler, LabelEncoder
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

from data_management import preprocessors


def get_preprocess_pipeline(data_schema):
    """
    Create a preprocessor pipeline to transform data as defined by data_schema.
    """
    column_selector = preprocessors.ColumnSelector(columns=data_schema.all_fields)
    string_caster = preprocessors.TypeCaster(
        vars=data_schema.categorical_features + [data_schema.id_field, data_schema.target_field],
        cast_type=str
    )
    float_caster = preprocessors.TypeCaster(
        vars=data_schema.numeric_features,
        cast_type=float
    )
    missing_indicator_numeric = AddMissingIndicator(variables=data_schema.numeric_features)
    mean_imputer_numeric = MeanMedianImputer(imputation_method="mean", variables=data_schema.numeric_features)
    standard_scaler = SklearnTransformerWrapper(StandardScaler(), variables=data_schema.numeric_features)
    outlier_value_clipper = preprocessors.ValueClipper(
        fields_to_clip=data_schema.numeric_features,
        min_val=-4.0,
        max_val=4.0
    )
    cat_most_frequent_imputer = preprocessors.MostFrequentImputer(
        cat_vars=data_schema.categorical_features,
        threshold=0.1
    )
    cat_imputer_with_missing_tag = preprocessors.FeatureEngineCategoricalTransformerWrapper(
        transformer=CategoricalImputer,
        cat_vars=data_schema.categorical_features,
        imputation_method="missing"
    )
    rare_label_encoder = preprocessors.FeatureEngineCategoricalTransformerWrapper(
        transformer=RareLabelEncoder,
        cat_vars=data_schema.categorical_features,
        tol=0.03,
        n_categories=1
    )
    one_hot_encoder = preprocessors.OneHotEncoderMultipleCols(ohe_columns=data_schema.categorical_features)
    cat_var_dropper = preprocessors.ColumnSelector(
        columns=data_schema.categorical_features,
        selector_type="drop"
    )
    
    pipeline = Pipeline([
        ("column_selector", column_selector),
        ("string_caster", string_caster),
        ("float_caster", float_caster),
        ("missing_indicator_numeric", missing_indicator_numeric),
        ("mean_imputer_numeric", mean_imputer_numeric),
        ("standard_scaler", standard_scaler),
        ("outlier_value_clipper", outlier_value_clipper),
        ("cat_most_frequent_imputer", cat_most_frequent_imputer),
        ("cat_imputer_with_missing_tag", cat_imputer_with_missing_tag),
        ("rare_label_encoder", rare_label_encoder),
        ("one_hot_encoder", one_hot_encoder),
        ("cat_var_dropper", cat_var_dropper)
    ])
    
    return pipeline


def get_fitted_binary_target_encoder(target_field:str, allowed_values: List[str], positive_class: str) -> LabelEncoder:
    """Create a LabelEncoder based on the data_schema.

    The positive class will be encoded as 1, and the negative class will be encoded as 0.

    Args:
        target_field: Name of the target field.
        allowed_values: A list of allowed target variable values.
        positive_class: The target value representing the positive class.

    Returns:
        A SciKit-Learn LabelEncoder instance.
    """

    # Create a LabelEncoder instance and fit it with the desired class order
    encoder = preprocessors.CustomLabelBinarizer(
        target_field=target_field,
        allowed_values=allowed_values,
        positive_class=positive_class
    )
    return encoder


def save_pipeline(pipeline: Pipeline, file_path_and_name: str) -> None:
    """Save the fitted pipeline to a pickle file.

    Args:
        pipeline (Pipeline): The fitted pipeline to be saved.
        file_path_and_name (str): The path where the pipeline should be saved.
    """
    joblib.dump(pipeline, file_path_and_name)


def save_label_encoder(label_encoder: LabelEncoder, file_path_and_name: str) -> None:
    """Save a fitted label encoder to a file using joblib.

    Args:
        label_encoder: A fitted LabelEncoder instance.
        file_path_and_name (str): The filepath to save the LabelEncoder to.
    """
    joblib.dump(label_encoder, file_path_and_name)


def load_pipeline(path: str) -> Pipeline:
    """Load the fitted pipeline from the given path.

    Args:
        path: Path to the saved pipeline.

    Returns:
        Fitted pipeline.
    """
    return joblib.load(path)


def load_label_encoder(path: str) -> LabelEncoder:
    """Load the fitted label encoder from the given path.

    Args:
        path: Path to the saved label encoder.

    Returns:
        Fitted label encoder.
    """
    return joblib.load(path)