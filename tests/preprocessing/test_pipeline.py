import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from preprocessing.pipeline import (
    get_preprocess_pipeline,
    train_pipeline,
    transform_inputs
)
from schema.data_schema import BinaryClassificationSchema


# Fixture to create a sample schema for testing
@pytest.fixture
def schema_provider():
    valid_schema = {
        "title": "test dataset",
        "description": "test dataset",
        "problemCategory": "binary_classification",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "id": {
            "name": "id",
            "description": "unique identifier."
        },
        "target": {
            "name": "target_field",
            "description":  "some target desc.",
            "allowedValues" :     ["A", "B"],
            "positiveClass": "A"
        },
        "predictors": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 10
            },
            {
                "name": "numeric_feature_2",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 1.1
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "allowedValues": ["A", "B", "C"]
            },
            {
                "name": "categorical_feature_2",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "allowedValues": ["X", "Y", "Z"]
            }
        ]
    }
    return BinaryClassificationSchema(valid_schema)



# Fixture to create a preprocessing config
@pytest.fixture
def preprocessing_config():
    config = {
        "numeric_transformers": {
            "missing_indicator": {},
            "mean_median_imputer": { "imputation_method": "mean" },
            "standard_scaler": {},
            "outlier_clipper": { "min_val": -4.0, "max_val": 4.0 }
        },
        "categorical_transformers": {
            "cat_most_frequent_imputer": { "threshold": 0.1 },
            "missing_tag_imputer": {
            "strategy": "constant",
            "fill_value": "missing"
            },
            "rare_label_encoder": {
            "tol": 0.03,
            "n_categories": 1,
            "replace_with": "__rare__"
            },
            "one_hot_encoder": { "handle_unknown": "ignore" }
        },
        "feature_selection_preprocessing": {
            "constant_feature_dropper": { "tol": 1 },
            "correlated_feature_dropper": { "threshold": 0.95 }
        }
    }
    return config

# Fixture to create a sample DataFrame for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, 30, 40, 50],
            "numeric_feature_2": [1.0, -2., 3, -4, 5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"]
        }
    )
    return data


def test_get_preprocess_pipeline(schema_provider, preprocessing_config):
    """
    Test if the get_preprocess_pipeline function returns a valid preprocessing pipeline
    for the given schema and input data.
    """
    try:
        pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    except Exception as exc:
        pytest.fail(f"Unexpected error while testing get_preprocess_pipeline: {str(exc)}")

    assert isinstance(pipeline, Pipeline), "Pipeline should be a sklearn Pipeline"


def test_train_pipeline(schema_provider, preprocessing_config, sample_data):
    """
    Test if the train_pipeline function returns a valid trained preprocessing pipeline
    for the given schema and input data.
    """
    try:
        pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
        trained_pipeline = train_pipeline(pipeline, sample_data)
    except Exception as exc:
        pytest.fail(f"Unexpected error while testing train_pipeline: {str(exc)}")

    assert isinstance(trained_pipeline, Pipeline), "Trained pipeline should be a sklearn Pipeline"


def test_transform_inputs(schema_provider, preprocessing_config, sample_data):
    """
    Test if the transform_inputs function returns a valid transformed data
    for the given schema and input data.
    """
    try:
        pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
        trained_pipeline = train_pipeline(pipeline, sample_data)
        transformed_data = transform_inputs(trained_pipeline, sample_data)
    except Exception as exc:
        pytest.fail(f"Unexpected error while testing transform_inputs: {str(exc)}")

    assert isinstance(transformed_data, pd.DataFrame), "Transformed data should be a pandas DataFrame"
    assert not transformed_data.empty, "Transformed data should not be empty"
    assert len(transformed_data) == len(sample_data), "Transformed data should have same number of rows as input data"


def test_numerical_features_are_present(schema_provider, preprocessing_config, sample_data):
    """
    Test if the numerical features are retained in the preprocessed data.
    """
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, sample_data)
    transformed_data = transform_inputs(trained_pipeline, sample_data)
    
    assert 'numeric_feature_1' in transformed_data.columns
    assert 'numeric_feature_2' in transformed_data.columns


def test_one_hot_encoded_categorical_features_are_present(
        schema_provider, preprocessing_config, sample_data):
    """
    Test if the numerical features are retained in the preprocessed data.
    """
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, sample_data)
    transformed_data = transform_inputs(trained_pipeline, sample_data)

    transformed_features = transformed_data.columns
    
    exp_features_per_cat_feature = [
        ("categorical_feature_1", 2), ("categorical_feature_2", 4)]

    for cat_feat, num_feat in exp_features_per_cat_feature:
        returned_feats = set([f for f in transformed_features
                          if f.startswith(cat_feat)])
        print(returned_feats)
        assert len(returned_feats) == num_feat


def test_pipeline_with_missing_values(schema_provider, preprocessing_config):
    """
    Test the pipeline with a DataFrame that has missing values in the numeric
    and categorical features. This test will help you ensure that the imputation
    steps are working as expected.
    """
    data_with_missing_values = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, np.nan, 30, np.nan, 50],
            "numeric_feature_2": [1.1, 2.2, np.nan, 4.4, 5.5],
            "categorical_feature_1": ["A", "B", None, "A", "B"],
            "categorical_feature_2": ["X", "Y", "Z", None, "Y"],
            "target_field": ["A", "B", "A", "B", "A"]
        }
    )
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, data_with_missing_values)
    transformed_data = transform_inputs(trained_pipeline, data_with_missing_values)

    assert transformed_data is not None, "Transformed data should not be None"
    assert not transformed_data.isna().any().any(), "Transformed data should not have any NaN values"


def test_pipeline_with_outliers(schema_provider, preprocessing_config):
    """
    Test the pipeline with a DataFrame that has outlier values in the numeric features.
    This test will help you ensure that the outlier clipping step is working as expected.
    """
    data_with_outliers = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, 30, 40, 1000],
            "numeric_feature_2": [1.1, -2.2, 3.3, -4.4, 5.5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["X", "Y", "Z", "X", "Y"],
            "target_field": ["A", "B", "A", "B", "A"]
        }
    )
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, data_with_outliers)
    transformed_data = transform_inputs(trained_pipeline, data_with_outliers)

    assert transformed_data is not None, "Preprocessed data should not be None"
    assert transformed_data["numeric_feature_1"].min() >= -4.0
    assert transformed_data["numeric_feature_1"].max() <= 4.0

    # check with tighter bounds on values
    preprocessing_config["numeric_transformers"]["outlier_clipper"] = {
         "min_val": -3.0, "max_val": 3.0
    }
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, data_with_outliers)
    transformed_data = transform_inputs(trained_pipeline, data_with_outliers)

    assert transformed_data is not None, "Preprocessed data should not be None"
    assert transformed_data["numeric_feature_1"].min() >= -3.0
    assert transformed_data["numeric_feature_1"].max() <= 3.0


def test_pipeline_with_rare_labels(schema_provider, preprocessing_config):
    """
    Test the pipeline with a DataFrame that has rare labelsin the categorical features. This test will help you ensure that the rare label encoding step is working as expected.
    """
    # labels which occur less than 20% of the time are considered rare, they will be grouped 
    preprocessing_config["categorical_transformers"]["rare_label_encoder"]["tol"] = 0.2

    data_with_rare_labels = pd.DataFrame(
        {
            "id": range(1, 11),
            "numeric_feature_1": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
            "numeric_feature_2": [1.1, 2.2, 3.3, 4.4, 5.5, 1.1, 2.2, 3.3, 4.4, 5.5],
            "categorical_feature_1": ["A", "A", "A", "A", "A", "A", "A", "A", "B", "C"],
            "categorical_feature_2": ["X", "Y", "Z", "X", "X", "X", "Y", "Z", "X", "Z"],
            "target_field": ["A", "B", "A", "B", "A", "A", "B", "A", "B", "A"]
        }
    )
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, data_with_rare_labels)
    transformed_data = transform_inputs(trained_pipeline, data_with_rare_labels)

    assert transformed_data is not None, "Preprocessed data should not be None"
    assert 'categorical_feature_1_B' not in transformed_data.columns
    assert 'categorical_feature_1_C' not in transformed_data.columns
    assert ('categorical_feature_1_A' in transformed_data.columns or
        'categorical_feature_1___rare__' in transformed_data.columns)
    
    
def test_pipeline_with_empty_dataframe(schema_provider, preprocessing_config):
    """
    Test the pipeline with an empty DataFrame. This test will help you ensure
    that the pipeline throws a ValueError for this case.
    """
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
        trained_pipeline = train_pipeline(pipeline, empty_data)
        _ = transform_inputs(trained_pipeline, empty_data)


def test_pipeline_with_invalid_dataframe(schema_provider, preprocessing_config):
    """
    Test the pipeline with an empty DataFrame. This test will help you ensure
    that the pipeline throws a TypeError for this case.
    """
    with pytest.raises(TypeError):
        pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
        _ = train_pipeline(pipeline, "invalid")

       
def test_pipeline_with_only_numeric_features(schema_provider, preprocessing_config):
    """
    Test the pipeline with a DataFrame that has only numeric features.
    This test will help you ensure that the pipeline can handle the case when there
    are no categorical features to preprocess.
    """
    numeric_only_data = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, 30, 40, 50],
            "numeric_feature_2": [1.0, -2., 3, -4, 5]
        }
    )
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    trained_pipeline = train_pipeline(pipeline, numeric_only_data)
    transformed_data = transform_inputs(trained_pipeline, numeric_only_data)

    assert transformed_data is not None, "Preprocessed data should not be None"
    assert 'numeric_feature_1' in transformed_data.columns
    assert 'numeric_feature_2' in transformed_data.columns
    


def test_pipeline_with_only_categorical_features(schema_provider, preprocessing_config):
    """
    Test the pipeline with a DataFrame that has only categorical features.
    This test will help you ensure that the pipeline can handle the case when there
    are no numeric features to preprocess.
    """
    pipeline = get_preprocess_pipeline(schema_provider, preprocessing_config)
    categorical_only_data = pd.DataFrame(
        {
            "id": range(1, 6),
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
        }
    )
    trained_pipeline = train_pipeline(pipeline, categorical_only_data)
    transformed_data = transform_inputs(trained_pipeline, categorical_only_data)

    assert transformed_data is not None, "Preprocessed data should not be None"
