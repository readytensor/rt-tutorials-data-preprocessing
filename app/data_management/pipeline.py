import os 
import joblib 
from sklearn.pipeline import Pipeline
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from sklearn.preprocessing import StandardScaler

import data_management.preprocessors as preprocessors


PREPROCESSOR_FNAME = "preprocessor.save"

def get_preprocess_pipeline(bc_schema):
    pipeline = Pipeline(
        [
            (
                # keep only the columns that were defined in the schema
                "column_selector",
                preprocessors.ColumnSelector(columns=bc_schema.all_fields),
            ),
            (
                # add missing indicator for nas in numerical features
                "missing_indicator_numeric",
                AddMissingIndicator(variables=bc_schema.numeric_features),
            ),
            (
                # impute numerical na with the mean
                "mean_imputer_numeric",
                MeanMedianImputer(imputation_method="mean",variables=bc_schema.numeric_features),
            ),
            (
                # standard scale the numerical features
                "standard_scaler",
                SklearnTransformerWrapper(
                    StandardScaler(), variables=bc_schema.numeric_features
                ),
            ),
            (
                # clip the standardized values to +/- 4.0, corresponding to +/- 4 std dev.
                "outlier_value_clipper",
                preprocessors.ValueClipper(
                    fields_to_clip=bc_schema.numeric_features,
                    min_val=-4.0,  # - 4 std dev
                    max_val=4.0,  # + 4 std dev
                ),
            ),            
            (
                # impute categorical na with most frequent category, when missing values are rare (under a threshold)
                "cat_most_frequent_imputer",
                preprocessors.MostFrequentImputer(
                    cat_vars=bc_schema.categorical_features,
                    threshold=0.1,
                ),
            ),
            (
                 # impute categorical na with string 'missing'
                "cat_imputer_with_missing_tag",
                CategoricalImputer(
                    imputation_method="missing",
                    variables=bc_schema.categorical_features
                ),
            ),
            (
                "rare_label_encoder",
                RareLabelEncoder(
                    tol=0.03,
                    n_categories=1,
                    variables=bc_schema.categorical_features,
                ),
            ),
            (
                # one-hot encode cat vars
                "one_hot_encoder",
                preprocessors.OneHotEncoderMultipleCols(
                    ohe_columns=bc_schema.categorical_features,
                ),
            ),
            (
                # drop the original cat vars, we keep the ohe variables
                "cat_var_dropper",
                preprocessors.ColumnSelector(
                    columns=bc_schema.categorical_features,
                    selector_type="drop"
                ),
            ),
            (
                "label_binarizer",
                preprocessors.CustomLabelBinarizer(
                    target_field=bc_schema.target_field,
                    target_class=bc_schema.target_class,
                ),
            )
        ]
    )
    return pipeline


def get_class_names(pipeline):
    label_binarizer = pipeline["label_binarizer"]
    class_names = label_binarizer.given_classes
    return class_names


def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    joblib.dump(preprocess_pipe, file_path_and_name)
    return


def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    preprocess_pipe = joblib.load(file_path_and_name)
    return preprocess_pipe