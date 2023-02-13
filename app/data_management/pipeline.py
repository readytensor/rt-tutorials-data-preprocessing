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
LABEL_ENCODER_FNAME = "label_encoder.save"

def get_preprocess_pipeline(bc_schema):
    pipeline = Pipeline(
        [
            (
                # keep only the columns that were defined in the schema
                "column_selector",
                preprocessors.ColumnSelector(columns=bc_schema.features),
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
            )
        ]
    )
    return pipeline

def get_label_encoder(bc_schema): 
    return preprocessors.CustomLabelBinarizer(
        target_field=bc_schema.target_field,
        target_class=bc_schema.target_class,
    )

def get_class_names(label_encoder):
    class_names = label_encoder.given_classes
    return class_names


def save_preprocessor_and_lbl_encoder(preprocess_pipe, label_encoder, file_path):
    joblib.dump(preprocess_pipe, os.path.join(file_path, PREPROCESSOR_FNAME))
    joblib.dump(label_encoder, os.path.join(file_path, LABEL_ENCODER_FNAME))
    return


def load_preprocessor_and_lbl_encoder(file_path):
    preprocess_pipe = joblib.load(os.path.join(file_path, PREPROCESSOR_FNAME))
    label_encoder = joblib.load(os.path.join(file_path, LABEL_ENCODER_FNAME))
    return preprocess_pipe, label_encoder