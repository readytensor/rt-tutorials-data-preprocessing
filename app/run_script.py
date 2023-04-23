
import sys
from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter

from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, get_fitted_binary_target_encoder, \
    save_pipeline, save_label_encoder
from data_management.data_utils import read_json_in_directory, read_csv_in_directory
import paths


def main():

    # load the json file schema into a dictionary and use it to instantiate the schema provider
    schema_dict = read_json_in_directory(file_dir_path=paths.SCHEMA_DIR)
    data_schema = BinaryClassificationSchema(schema_dict)

    # load train data
    train_data = read_csv_in_directory(file_dir_path=paths.TRAIN_DIR)

    # create preprocessing pipeline, transform data, and save pipeline
    preprocess_pipeline = get_preprocess_pipeline(data_schema = data_schema)
    transformed_data = preprocess_pipeline.fit_transform(train_data)
    save_pipeline(pipeline=preprocess_pipeline, file_path_and_name = paths.PREPROCESSOR_FILE_PATH)
    
    # create fitted label_encoder, transform labels, and save encoder
    label_encoder = get_fitted_binary_target_encoder(
        target_field=data_schema.target_field,
        allowed_values=data_schema.allowed_target_values,
        positive_class=data_schema.positive_class)
    transformed_labels = label_encoder.transform(train_data[[data_schema.target_field]])
    save_label_encoder(label_encoder=label_encoder, file_path_and_name = paths.LABEL_ENCODER_FILE_PATH)

    # handle class imbalance using SMOTE
    smote = SMOTE()
    balanced_data, balanced_labels = smote.fit_resample(transformed_data, transformed_labels)

    print("*"*60)
    print("Balanced features:")
    print(pd.DataFrame(balanced_data).head(5))
    print("*"*60)

    # Print original and balanced class counts
    print("Original class counts:", Counter(transformed_labels.values.ravel()))
    print("Balanced class counts:", Counter(balanced_labels.values.ravel()))


if __name__ == "__main__":
    main()