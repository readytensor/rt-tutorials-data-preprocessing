import pandas as pd
import sys
from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, save_preprocessor_and_lbl_encoder, get_label_encoder
from data_management.data_reader import read_data

# path to the schema file
SCHEMA_FPATH = "./inputs/titanic_schema.json"
TRAIN_DATA_FPATH = "./inputs/titanic_train.csv"
TEST_DATA_FPATH = "./inputs/titanic_test.csv"
MODEL_ARTIFACTS_PATH = "./outputs/artifacts/"


def main():     

    # instantiate schem provider which loads the schema
    data_schema = BinaryClassificationSchema(SCHEMA_FPATH)

    # load train data
    train_data = read_data(data_path=TRAIN_DATA_FPATH, data_schema=data_schema)
    # print(train_data.head())

    # preprocessing
    preprocess_pipeline = get_preprocess_pipeline(data_schema)
    transformed_data = preprocess_pipeline.fit_transform(train_data.drop(data_schema.id_field, axis=1))  
    label_encoder = get_label_encoder(data_schema)
    transformed_labels = label_encoder.fit_transform(train_data[[data_schema.target_field]])

    print("*"*60)
    print("Transformed features:")
    print(transformed_data.head(10))
    print("*"*60)
    print("Transformed labels:")
    print(transformed_labels[:10])
    print("*"*60)

    # save preprocessor
    save_preprocessor_and_lbl_encoder(preprocess_pipeline, label_encoder, MODEL_ARTIFACTS_PATH)


if __name__ == "__main__": 
    main()