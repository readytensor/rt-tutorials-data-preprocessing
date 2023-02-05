import pandas as pd

from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, save_preprocessor
from data_management.data_reader import read_data

# path to the schema file
SCHEMA_FPATH = "./inputs/titanic_schema.json"
TRAIN_DATA_FPATH = "./inputs/titanic_train.csv"
TEST_DATA_FPATH = "./inputs/titanic_test.csv"
MODEL_ARTIFACTS_PATH = "./outputs/artifacts/"


def main():     

    # instantiate schem provider which loads the schema
    bc_schema = BinaryClassificationSchema(SCHEMA_FPATH)

    # load train data
    train_data = read_data(data_path=TRAIN_DATA_FPATH, bc_schema=bc_schema)
    # print(train_data.head())

    # preprocessing
    preprocess_pipeline = get_preprocess_pipeline(bc_schema)
    transformed_data = preprocess_pipeline.fit_transform(train_data.drop(bc_schema.id_field, axis=1))
    print(transformed_data.head(10))

    # save preprocessor
    save_preprocessor(preprocess_pipeline, MODEL_ARTIFACTS_PATH)


if __name__ == "__main__": 
    main()