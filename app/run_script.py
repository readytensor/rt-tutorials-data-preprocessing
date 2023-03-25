from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, save_preprocessor_and_lbl_encoder, get_label_encoder
from data_management.data_utils import read_json_in_directory, read_csv_in_directory
import paths


def main():

    # load the json file schema into a dictionary and use it to instantiate the schema provider
    schema_dict = read_json_in_directory(file_dir_path=paths.SCHEMA_DIR)
    data_schema = BinaryClassificationSchema(schema_dict)

    # load train data
    train_data = read_csv_in_directory(file_dir_path=paths.TRAIN_DIR)

    # create preprocessing pipeline and label encoder
    preprocess_pipeline = get_preprocess_pipeline(data_schema)
    label_encoder = get_label_encoder(data_schema)

    # fit preprocessing pipeline and transform data, fit label_encoder and transform labels
    transformed_data = preprocess_pipeline.fit_transform(train_data.drop(data_schema.id_field, axis=1))
    transformed_labels = label_encoder.fit_transform(train_data[[data_schema.target_field]])

    print("*"*60)
    print("Transformed features:")
    print(transformed_data.head(10))
    print("*"*60)
    print("Transformed labels:")
    print(transformed_labels[:10])
    print("*"*60)

    # save preprocessing pipeline and label encoder
    save_preprocessor_and_lbl_encoder(preprocess_pipeline, label_encoder, paths.MODEL_ARTIFACTS_PATH)


if __name__ == "__main__":
    main()