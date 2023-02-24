from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, save_preprocessor_and_lbl_encoder, get_label_encoder
from data_management.data_reader import read_data
import paths


def main():

    # instantiate schem provider which loads the schema
    data_schema = BinaryClassificationSchema(paths.SCHEMA_FPATH)

    # load train data
    train_data = read_data(data_path=paths.TRAIN_DATA_FPATH, data_schema=data_schema)

    # create preprocessing pipeline and label encoder
    preprocess_pipeline = get_preprocess_pipeline(data_schema)
    label_encoder = get_label_encoder(data_schema)

    # fit preprocessing pipeline and transform data and labels
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