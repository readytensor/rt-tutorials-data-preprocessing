from collections import Counter
from utils import (
    load_data_schema, 
    read_csv_in_directory, 
    read_json_as_dict, 
    split_train_val, 
    get_validation_percentage
)
from data_management.preprocess import (
    create_pipeline_and_label_encoder,
    train_pipeline_and_label_encoder,
    save_pipeline_and_label_encoder,
    transform_data,
    handle_class_imbalance
)
from config import paths
from utils import set_seeds



def run_training():
    """
    Run the training process for the binary classification model.
    """
    # set seeds
    set_seeds(seed_value=0)

    # load the json file schema into a dictionary and use it to instantiate the schema provider
    data_schema = load_data_schema(paths.SCHEMA_DIR)

    # load train data
    train_data = read_csv_in_directory(file_dir_path=paths.TRAIN_DIR)

    # load the model configuration
    model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)

    # get the validation percentage from the configuration
    val_pct = get_validation_percentage(model_config)

    # split train data into training and validation sets
    train_split, val_split = split_train_val(train_data, val_pct=val_pct)

    # create preprocessing pipeline and target encoder
    preprocess_pipeline, label_encoder = create_pipeline_and_label_encoder(model_config, data_schema)

    # train pipeline and label encoder
    transformed_data, transformed_labels = train_pipeline_and_label_encoder( \
        preprocess_pipeline, label_encoder, train_split, data_schema)

    # handle class imbalance using SMOTE
    balanced_data, balanced_labels = handle_class_imbalance(transformed_data, transformed_labels, random_state=0)

    # save pipeline and label encoder
    save_pipeline_and_label_encoder(preprocess_pipeline, label_encoder,
           paths.PIPELINE_FILE_PATH, paths.LABEL_ENCODER_FILE_PATH)
    
    print("*" * 60)
    print("Original data shape:", train_data.shape)
    print("Original train and valid split shapes:", train_split.shape, val_split.shape)
    print("Processed train X/y shapes:", transformed_data.shape, transformed_labels.shape)
    print("Balanced train X/y shapes:", balanced_data.shape, balanced_labels.shape)
    print("*" * 60)
    print("Balanced train data:")
    print(balanced_data.head(5))
    print("*" * 60)

    # Print original and balanced class counts
    print("Original train data class counts:", Counter(transformed_labels.values.ravel()))
    print("Balanced train data class counts:", Counter(balanced_labels.values.ravel()))

    # transform validation data
    transformed_val_data, transformed_val_labels, _ = transform_data(preprocess_pipeline, label_encoder, val_split, data_schema)
    
    print("*" * 60)
    print("Processed validation data:")
    print(transformed_val_data.head())
    print("Processed validation X/y shapes:", transformed_val_data.shape, transformed_val_labels.shape)
    print("Processed validation data class counts:", Counter(transformed_val_labels.values.ravel()))



if __name__ == "__main__": 
    run_training()