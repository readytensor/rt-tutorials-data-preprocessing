from typing import Tuple
from collections import Counter
import pandas as pd

from config import paths
from schema.data_schema import load_json_data_schema, save_schema
from config import paths
from utils import (
    set_seeds,
    read_csv_in_directory,
    split_train_val,
    read_json_as_dict
)
from preprocessing.preprocess import (
    train_pipeline_and_target_encoder,
    transform_data,
    save_pipeline_and_target_encoder,
    handle_class_imbalance
)

def load_and_split_data(val_pct: float) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the data into training and validation sets.

    Args:
        val_pct: The percentage of the data to be used for validation.

    Returns:
        A tuple containing the data schema, training split, and validation split.
    """
    train_data = read_csv_in_directory(file_dir_path=paths.TRAIN_DIR)
    train_split, val_split = split_train_val(train_data, val_pct=val_pct)
    return train_split, val_split



def check_preprocessing():
    """Reads the binary classification schema."""
    set_seeds(seed_value=0)

    # load and save schema
    data_schema = load_json_data_schema(paths.INPUT_SCHEMA_DIR)
    save_schema(schema=data_schema, output_path=paths.SAVED_SCHEMA_PATH)

    # load model config, and perform train/validation split
    model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)
    train_split, val_split = load_and_split_data(
        val_pct=model_config.get("validation_split", 0.2))

    # fit and transform using pipeline and target encoder, then save them
    pipeline, target_encoder = train_pipeline_and_target_encoder(
        data_schema, train_split)
    transformed_train_inputs, transformed_train_targets = transform_data(
        pipeline, target_encoder, train_split)
    transformed_val_inputs, transformed_val_labels = transform_data(
        pipeline, target_encoder, val_split)
    balanced_train_inputs, balanced_train_labels = \
        handle_class_imbalance(transformed_train_inputs,
                               transformed_train_targets)

    # visualize inspect processed inputs and targets
    print("*" * 60)
    print("Original train and valid split shapes:", train_split.shape, val_split.shape)
    print("Transformed and balanced train X/y shapes:", balanced_train_inputs.shape, balanced_train_labels.shape)
    print("Balanced train data class counts:", Counter(balanced_train_labels.values.ravel()))
    print("Processed validation X/y shapes:", transformed_val_inputs.shape, transformed_val_labels.shape)

    print("Transformed and balanced train data:")
    print(balanced_train_inputs.head())

    print("Transformed and balanced validation data:")
    print(transformed_val_inputs.head())


    save_pipeline_and_target_encoder(
        pipeline, target_encoder,
        paths.PIPELINE_FILE_PATH,
        paths.TARGET_ENCODER_FILE_PATH)


if __name__ == "__main__":
    check_preprocessing()
