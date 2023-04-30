from typing import Tuple, Dict, Any
import pandas as pd
from collections import Counter
from config import paths
from sklearn.pipeline import Pipeline
from utils import (
    set_seeds,
    load_data_schema, 
    read_csv_in_directory, 
    read_json_as_dict, 
    get_validation_percentage
)
from data_management.schema_provider import BinaryClassificationSchema
from data_management.preprocess import (
    create_pipeline_and_label_encoder,
    train_pipeline_and_label_encoder,
    save_pipeline_and_label_encoder,
    transform_data,
    split_train_val, 
    handle_class_imbalance
)
from data_management.label_encoder import CustomLabelBinarizer


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


def preprocess_and_balance_data(
        model_config: Dict[str, Any],
        data_schema: BinaryClassificationSchema,
        train_split: pd.DataFrame) -> Tuple[Pipeline, CustomLabelBinarizer, pd.DataFrame, pd.Series]:
    """
    Preprocess and balance the data using the provided model configuration and data schema.

    Args:
        model_config: A dictionary containing the model configuration.
        data_schema: A dictionary containing the data schema.
        train_split: A pandas DataFrame containing the data split.

    Returns:
        A tuple containing the preprocessed pipeline, label encoder, balanced data, and balanced labels.
    """
    preprocess_pipeline, label_encoder = \
        create_pipeline_and_label_encoder(model_config, data_schema)
    transformed_data, transformed_labels = train_pipeline_and_label_encoder(
        preprocess_pipeline, label_encoder, train_split, data_schema)
    balanced_data, balanced_labels = \
        handle_class_imbalance(transformed_data, transformed_labels, random_state=0)
    
    return preprocess_pipeline, label_encoder, balanced_data, balanced_labels


def run_training():
    """
    Run the training process for the binary classification model.
    """
    set_seeds(seed_value=0)

    data_schema = load_data_schema(paths.SCHEMA_DIR)

    model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)
    val_pct = get_validation_percentage(model_config)

    train_split, val_split = load_and_split_data(val_pct)

    preprocess_pipeline, label_encoder, balanced_train_data, balanced_train_labels = \
        preprocess_and_balance_data(model_config, data_schema, train_split)
    print("*" * 60)
    
    transformed_val_data, transformed_val_labels, _ = \
        transform_data(preprocess_pipeline, label_encoder, val_split, data_schema)
    
    save_pipeline_and_label_encoder(preprocess_pipeline, label_encoder,
           paths.PIPELINE_FILE_PATH, paths.LABEL_ENCODER_FILE_PATH)
    
    print("*" * 60)
    print("Original train and valid split shapes:", train_split.shape, val_split.shape)
    print("Transformed and balanced train X/y shapes:", balanced_train_data.shape, balanced_train_labels.shape)
    print("Balanced train data class counts:", Counter(balanced_train_labels.values.ravel()))
    print("Processed validation X/y shapes:", transformed_val_data.shape, transformed_val_labels.shape)

    print("Transformed and balanced train data:")
    print(balanced_train_data.head())

    print("Transformed and balanced validation data:")
    print(transformed_val_data.head())


if __name__ == "__main__":
    run_training()