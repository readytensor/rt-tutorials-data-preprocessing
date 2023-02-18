import pandas as pd, numpy as np
from data_management.schema_provider import BinaryClassificationSchema



def read_data(data_path: str, data_schema: BinaryClassificationSchema) -> pd.DataFrame:
    """
    Reads data and casts fields to the expected type as per the provided schema.

    Args:
        data_path (str): The path to the training data file.
        bc_schema (BinaryClassificationSchema): The schema provider object.

    Returns:
        pd.DataFrame: The training data as a pandas DataFrame with casted field types.
    """

    data = pd.read_csv(data_path)

    # Cast id field to string
    data[data_schema.id_field] = data[data_schema.id_field].astype(str)
    
    # Cast target field to string
    if data_schema.target_field in data.columns:
        data[data_schema.target_field] = data[data_schema.target_field].astype(str)
    
    # Cast categorical features to string
    for c in data_schema.categorical_features:
        if c in data.columns:
            data[c] = data[c].astype(str)
    
    # Cast numeric features to float
    for c in data_schema.numeric_features:
        if c in data.columns:
            data[c] = data[c].astype(np.float32)
    
    return data