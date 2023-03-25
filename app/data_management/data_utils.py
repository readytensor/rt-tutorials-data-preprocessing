import os
import json
import pandas as pd

def read_json_in_directory(file_dir_path: str) -> dict:
    """
    Reads a JSON file in the given directory path as a dictionary and returns the dictionary.
    
    Args:
    - file_dir_path (str): The path to the directory containing the JSON file.
    
    Returns:
    - dict: The contents of the JSON file as a dictionary.
    
    Raises:
    - ValueError: If no JSON file is found in the directory or if multiple JSON files are found in the directory.
    """
    json_files = [file for file in os.listdir(file_dir_path) if file.endswith('.json')]
    
    if not json_files:
        raise ValueError('No JSON file found in directory')
    
    if len(json_files) > 1:
        raise ValueError('Multiple JSON files found in directory')
    
    json_file_path = os.path.join(file_dir_path, json_files[0])
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data


def read_csv_in_directory(file_dir_path: str) -> pd.DataFrame:
    """
    Reads a CSV file in the given directory path as a pandas dataframe and returns the dataframe.
    
    Args:
    - file_dir_path (str): The path to the directory containing the CSV file.
    
    Returns:
    - pd.DataFrame: The pandas dataframe containing the data from the CSV file.
    
    Raises:
    - ValueError: If no CSV file is found in the directory or if multiple CSV files are found in the directory.
    """
    csv_files = [file for file in os.listdir(file_dir_path) if file.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f'No CSV file found in directory {file_dir_path}')
    
    if len(csv_files) > 1:
        raise ValueError(f'Multiple CSV files found in directory {file_dir_path}.')
    
    csv_file_path = os.path.join(file_dir_path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    return df


def save_csv_to_directory(df: pd.DataFrame, file_dir_path: str, file_name: str) -> None:
    """
    Saves a pandas dataframe to a CSV file in the given directory path.
    
    Args:
    - df (pd.DataFrame): The pandas dataframe to be saved.
    - file_dir_path (str): The path to the directory where the CSV file should be saved.
    - file_name (str): The name of the CSV file.
    
    Returns:
    - None
    """
    csv_file_path = os.path.join(file_dir_path, file_name)
    df.to_csv(csv_file_path, index=False, float_format='%.4f')