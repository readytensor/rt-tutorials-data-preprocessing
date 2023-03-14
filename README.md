## Introduction

This repository demonstrates how to create generalized data preprocessing pipeline for your machine learning algorithm implementation. The repo provides examples of how you can avoid hard-coding your implementation to a specific dataset, which can make it easier to apply your algorithms to new datasets in the future.

## Repository Contents

The `app/` folder in the repository contains the following key folders/sub-folders:

- `data_management/` will all files related to handling and preprocessing data.
- `inputs/` contains the input files related to the _titanic_ dataset.
- `model/` is a folder to save model artifacts and other assets specific to the trained model. Within this folder:
  - `artifacts/` is location to save model artifacts (i.e. the saved model including the trained preprocessing pipeline)
- `paths.py`: script contains variables which represent various paths to be used in the repository.
- `run_script.py`: an example script that shows the usage of the preprocessing pipeline and processed outputs.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Place the train data file in csv format in the path `./app/inputs/`
- Place the schema file in JSON format in the path `./app/inputs/`. The schema conforms to Ready Tensor specification for the **Binary Classification-Base** category.
- Update the file paths in the `run_script.py` file in `./app/` and run the script as follows.

The script will print top 10 rows of the transformed data and also save the preprocessing pipeline in the path `./app/outputs/artifacts/`.

## Requirements

The code requires Python 3 and the following libraries:

```makefile
json==2.0.9
pandas==1.5.2
numpy==1.20.3
scikit-learn==1.0
feature-engine==1.2.0
```

These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```
