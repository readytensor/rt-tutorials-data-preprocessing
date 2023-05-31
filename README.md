## Introduction

This repository is part of a comprehensive tutorial series on [Ready Tensor](https://docs.readytensor.ai/category/creating-adaptable-ml-models) aimed at building adaptable machine learning models. In this repository, we add an adaptable data preprocessing pipeline for a binary classification machine learning algorithm implementation. We also have a target encoder to handle the targets.

The repo provides examples of how you can avoid hard-coding your implementation to a specific dataset, which can make it easier to apply your algorithms to new datasets in the future.

## Project Structure

```bash
binary_class_project/
├── examples/
│   ├── titanic_schema.json
│   ├── titanic_train.csv
│   └── titanic_test.csv
├── inputs/
│   ├── data/
│   │   ├── testing/
│   │   └── training/
│   └── schema/
├── model/
│   └── artifacts/
├── outputs/
│   ├── errors/
│   ├── hpt_outputs/
│   └── predictions/
├── src/
│   ├── config/
│   │   ├── model_config.json
│   │   ├── paths.py
│   │   └── preprocessing.json
│   ├── data_models/
│   ├── hyperparameter_tuning/
│   ├── prediction/
│   ├── preprocessing/
│   │   ├── custom_transformers.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   └── target_encoder.py
│   ├── schema/
│   │   └── data_schema.py
│   ├── xai/
│   ├── check_preprocessing.py
│   └── utils.py
├── tests/
│   ├── integration_tests/
│   ├── performance_tests/
│   └── unit_tests/
│       ├── <mirrors /src structure>
│       └── ...
├── tmp/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
└── requirements-test.txt
```

- **`/examples`**: This directory contains example files for the titanic dataset. Three files are included: `titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`/inputs`**: This directory contains all the input files for your project, including the data and schema files. The data is further divided into testing and training subsets.
- **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.
- **`/outputs`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results. Note that model artifacts should not be saved in this directory. Instead, they should be saved in the `/model/artifacts` directory.
- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories such as `config` for configuration files, `data_models` for data models for input validation, `hyperparameter_tuning` for hyperparameter-tuning (HPT) related files, `prediction` for prediction model scripts, `preprocessing` for data preprocessing scripts, `schema` for schema scripts, and `xai` for explainable AI scripts. The following scripts under `src/preprocessing` are used for data preprocessing:
  - `custom_transformers.py` contains custom-created transformers which conform to the scikit-learn API for transformers.
  - `pipeline.py` contains the preprocessing pipeline which is used to transform the data. In addition to the custom transformers, it also contains built-in transformers from the `feature-engine` library.
  - `target_encoder.py` contains the target encoder which is used to encode the target variable.
  - `preprocess.py` contains main functions used to train, save, and load the preprocessing pipeline and target encoder. Also contained is a function to transform the data using the pipeline and encoder.
- **`/tests`**: This directory contains all the tests for the project. It contains sub-directories for specific types of tests such as unit tests, integration tests, and performance tests. For unit tests, the directory structure mirrors the `/src` directory structure.
- **`/tmp`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains the license for the project.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.
- **`requirements.txt`**: This file lists the dependencies for the project, making it easier to install all necessary packages.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Move the three example files (`titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`) into the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- - To run the code, simply run the script as follows.

```bash
python src/check_schema.py
```

The script will print information related to the transformed data to the terminal. Also, the trained pipeline and target encoder will be saved to the path `model/artifacts`.

## Requirements

The code requires Python 3 and the following libraries:

```makefile
pandas==1.5.2
numpy==1.20.3
scikit-learn==1.0
feature-engine==1.2.0
imbalanced-learn==0.8.1
```

These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```
