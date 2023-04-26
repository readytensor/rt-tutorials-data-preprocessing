import os

# Path to the root directory, which is `app` directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to config directory
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
# Path to model config
MODEL_CONFIG_FILE_PATH  = os.path.join(CONFIG_DIR, "model_config.json")
# Path to model config
HYPERPARAMETERS_FILE_PATH  = os.path.join(CONFIG_DIR, "hyperparameters.json")

# Path to inputs
INPUT_DIR = os.path.join(ROOT_DIR, "inputs")
# File path for input schema file
SCHEMA_DIR = os.path.join(INPUT_DIR, "data_config")
# Path to data directory inside inputs directory
DATA_DIR = os.path.join(INPUT_DIR, "data")
# Path to training directory inside data directory
TRAIN_DIR = os.path.join(DATA_DIR, "training")
# Path to test directory inside data directory
TEST_DIR = os.path.join(DATA_DIR, "testing")

# Path to model directory
MODEL_PATH = os.path.join(ROOT_DIR, "model")
# Path to artifacts directory inside model directory
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
# Name of the classifier model file inside artifacts directory
CLASSIFIER_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "classifier.joblib")
# Name of the preprocessing pipeline file
PIPELINE_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "pipeline.joblib")
# Name of the label encoder file
LABEL_ENCODER_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "label_encoder.joblib")

# Path to outputs
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
# Path to logs directory inside outputs directory
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
# Path to testing outputs directory inside outputs directory
BATCH_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, "batch_outputs")
# Name of the file containing the predictions
PREDICTIONS_FILE_PATH = os.path.join(BATCH_OUTPUTS_DIR, "predictions.csv")

# Log file paths
TRAIN_LOG_FILE_PATH = os.path.join(LOGS_DIR, "train.log")
PREDICT_LOG_FILE_PATH = os.path.join(LOGS_DIR, "predict.log")
SERVE_LOG_FILE_PATH = os.path.join(LOGS_DIR, "serve.log")

# Error file paths
TRAIN_ERROR_FILE_PATH = os.path.join(LOGS_DIR, "train-error.log")
PREDICT_ERROR_FILE_PATH = os.path.join(LOGS_DIR, "predict-error.log")
SERVE_ERROR_FILE_PATH = os.path.join(LOGS_DIR, "serve-error.log")
