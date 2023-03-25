import os


# Path to the current file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to inputs 
INPUT_DIR = os.path.join(CURRENT_DIR, "inputs")
# File path for input schema file 
SCHEMA_DIR = os.path.join(INPUT_DIR, "data_config")
# Path to data directory inside inputs directory
DATA_DIR = os.path.join(INPUT_DIR, "data")
# Path to training directory inside data directory
TRAIN_DIR = os.path.join(DATA_DIR, "training")
# Path to test directory inside data directory
TEST_DIR = os.path.join(DATA_DIR, "testing")

# Path to model directory 
MODEL_PATH = os.path.join(CURRENT_DIR, "model")
# Path to artifacts directory inside model directory
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")

# Path to outputs 
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
# Path to logs directory inside outputs directory
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
# Path to testing outputs directory inside outputs directory
TEST_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, "testing_outputs")
# Name of the file containing the predictions
PREDICTIONS_FILE_NAME = "predictions.csv"