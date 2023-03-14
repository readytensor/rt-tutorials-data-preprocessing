import os


# Path to the current file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to inputs inside ml_vol
INPUT_DIR = os.path.join(CURRENT_DIR, "inputs")

# File path for input schema file 
SCHEMA_DIR = os.path.join(INPUT_DIR, "data_config")

# Path to data directory inside inputs directory
DATA_DIR = os.path.join(INPUT_DIR, "data")
# Path to training directory inside data directory
TRAIN_DIR = os.path.join(DATA_DIR, "training")
# Path to test directory inside data directory
TEST_DIR = os.path.join(DATA_DIR, "testing")

# Path to model directory inside ml_vol
MODEL_PATH = os.path.join(CURRENT_DIR, "model")
# Path to artifacts directory inside model directory
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
