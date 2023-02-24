import os

THIS_FPATH = os.path.dirname(os.path.abspath(__file__))
SCHEMA_FPATH = os.path.join(THIS_FPATH, "inputs", "titanic_schema.json")
TRAIN_DATA_FPATH = os.path.join(THIS_FPATH, "inputs", "titanic_train.csv")
TEST_DATA_FPATH = os.path.join(THIS_FPATH, "inputs", "titanic_test.csv")
MODEL_ARTIFACTS_PATH = os.path.join(THIS_FPATH, "outputs", "artifacts")
MODEL_PREDICTIONS_PATH = os.path.join(THIS_FPATH, "outputs", "predictions", "predictions.csv")
