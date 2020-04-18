from math import floor
from pathlib import Path


src_directory = Path(__file__).parent
application_directory = src_directory.parent

current_dictory = Path.cwd()
# Data directory, where images, models, and eventually results are stored
data_directory = current_dictory / "data"
extracted_frames_directory = data_directory / "extracted_frames"
matches_directory = data_directory / "matches"
models_directory = data_directory / "models"

test_directory = data_directory / "test"
test_equal = test_directory / "equal"
test_unequal = test_directory / "unequal"

training_directory = data_directory / "training"
training_equal = training_directory / "equal"
training_unequal = training_directory / "unequal"

IMAGE_SIZE_LARGE = 128
IMAGE_SIZE_SMALL = 32

TRAINING_SET_EQUAL = 2500
TRAINING_SET_UNEQUAL = 4000

TEST_SET_EQUAL = 500
TEST_SET_UNEQUAL = 500

EPOCHS = 50
TRAINING_RECORDS_PER_EPOCH = floor(
    (TRAINING_SET_EQUAL + TRAINING_SET_UNEQUAL) / EPOCHS)
TEST_RECORDS_PER_EPOCH = floor(
    (TEST_SET_EQUAL + TEST_SET_UNEQUAL) / EPOCHS)
