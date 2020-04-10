from pathlib import Path

root_directory = Path()
# Data directory, where images, models, and eventually results are stored
data_directory = root_directory / "data"
extracted_frames = data_directory / "extracted_frames"
matches = data_directory / "matches"
models = data_directory / "models"

test_directory = data_directory / "test"
test_equal = test_directory / "equal"
test_unequal = test_directory / "unequal"

training_directory = data_directory / "training"
training_equal = training_directory / "equal"
training_unequal = training_directory / "unequal"

IMAGE_SIZE_LARGE = 256
IMAGE_SIZE_SMALL = 64

TRAINING_SET_EQUAL = 2000
TRAINING_SET_UNEQUAL = 8000

TEST_SET_EQUAL = 500
TEST_SET_UNEQUAL = 4000
