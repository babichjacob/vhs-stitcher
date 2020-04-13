from itertools import islice
from typing import Tuple

from numpy import array, absolute, equal
from thinc.api import Adam, chain, expand_window, fix_random_seed, HashEmbed, Logistic, Model, Optimizer, prefer_gpu, PyTorchWrapper, Relu, Softmax, with_array, with_reshape
from thinc.types import Floats1d, Floats2d, Floats3d
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential
from torch.nn.functional import log_softmax, max_pool2d
from tqdm import tqdm

from . import application_directory, IMAGE_SIZE_SMALL, models_directory, SESSIONS, TEST_RECORDS_PER_SESSION, TRAINING_RECORDS_PER_SESSION
from .assemble import load_set, unzip


def create_neural_network() -> Tuple[Model, Optimizer]:
    # https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    class ConvolutionalNeuralNetwork(Module):
        def __init__(self):
            super().__init__()
            self.convolutional_1 = Sequential(
                Conv2d(in_channels=1, out_channels=48, kernel_size=5, stride=1, padding=2),
                ReLU(),
                MaxPool2d(kernel_size=2),
            )

            self.convolutional_2 = Sequential(
                Conv2d(in_channels=48, out_channels=16, kernel_size=5, stride=1, padding=2),
                ReLU(),
                MaxPool2d(kernel_size=2),
            )

            self.drop_out_1 = Dropout()

            self.fully_connected_1 = Linear(in_features=8192, out_features=96)
            self.fully_connected_2 = Linear(in_features=96, out_features=1)

        def forward(self, x):
            x = self.convolutional_1(x)
            x = self.convolutional_2(x)

            x = x.reshape(x.size(0), -1)

            x = self.drop_out_1(x)

            x = self.fully_connected_1(x)
            x = self.fully_connected_2(x)
            return x

    model = PyTorchWrapper(ConvolutionalNeuralNetwork())
    optimizer = Adam(0.005)

    return model, optimizer


def get_trained_network() -> Model:
    model, _ = create_neural_network()

    try:
        model.from_disk(models_directory / "image_equal.model")
    except FileNotFoundError:
        raise RuntimeError(f"the neural network has not been trained yet -- you need to run `stitch train` to fix this")

    return model



def main(fresh: bool = False, move_studied_records: bool = True):
    # Reproducidibility
    fix_random_seed(100)

    # Try to use the GPU but it's ok if it's not possible
    # Like it isn't for me on my AMD GPU :(
    is_gpu = prefer_gpu()
    print("using gpu" if is_gpu else "!!! not using gpu (this probably sucks!)")

    # Load the training and test sets
    training_questions, training_answers = unzip(load_set(True))
    test_questions, test_answers = unzip(load_set(False))

    model, optimizer = create_neural_network()

    if not fresh:
        try:
            model.from_disk(models_directory / "image_equal.model")
        except FileNotFoundError:
            fresh = True

    if fresh:
        print(f"training the model fresh!")
        # Since the model is fresh, it needs to be initialized
        some_training_questions = array(list(islice(training_questions, 0, 5)))
        some_training_answers = array(list(islice(training_answers, 0, 5)))
        model.initialize(X=some_training_questions, Y=some_training_answers)
    else:
        print(f"continuining to train the pre-existing model")

    # https://github.com/explosion/thinc/blob/master/examples/00_intro_to_thinc.ipynb
    with tqdm(total=100, desc="accuracy", unit="%") as accuracy, tqdm(range(SESSIONS),
                                                                      desc="running training sessions", unit="sessions") as sessions:
        correct = 0
        total = 0

        for session in sessions:
            # Train
            X = array(
                list(
                    islice(
                        training_questions,
                        0,
                        TRAINING_RECORDS_PER_SESSION)))
            Y = array(
                list(
                    islice(
                        training_answers,
                        0,
                        TRAINING_RECORDS_PER_SESSION)))

            # Sometimes we run out of records sooner than expected because of
            # poor software design by me
            if len(X) == 0:
                break

            Yh, backpropagate = model.begin_update(X)
            backpropagate(Yh - Y)
            model.finish_update(optimizer)

            # Test
            X = array(
                list(islice(test_questions, 0, TEST_RECORDS_PER_SESSION)))
            Y = array(
                list(islice(test_answers, 0, TEST_RECORDS_PER_SESSION)))
            Yh = model.predict(X)

            close_enough = absolute(Yh - Y) < 0.5
            correct += close_enough.sum()
            total += Yh.shape[0]

            score = correct / total
            accuracy.n = int(score * 100)
            accuracy.refresh()

    models_directory.mkdir(parents=True, exist_ok=True)
    model.to_disk(models_directory / "image_equal.model")
    print(f"training is complete and saved to {models_directory}")
    print(f"your advised next step is to find two videos you want to combine and run `stitch combine` on them")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
