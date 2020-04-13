from itertools import islice
from typing import Tuple

from numpy import array, absolute, equal
from thinc.api import Adam, chain, decaying, fix_random_seed, L2Distance, Model, Optimizer, prefer_gpu, PyTorchWrapper
from thinc.types import Floats1d, Floats2d, Floats3d
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid, Softmax
from torch.nn.functional import log_softmax, max_pool2d
from tqdm import tqdm

from . import application_directory, IMAGE_SIZE_SMALL, models_directory, SESSIONS, TEST_RECORDS_PER_SESSION, TRAINING_RECORDS_PER_SESSION
from .assemble import load_set, unzip


# Adapted from
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
class ConvolutionalNeuralNetwork(Module):
    def __init__(
            self,
            kernel_size: int,
            padding: int,
            layer_1_convolutions: int,
            layer_2_convolutions: int,
            dropout_probability: float,
            fully_connected_1_neurons: int,
            fully_connected_2_neurons: int,
            outputs: int):
        super().__init__()
        # The first 2 layers (combined into a sequential layer for PyTorch) are
        # a 2D convolutional one followed by a 2x2->1 max pooling layer with
        # relu activation
        self.convolutional_1 = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=layer_1_convolutions,
                kernel_size=kernel_size,
                stride=1,
                padding=padding),
            MaxPool2d(
                kernel_size=2),
            ReLU(),
        )

        # The 3rd and 4th layer are the same
        self.convolutional_2 = Sequential(
            Conv2d(
                in_channels=layer_1_convolutions,
                out_channels=layer_2_convolutions,
                kernel_size=kernel_size,
                stride=1,
                padding=padding),
            MaxPool2d(
                kernel_size=2),
            ReLU(),
        )

        # Then there's a dropout "layer" to make all the neurons contribute to
        # the result instead of a select few
        self.drop_out_1 = Dropout(p=dropout_probability)

        # Penultimately, we have two layers of regular neural network stuff
        # with linear activation functions
        self.fully_connected_1 = Linear(
            in_features=fully_connected_1_neurons,
            out_features=fully_connected_2_neurons)
        self.fully_connected_2 = Linear(
            in_features=fully_connected_2_neurons,
            out_features=outputs)

        # Finally, the last layer is a softmax calculator so we get probability-like behavior
        # This used not to be necessary but I've started using it to get more
        # flexibility
        self.softmax = Softmax(dim=1)

    def forward(self, X):
        # Call all the layers in the right order
        X = self.convolutional_1(X)
        X = self.convolutional_2(X)

        # Flatten the 2-dimensional "image" for the upcoming layers
        X = X.reshape(X.size(0), -1)

        X = self.drop_out_1(X)

        X = self.fully_connected_1(X)
        X = self.fully_connected_2(X)

        return self.softmax(X)


def create_neural_network() -> Tuple[Model, Optimizer]:
    # These parameters managed to result in high accuracy
    layer_1_convolutions = 32
    layer_2_convolutions = 64

    # Remember, there are two horizontally stacked images in the input
    image_width = IMAGE_SIZE_SMALL * 2
    image_height = IMAGE_SIZE_SMALL

    # For some reason, this empirically determined formula gives the correct
    # (i.e. there will be an error if it's wrong) number of outputting neurons
    # from the image size and number of convolutional filters used
    scalar = int(layer_1_convolutions * layer_2_convolutions / (16 * 32))
    fully_connected_1_neurons = scalar * (image_width * image_height)

    # There are 2 neurons: one for equality confidence and one for inequality confidence
    # These used to be just 1 neuron but I revised it to this for flexibility
    outputs = 2

    network = ConvolutionalNeuralNetwork(
        # Okay sized convolutional masks
        kernel_size=5,
        padding=2,
        layer_1_convolutions=layer_1_convolutions,
        layer_2_convolutions=layer_2_convolutions,
        dropout_probability=0.60,
        fully_connected_1_neurons=fully_connected_1_neurons,
        # Using the geometric mean to get a "good" intermediate valued layer
        fully_connected_2_neurons=int(
            (fully_connected_1_neurons * outputs)**0.5),
        outputs=outputs,
    )

    # Linearly decaying learning learning rate
    # i.e. start high and gradually decrease
    learning_rates = decaying(0.001, 0.5)

    model = PyTorchWrapper(network)
    optimizers = map(Adam, learning_rates)
    loss = L2Distance()

    return model, optimizers, loss


def get_trained_network() -> Model:
    model, *_ = create_neural_network()

    try:
        model.from_disk(models_directory / "images_are_equal.model")
    except FileNotFoundError:
        raise RuntimeError(
            f"the neural network has not been trained yet -- you need to run `stitch train` to fix this")

    return model


def main(fresh: bool = False, move_studied_records: bool = True):
    # Reproducidibility
    fix_random_seed(0)

    # Try to use the GPU but it's ok if it's not possible
    # Like it isn't for me on an Intel iGPU or AMD GPU :(
    using_gpu: bool = prefer_gpu()
    print("using GPU" if using_gpu else "not able to use GPU")

    # Load the training and test sets
    training_questions, training_answers = unzip(load_set(True))
    test_questions, test_answers = unzip(load_set(False))

    model, optimizers, loss = create_neural_network()

    if not fresh:
        try:
            model = get_trained_network()
        except RuntimeError:
            fresh = True

    if fresh:
        print(f"training the model fresh!")
        # Since the model is fresh, it needs to be initialized
        some_training_questions = array(list(islice(training_questions, 0, 3)))
        some_training_answers = array(list(islice(training_answers, 0, 3)))
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
            backpropagate(loss(Yh, Y))
            model.finish_update(next(optimizers))

            # Test
            X = array(
                list(islice(test_questions, 0, TEST_RECORDS_PER_SESSION)))
            Y = array(
                list(islice(test_answers, 0, TEST_RECORDS_PER_SESSION)))

            Yh = model.predict(X)

            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]

            score = correct / total
            accuracy.n = int(score * 100)
            accuracy.refresh()

    models_directory.mkdir(parents=True, exist_ok=True)
    model.to_disk(models_directory / "images_are_equal.model")

    print(f"training is complete and saved to {models_directory}")
    print(f"your advised next step is to find two videos you want to combine and run `stitch combine` on them")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
