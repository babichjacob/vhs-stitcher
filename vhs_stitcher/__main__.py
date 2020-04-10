from imageio import imwrite
from itertools import islice
from ml_datasets import mnist
from numpy import array
from pathlib import Path
from thinc.api import Adam, chain, Config, fix_random_seed, Model, prefer_gpu, registry
from tqdm import tqdm


def train_model(data, model, optimizer, n_iter, batch_size):
    (train_X, train_Y), (dev_X, dev_Y) = data
    indices = model.ops.xp.arange(train_X.shape[0], dtype="i")

    for i in range(n_iter):
        batches = model.ops.multibatch(
            batch_size, train_X, train_Y, shuffle=True)

        for X, Y in tqdm(batches, leave=False):
            Yh, backpropagate = model.begin_update(X)
            backpropagate(Yh - Y)
            model.finish_update(optimizer)

        correct = 0
        total = 0

        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]

        score = correct / total

        print(f"  {i}: {float(score):.3f}")


def main(
    *,
    train: bool = False,
    save: bool = False,
        examine_data: bool = False):
    fix_random_seed(0)
    is_gpu = prefer_gpu()
    print("using gpu" if is_gpu else "not using gpu")

    this_directory = Path(__file__).parent
    parent_directory = this_directory.parent

    with open(parent_directory / "thinc.ini") as config_ini:
        config_str = config_ini.read()

    config = Config().from_str(config_str)
    loaded_config = registry.make_from_config(config)

    model = loaded_config["model"]
    optimizer = loaded_config["optimizer"]
    n_iter = loaded_config["training"]["n_iter"]
    batch_size = loaded_config["training"]["batch_size"]

    (train_X, train_Y), (dev_X, dev_Y) = mnist()

    model.initialize(X=train_X[:5], Y=train_Y[:5])

    if train:
        print("Training:")
        train_model(((train_X, train_Y), (dev_X, dev_Y)),
                    model, optimizer, n_iter, batch_size)

        if save:
            model_directory = parent_directory / "models"
            from json import dump
            from pprint import pprint
            pprint(model.to_dict())
            model.to_disk(model_directory / "mnist.model")
            print("saved")

    elif examine_data:
        print("Take a look at this data:")

        image_directory = parent_directory / "data"
        image_directory.mkdir(exist_ok=True)

        for i, (X, _) in tqdm(enumerate(zip(train_X, train_Y))):
            if i > 20:
                break

            size = int(len(X)**0.5)
            unflattened = array(
                [array([X[i * size + j] * 255 for j in range(size)]) for i in range(size)])

            imwrite(image_directory / f"{i}.jpg", unflattened)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
