from itertools import tee
from pathlib import Path
from random import shuffle
from typing import cast, Dict, Generator, Iterable, Iterator, List, Set, Tuple, TypeVar

from imageio import imread, imwrite
from numpy import array, ndarray, ones, zeros
from numpy.random import choice
from PIL import Image
from PIL.Image import Image as ImageType
from PIL.ImageChops import difference
from tqdm import tqdm
from uuid import uuid4

from . import matches_directory, IMAGE_SIZE_LARGE, IMAGE_SIZE_SMALL, TEST_SET_EQUAL, TEST_SET_UNEQUAL, TRAINING_SET_EQUAL, TRAINING_SET_UNEQUAL, training_equal, training_unequal, test_equal, test_unequal
from .image_editing import grayscale, prepare_image, side_by_side

Record = Tuple[Path, Path]


def assemble_sets(match_names: List[Path],
                  training_sets: bool,
                  test_sets: bool) -> Dict[str,
                                           Dict[str,
                                                Set[Record]]]:
    if training_sets:
        training_equals: Set[Record] = set()
        selected_matches = choice(match_names, TRAINING_SET_EQUAL)
        for match in tqdm(
                selected_matches,
                desc="assembling training set for equals",
                unit="records"):
            random_match = choice(match_names)
            paired_with_itself = (random_match, random_match)
            training_equals.add(paired_with_itself)

        training_unequals: Set[Record] = set()
        for _ in tqdm(
                range(TRAINING_SET_UNEQUAL),
                desc="assembling training set for unequals",
                unit="records"):
            while True:
                random_pair = cast(Record, tuple(choice(match_names, 2)))
                if random_pair not in training_unequals:
                    break

            training_unequals.add(random_pair)

    if test_sets:
        test_equals: Set[Record] = set()
        for _ in tqdm(
                range(TEST_SET_EQUAL),
                desc="assembling test set for equals",
                unit="records"):
            while True:
                random_match = choice(match_names)
                paired_with_itself = (random_match, random_match)
                # Do not allow records from the training set into the test set
                if paired_with_itself not in training_equals:
                    break

            test_equals.add(paired_with_itself)

        test_unequals: Set[Record] = set()
        for _ in tqdm(
                range(TEST_SET_UNEQUAL),
                desc="assembling test set for unequals",
                unit="records"):
            while True:
                random_pair = cast(Record, tuple(choice(match_names, 2)))
                # Do not allow records from the training set into the test set
                if random_pair not in training_unequals:
                    break

            test_unequals.add(random_pair)

    return {
        **({"training": {
            "equals": training_equals,
            "unequals": training_unequals,
        }} if training_sets else {}),
        **({"test": {
            "equals": test_equals,
            "unequals": test_unequals,
        }} if test_sets else {})
    }


def create_image_comparison(
        image_1: ImageType,
        image_2: ImageType) -> ImageType:
    image_1_results, image_2_results = [
        prepare_image(image) for image in [
            image_1, image_2]]

    edge_difference = difference(
        image_1_results["edges_small"], image_2_results["edges_small"])
    blurred_difference = difference(
        image_1_results["blurred_small"], image_2_results["blurred_small"])

    return side_by_side(
        grayscale(edge_difference),
        grayscale(blurred_difference))


def create_and_save_records(sets, training_sets: bool, test_sets: bool):
    if training_sets:
        training_equal.mkdir(parents=True, exist_ok=True)
        for (
                match_1,
                _) in tqdm(
                sets["training"]["equals"],
                desc="creating equal training records",
                unit="records"):

            image_paths = [Path(match_1) / f"{i}.jpg" for i in [0, 1]]
            image_1, image_2 = [
                Image.fromarray(
                    imread(image_path)) for image_path in image_paths]

            image_comparison = create_image_comparison(image_1, image_2)

            record_id = uuid4()
            imwrite(training_equal / f"{record_id}.png", image_comparison)

        training_unequal.mkdir(parents=True, exist_ok=True)
        for (match_1, match_2) in tqdm(
                sets["training"]["unequals"],
                desc="creating unequal training records",
                unit="records"):

            index_1, index_2 = choice([0, 1], 2)

            image_paths = [
                Path(match_1) /
                f"{index_1}.jpg",
                Path(match_2) /
                f"{index_2}.jpg"]
            image_1, image_2 = [
                Image.fromarray(
                    imread(image_path)) for image_path in image_paths]

            image_comparison = create_image_comparison(image_1, image_2)

            record_id = uuid4()
            imwrite(training_unequal / f"{record_id}.png", image_comparison)

    if test_sets:
        test_equal.mkdir(parents=True, exist_ok=True)
        for (
                match_1,
                _) in tqdm(
                sets["test"]["equals"],
                desc="creating equal test records",
                unit="records"):

            image_paths = [Path(match_1) / f"{i}.jpg" for i in [0, 1]]
            image_1, image_2 = [Image.fromarray(
                imread(image_path)) for image_path in image_paths]

            image_comparison = create_image_comparison(image_1, image_2)

            record_id = uuid4()
            imwrite(test_equal / f"{record_id}.png", image_comparison)

        test_unequal.mkdir(parents=True, exist_ok=True)
        for (match_1, match_2) in tqdm(
                sets["test"]["unequals"],
                desc="creating unequal test records",
                unit="records"):

            index_1, index_2 = choice([0, 1], 2)

            image_paths = [
                Path(match_1) /
                f"{index_1}.jpg",
                Path(match_2) /
                f"{index_2}.jpg"]
            image_1, image_2 = [Image.fromarray(
                imread(image_path)) for image_path in image_paths]

            image_comparison = create_image_comparison(image_1, image_2)

            record_id = uuid4()
            imwrite(test_unequal / f"{record_id}.png", image_comparison)


def images_in(directory: Path) -> Generator[Path, None, None]:
    for path in directory.iterdir():
        if path.suffix not in [".jpg", ".png"]:
            continue
        # Ignore macOS dotfiles
        if path.stem.startswith("._"):
            continue

        yield path


def load_set(training: bool) -> Generator[Tuple[ndarray, ndarray], None, None]:
    "Load either the training (True) or test (False) set"

    paths: List[Path] = []

    if training:
        paths.extend(images_in(training_equal))
        paths.extend(images_in(training_unequal))
    else:
        paths.extend(images_in(test_equal))
        paths.extend(images_in(test_unequal))

    # Randomize the entries so that the network gets a balanced curriculum
    shuffle(paths)

    for path in paths:
        image = imread(path)
        image_array = array(image).ravel()
        answer = zeros(1) if "unequal" in str(path.parent) else ones(1)
        yield (image_array.astype(dtype="float32"), answer.astype(dtype="float32"))


Type1 = TypeVar("Type1")
Type2 = TypeVar("Type2")


def unzip(iterable: Iterable[Tuple[Type1, Type2]]
          ) -> Tuple[Iterator[Type1], Iterator[Type2]]:
    copy_1, copy_2 = tee(iterable)

    def iter_1():
        for val_1, _ in copy_1:
            yield val_1

    def iter_2():
        for _, val_2 in copy_2:
            yield val_2

    return iter_1(), iter_2()


def main(no_training_sets: bool = False, no_test_sets: bool = False):
    match_names = sorted(matches_directory.iterdir())
    sets = assemble_sets(match_names, not no_training_sets, not no_test_sets)
    # Create a line separating the progress bars for assembling sets and
    # creating records
    print()
    create_and_save_records(sets, not no_training_sets, not no_test_sets)

    print()
    print(f"{'' if no_training_sets else 'training / '}{'' if no_test_sets else 'test'} sets for equal / unequal images have been created")
    print(f"your advised next step is to train and test the neural network off these records with `stitch train`")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
