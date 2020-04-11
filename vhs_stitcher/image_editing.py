from typing import Dict
from uuid import uuid4

from imageio import imread, imwrite
from numpy import array, hstack, ndarray, subtract
from PIL import Image
from PIL.Image import Image as ImageType
from PIL.ImageFilter import FIND_EDGES, GaussianBlur
from tqdm import tqdm

from . import IMAGE_SIZE_LARGE, IMAGE_SIZE_SMALL


def grayscale(image: ImageType) -> ImageType:
    return image.convert('L')


def prepare_image(image: ImageType):
    """
    1. Downscale the image a little bit to a square
    2. Blur (1)
    3. Find the edges in (1)
    4. Downscale both a lot and return them
    """

    image_square = image.resize((IMAGE_SIZE_LARGE, IMAGE_SIZE_LARGE))

    image_blurred = image_square.filter(GaussianBlur(radius=4))
    image_blurred_small = image_blurred.resize(
        (IMAGE_SIZE_SMALL, IMAGE_SIZE_SMALL))

    image_edges = image_square.filter(FIND_EDGES)
    image_edges_small = image_edges.resize(
        (IMAGE_SIZE_SMALL, IMAGE_SIZE_SMALL))

    return {
        "edges_small": image_edges_small,
        "blurred_small": image_blurred_small,
    }


def side_by_side(image_1: ImageType, image_2: ImageType) -> ImageType:
    "Create an image with the two images side by side (horizontally)"

    horizontal = hstack((array(image_1), array(image_2)))
    return Image.fromarray(horizontal)


def flatten(image: ImageType) -> array:
    multidimensional = array(image)
