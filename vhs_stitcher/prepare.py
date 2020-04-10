from typing import Dict
from uuid import uuid4

from imageio import imread, imwrite
from numpy import absolute, array, hstack, ndarray, subtract
from PIL import Image
from PIL.Image import Image as ImageType
from PIL.ImageFilter import FIND_EDGES, GaussianBlur
from tqdm import tqdm

from . import IMAGE_SIZE_LARGE, IMAGE_SIZE_SMALL


def prepare_image(image: ImageType):
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


def side_by_side(image1: ImageType, image2: ImageType) -> ImageType:
    horizontal = hstack((array(image1), array(image2)))
    return Image.fromarray(horizontal)
