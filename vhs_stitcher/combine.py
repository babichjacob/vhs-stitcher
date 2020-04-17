from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from shlex import quote
from subprocess import run
from typing import Dict, Iterable, List, Tuple, TypeVar

from imageio import imread
from numpy import argmax, array, floor, linspace
from PIL import Image
from PIL.Image import Image as ImageType
from tqdm import tqdm

from . import extracted_frames_directory
from .assemble import create_image_comparison
from .extract import extract
from .train import get_trained_network

ItemType = TypeVar("ItemType")


def main(chunk_1: str, chunk_2: str, output: str):
    # Extract the frames for analysis
    extraction_data = extract(
        chunk_1=chunk_1, chunk_2=chunk_2, delete_last=True)

    ending_frames = extraction_data["ending_frames"]
    continuing_frames = extraction_data["continuing_frames"]
    ending_video_fps = extraction_data["ending_video_fps"]
    continuing_video_fps = extraction_data["continuing_video_fps"]
    ending_video_start = extraction_data["ending_video_start"]

    # Select 11 frames equally spread across the ending video's frames
    # To compare against every selected continuing video frame to find a match
    ending_search_points = linspace(start=0, stop=1, num=11)
    selected_ending_frames_indices = floor(
        ending_search_points * (ending_frames - 1)).astype(int)
    # Read into memory beforehand (optimization)
    selected_ending_frames = [
        imread(extracted_frames_directory /
               "ending" / f"{ending_frame_index}.jpg")
        for ending_frame_index in selected_ending_frames_indices
    ]
    # And convert to an image beforehand (optimization)
    selected_ending_images = [Image.fromarray(
        ending_frame) for ending_frame in selected_ending_frames]

    # TODO: optimize
    selected_continuing_frame_indices = range(continuing_frames)

    # Use the trained neural network to find matches
    model = get_trained_network()

    potential_matches: Dict[int, Dict[int, float]] = defaultdict(dict)
    for continuing_frame_index in tqdm(
            selected_continuing_frame_indices,
            desc="scanning for matches between the two videos",
            unit="frames"):
        continuing_frame = imread(
            extracted_frames_directory /
            "continuing" /
            f"{continuing_frame_index}.jpg")

        continuing_image = Image.fromarray(continuing_frame)

        # TODO: predict multiple at once
        for ending_image, ending_frame_index in zip(selected_ending_images, selected_ending_frames_indices):
            comparison = create_image_comparison(
                continuing_image, ending_image)
            comparison_array = array(comparison)
            correctly_formatted_comparison_array = comparison_array.reshape(
                1, 1, comparison_array.shape[0], comparison_array.shape[1]).astype(
                dtype="float32") / 255

            [[equal_chance, unequal_chance]] = model.predict(
                correctly_formatted_comparison_array)

            if equal_chance > 0.88:
                # Add this to candidate matches
                potential_matches[ending_frame_index][continuing_frame_index] = equal_chance

    best_matches: Dict[Tuple[int, int], float] = {}
    for ending_frame, continuing_frames_with_probabilities in potential_matches.items():
        best_continuing_probability = max(
            continuing_frames_with_probabilities.values())
        # It's possible one frame has the same probability as another. Ignore that
        best_continuing_frame = [frame for frame, probability in continuing_frames_with_probabilities.items(
        ) if probability == best_continuing_probability][0]

        best_matches[(ending_frame, best_continuing_frame)
                     ] = best_continuing_probability

    best_probability_overall = max(best_matches.values())
    # It's possible one frame has the same probability as another. Ignore that
    best_ending_frame_overall, best_continuing_frame_overall = [frame_pair for frame_pair, probability in best_matches.items(
    ) if probability == best_probability_overall][0]

    # Where to stop the first video
    ending_outpoint = ending_video_start + \
        (best_ending_frame_overall / ending_video_fps)
    # Where to resume from the second
    continuing_inpoint = best_continuing_frame_overall / continuing_video_fps

    # We have all the information we need to combine the videos

    # This data has to be written to a file because FFMPEG
    output_path = Path(output)
    output_directory = output_path.parent

    # https://ffmpeg.org/ffmpeg-all.html#Syntax-1
    ffmpeg_input = [
        "ffconcat version 1.0",
        f"file {quote(str(Path(chunk_1).relative_to(output_directory).as_posix()))}",
        f"outpoint {ending_outpoint}",
        f"file {quote(str(Path(chunk_2).relative_to(output_directory).as_posix()))}",
        f"inpoint {continuing_inpoint}",
    ]

    # Try to make it hidden to the user on POSIX operating systems
    ffmpeg_input_path = (output_directory / ".ffmpeg_input.txt")
    ffmpeg_input_path.write_text("\n".join(ffmpeg_input), encoding="utf8")

    ffmpeg_command = ["ffmpeg", "-safe", "0", "-i", str(ffmpeg_input_path.as_posix()), "-c", "copy",
                      # https://superuser.com/a/1064356
                      "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
                      str(output_path.as_posix())]

    try:
        run(ffmpeg_command, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("FFMPEG is not installed") from exc

    # Don't leave the clutter behind
    ffmpeg_input_path.unlink()

    # Convert total seconds to hours, minutes, seconds, like a clock
    combination_hours, combination_seconds = divmod(ending_outpoint, 3600)
    combination_minutes, combination_seconds = divmod(ending_outpoint, 60)

    print()
    print()
    print()
    print(
        f"these two videos have been combined/'stitched' at {int(combination_hours):02}:{int(combination_minutes):02}:{int(combination_seconds):02}")
    print("see if you can spot the change!")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
