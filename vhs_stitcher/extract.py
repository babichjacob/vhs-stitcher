from contextlib import closing
from itertools import islice
from math import ceil, floor
from pathlib import Path
from shutil import rmtree
from typing import cast
from warnings import warn

from imageio import get_reader, imwrite
from tqdm import tqdm

from . import extracted_frames_directory


def extract(
        chunk_1: str,
        chunk_2: str,
        seconds: float = 90.0,
        delete_last: bool = False) -> None:
    "Extract the endings of and beginnings of the first chunk and second chunk respectively to the extracted_frames directory"

    extracted_frames_directory.mkdir(parents=True, exist_ok=True)

    if any(extracted_frames_directory.iterdir()):
        if delete_last:
            print(
                f"deleting previous frames from the {extracted_frames_directory} directory")
            rmtree(extracted_frames_directory)
            extracted_frames_directory.mkdir(parents=True, exist_ok=True)
        else:
            warn(
                f"there were still files in the {extracted_frames_directory} directory, so they are likely to be overwritten or be hard to distinguish from freshly extracted frames. run again with --delete-last to delete them before extracting new frames")

    return_data = {}

    for chunk, ending in [[chunk_1, True], [chunk_2, False]]:
        chunk_path = Path(chunk)

        # Just make a reader only for metadata necessary for trimming
        metadata_reader = get_reader(chunk_path, format="FFMPEG")

        metadata = metadata_reader.get_meta_data()
        fps: float = metadata["fps"]

        if ending:
            frames: int = metadata_reader.count_frames()
            video_length_seconds: float = frames / fps

            start = video_length_seconds - seconds
            end = video_length_seconds

            return_data["ending_video_fps"] = fps
            return_data["ending_video_start"] = start
            return_data["ending_video_end"] = end
        else:
            start = 0.0
            end = seconds

            return_data["continuing_video_fps"] = fps
            return_data["continuing_video_start"] = start
            return_data["continuing_video_end"] = end

        total_seconds_extracted = end - start
        expected_total_frames_extracted = total_seconds_extracted * fps

        output_directory = extracted_frames_directory / \
            ("ending" if ending else "continuing")
        output_directory.mkdir(parents=True, exist_ok=True)

        # Create a new reader, this time to actually read data from the video
        # file
        input_params = ["-ss", f"{start}"]
        output_params = ["-t", f"{total_seconds_extracted}"]
        reader = get_reader(
            chunk_path,
            format="FFMPEG",
            input_params=input_params,
            output_params=output_params)

        progress_bar = tqdm(
            enumerate(reader),
            total=ceil(expected_total_frames_extracted),
            desc=f"extracting {chunk}",
            unit="frames")
        for index, image in progress_bar:
            imwrite(output_directory / f"{index}.jpg", image)

        # Keep track of the true number of frames extracted
        if ending:
            return_data["ending_frames"] = index + 1
        else:
            return_data["continuing_frames"] = index + 1

    return return_data


def main(
        chunk_1: str,
        chunk_2: str,
        seconds: float = 90.0,
        delete_last: bool = False):
    extract(
        chunk_1=chunk_1,
        chunk_2=chunk_2,
        seconds=seconds,
        delete_last=delete_last)

    print(f"all frames have been extracted to {extracted_frames_directory}")
    print(f"your advised next step is to open up the file viewer to identify ranges of matches from `continuing` and `ending`")
    print(f"then use `stitch match` to organize these into matches")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
