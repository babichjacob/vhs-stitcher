from contextlib import closing
from itertools import islice
from math import ceil, floor
from pathlib import Path
from shutil import rmtree
from typing import cast
from warnings import warn

from imageio import get_reader, imwrite
from tqdm import tqdm

from . import extracted_frames


def main(
        chunk_1: str,
        chunk_2: str,
        seconds: float = 90.0,
        delete_last: bool = False) -> None:
    "Extract the endings of and beginnings of the first chunk and second chunk respectively to the extracted_frames directory"

    extracted_frames.mkdir(parents=True, exist_ok=True)

    if any(extracted_frames.iterdir()):
        if delete_last:
            print(
                f"deleting previous frames from the {extracted_frames} directory")
            rmtree(extracted_frames)
            extracted_frames.mkdir(parents=True, exist_ok=True)
        else:
            warn(
                f"there were still files in the {extracted_frames} directory, so they are likely to be overwritten or be hard to distinguish from freshly extracted frames. run again with --delete-last to delete them before extracting new frames")

    for chunk, ending in [[chunk_1, True], [chunk_2, False]]:
        chunk_path = Path(cast(str, chunk))

        # Just make a reader only for metadata necessary for trimming
        metadata_reader = get_reader(chunk_path, format="FFMPEG")

        metadata = metadata_reader.get_meta_data()
        fps: float = metadata["fps"]

        if ending:
            frames: int = metadata_reader.count_frames()
            video_length_seconds: float = frames / fps
            start = video_length_seconds - seconds
            end = video_length_seconds
        else:
            start = 0.0
            end = seconds

        total_seconds_extracted = end - start
        total_frames_extracted = total_seconds_extracted * fps

        output_directory = extracted_frames / \
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
            total=ceil(total_frames_extracted),
            desc=f"extracting {chunk}",
            unit="frames")
        for index, image in progress_bar:
            imwrite(output_directory / f"{index}.jpg", image)

    print(f"all frames have been extracted to {extracted_frames}")
    print(f"your advised next step is to open up the file viewer to identify ranges of matches from `continuing` and `ending`")
    print(f"then use `stitch match` to organize these into matches")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
