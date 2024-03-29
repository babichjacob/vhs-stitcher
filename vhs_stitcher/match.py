from uuid import uuid4

from tqdm import tqdm

from . import extracted_frames_directory, matches_directory


def main(ending_range_str: str, continuing_range_str: str):
    ending_start_str, _, ending_end_str = ending_range_str.partition(":")
    continuing_start_str, _, continuing_end_str = continuing_range_str.partition(
        ":")

    ending_start, ending_end = int(ending_start_str), int(ending_end_str)
    continuing_start, continuing_end = int(
        continuing_start_str), int(continuing_end_str)

    ending_range = range(ending_start, ending_end + 1)
    continuing_range = range(continuing_start, continuing_end + 1)
    # Demand that the ranges are the same length
    if len(ending_range) != len(continuing_range):
        raise ValueError(
            f"ranges cannot be different lengths: {len(ending_range)} (ending) vs {len(continuing_range)} (continuing). look over the frames again")

    matches_directory.mkdir(parents=True, exist_ok=True)
    for ending_image_index, continuing_image_index in tqdm(zip(
            ending_range, continuing_range), total=len(ending_range), desc="matching", unit="pairings"):
        ending_path = extracted_frames_directory / \
            "ending" / f"{ending_image_index}.jpg"
        continuing_path = extracted_frames_directory / \
            "continuing" / f"{continuing_image_index}.jpg"

        match_id = uuid4()
        match_directory = matches_directory / f"{match_id}"
        match_directory.mkdir(parents=True, exist_ok=True)

        ending_path.rename(match_directory / "0.jpg")
        continuing_path.rename(match_directory / "1.jpg")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
