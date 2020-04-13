from .extract import main as extract
from .train import get_trained_network


def main(chunk_1: str, chunk_2: str):
    extract(chunk_1=chunk_1, chunk_2=chunk_2)

    model = get_trained_network()

    # todo
    ...


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
