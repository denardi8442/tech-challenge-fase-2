
from pathlib import Path
from urllib.request import urlretrieve

URL_DATASET = (
    "https://github.com/denardi8442/"
    "tech-challenge-fase-2/releases/download/"
    "dataset-v1/online_shoppers_intention.csv"
)

DESTINO = Path("data/raw/online_shoppers_intention.csv")


def baixar_dataset() -> None:
    """Baixa o dataset público caso ele não exista localmente."""

    if DESTINO.exists():
        print("Dataset já disponível.")
        return

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL_DATASET, DESTINO)
    print(f"Dataset baixado: {DESTINO}")


if __name__ == "__main__":
    baixar_dataset()
