from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "online_shoppers_intention.csv"
)


def test_dataset_existe():
    assert DATA_PATH.is_file(), "Dataset não encontrado"


def test_dataset_nao_esta_vazio():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset está vazio"


def test_variavel_alvo_existe():
    df = pd.read_csv(DATA_PATH)
    assert "Revenue" in df.columns, "Coluna Revenue não encontrada"


def test_variavel_alvo_valida():
    df = pd.read_csv(DATA_PATH)
    assert df["Revenue"].notna().all(), "Revenue possui valores nulos"
    assert df["Revenue"].isin([True, False, 0, 1]).all(), (
        "Revenue possui valores inválidos"
    )
