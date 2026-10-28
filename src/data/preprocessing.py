
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


def carregar_dados(nome_arquivo: str) -> pd.DataFrame:
    """Carrega um dataset CSV da pasta de dados brutos."""

    caminho = RAW_DATA_DIR / nome_arquivo

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho}"
        )

    dados = pd.read_csv(caminho)

    if dados.empty:
        raise ValueError("O dataset está vazio.")

    return dados


def resumo_dados(dados: pd.DataFrame) -> None:
    """Apresenta informações básicas do dataset."""

    print(f"Linhas: {dados.shape[0]}")
    print(f"Colunas: {dados.shape[1]}")
    print("\nTipos de dados:")
    print(dados.dtypes)
    print("\nValores ausentes:")
    print(dados.isnull().sum())


def preparar_dados(dados: pd.DataFrame):
    """Prepara os dados para classificação de intenção de compra."""

    if "Revenue" not in dados.columns:
        raise ValueError("Coluna Revenue não encontrada.")

    X = dados.drop(columns=["Revenue"])
    y = dados["Revenue"].astype(int)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    colunas_numericas = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    colunas_categoricas = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessador = ColumnTransformer([
        ("numerico", pipeline_numerico, colunas_numericas),
        ("categorico", pipeline_categorico, colunas_categoricas),
    ])

    return X_treino, X_teste, y_treino, y_teste, preprocessador


def executar_preprocessamento() -> None:
    """Gera o dataset preparado para o pipeline DVC."""

    dados = carregar_dados("online_shoppers_intention.csv")

    pasta_saida = Path("data/processed")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    caminho_saida = pasta_saida / "online_shoppers_processed.csv"

    # Mantém os dados originais e padroniza a variável-alvo.
    dados["Revenue"] = dados["Revenue"].astype(int)

    dados.to_csv(caminho_saida, index=False)

    print(f"Dataset preparado: {caminho_saida}")
    print(f"Registros: {len(dados)}")


if __name__ == "__main__":
    executar_preprocessamento()
