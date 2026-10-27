
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

from src.config import BASE_DIR
from src.data.preprocessing import carregar_dados, preparar_dados


def avaliar_modelo():
    dados = carregar_dados("online_shoppers_intention.csv")

    _, X_teste, _, y_teste, _ = preparar_dados(dados)

    mlflow.set_experiment("tech-challenge-fase-2")

    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName = 'random_forest'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError("Treinamento Random Forest não encontrado.")

    run_id = runs.iloc[0]["run_id"]
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")

    previsoes = modelo.predict(X_teste)

    print(classification_report(
        y_teste,
        previsoes,
        target_names=["Sem compra", "Com compra"],
        zero_division=0,
    ))

    pasta_resultados = BASE_DIR / "reports"
    pasta_resultados.mkdir(exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_teste,
        previsoes,
        display_labels=["Sem compra", "Com compra"],
        cmap="Blues",
        values_format="d",
    )

    plt.title("Matriz de Confusão - Random Forest")
    plt.tight_layout()
    plt.savefig(
        pasta_resultados / "matriz_confusao_random_forest.png",
        dpi=150,
    )
    plt.close()

    print("Matriz de confusão salva em reports/")


if __name__ == "__main__":
    avaliar_modelo()
