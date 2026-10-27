
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data.preprocessing import carregar_dados, preparar_dados


def treinar_random_forest():
    dados = carregar_dados("online_shoppers_intention.csv")

    X_treino, X_teste, y_treino, y_teste, preprocessador = (
        preparar_dados(dados)
    )

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        ("classificador", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    mlflow.set_experiment("tech-challenge-fase-2")

    with mlflow.start_run(run_name="random_forest"):
        modelo.fit(X_treino, y_treino)

        previsoes = modelo.predict(X_teste)
        probabilidades = modelo.predict_proba(X_teste)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_teste, previsoes),
            "precision": precision_score(y_teste, previsoes),
            "recall": recall_score(y_teste, previsoes),
            "f1_score": f1_score(y_teste, previsoes),
            "roc_auc": roc_auc_score(y_teste, probabilidades),
        }

        mlflow.log_params({
            "modelo": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced",
            "test_size": 0.20,
        })

        mlflow.log_metrics(metricas)

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            serialization_format="cloudpickle",
        )

        for nome, valor in metricas.items():
            print(f"{nome}: {valor:.4f}")

    return modelo, metricas


if __name__ == "__main__":
    treinar_random_forest()
