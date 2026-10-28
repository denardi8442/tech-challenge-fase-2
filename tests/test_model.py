import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def test_modelo_treinamento():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
    ])

    y = np.array([0, 0, 1, 1])

    modelo = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    modelo.fit(X, y)

    assert hasattr(modelo, "classes_")


def test_modelo_previsao():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
    ])

    y = np.array([0, 0, 1, 1])

    modelo = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    assert len(previsoes) == len(y)
    assert set(previsoes).issubset({0, 1})


def test_modelo_acuracia():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
    ])

    y = np.array([0, 0, 1, 1])

    modelo = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    modelo.fit(X, y)

    previsoes = modelo.predict(X)
    acuracia = accuracy_score(y, previsoes)

    assert 0 <= acuracia <= 1
