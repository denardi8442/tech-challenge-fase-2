# Tech Challenge — Fase 2

## Previsão de Intenção de Compra em E-commerce

Projeto desenvolvido para o Tech Challenge da Pós-Graduação em Machine Learning Engineering da FIAP.

O projeto utiliza técnicas de Machine Learning para prever a intenção de compra de usuários em um ambiente de e-commerce, considerando informações relacionadas ao comportamento de navegação.

O foco principal é a construção de um pipeline reproduzível, utilizando boas práticas de Engenharia de Machine Learning, versionamento de dados, gerenciamento de dependências, containerização e rastreamento de experimentos.

## 1. Objetivo

Desenvolver um pipeline de classificação capaz de identificar a propensão de compra de usuários de um e-commerce.

A variável-alvo é `Revenue`, que indica se uma sessão de navegação resultou em compra.

Foram implementados dois modelos de classificação:

- Regressão Logística, utilizada como baseline.
- Random Forest, utilizado para comparação e seleção do melhor modelo.

## 2. Tecnologias utilizadas

- **Python:** linguagem de programação.
- **Scikit-learn:** pré-processamento, treinamento e avaliação dos modelos.
- **MLflow:** rastreamento de experimentos, métricas e registro de modelos.
- **DVC:** versionamento de dados e gerenciamento do pipeline.
- **Poetry:** gerenciamento de dependências e ambiente virtual.
- **Docker:** execução da aplicação em contêiner.
- **Git e GitHub:** versionamento e disponibilização do código.

## 3. Dataset

Foi utilizado o **Online Shoppers Purchasing Intention Dataset**, conjunto de dados voltado à análise do comportamento de navegação de consumidores.

O dataset possui 12.330 registros e utiliza a variável `Revenue` como alvo da classificação.

O arquivo original é controlado pelo DVC e também está disponibilizado em uma release pública do GitHub.

**Download do dataset:**

https://github.com/denardi8442/tech-challenge-fase-2/releases/tag/dataset-v1

## 4. Estrutura do projeto

```text
tech-challenge-fase-2/
├── .github/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_random_forest.py
│   └── evaluation/
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
├── compose.yaml
├── Dockerfile
├── download_dataset.py
├── dvc.yaml
├── dvc.lock
├── main.py
├── poetry.lock
├── pyproject.toml
└── README.md
```

## 5. Instalação do projeto

### 5.1. Pré-requisitos

Para executar o projeto localmente, são necessários:

- Git
- Python 3.13
- Poetry
- Docker, caso seja utilizada a execução em contêiner

### 5.2. Clonar o repositório

```bash
git clone https://github.com/denardi8442/tech-challenge-fase-2.git
cd tech-challenge-fase-2
```

### 5.3. Instalar as dependências

```bash
poetry install
```

As dependências estão definidas no arquivo `pyproject.toml`, enquanto o `poetry.lock` permite reproduzir as versões utilizadas no desenvolvimento.

## 6. Recuperação do dataset

O arquivo CSV não é armazenado diretamente no histórico convencional do Git.

Para recuperá-lo, execute:

```bash
poetry run python download_dataset.py
```

O script baixa o dataset da release pública do GitHub e o salva em:

```text
data/raw/online_shoppers_intention.csv
```

Caso o arquivo já exista, o download será ignorado.

O arquivo `.dvc` permite verificar a versão do dataset utilizada pelo projeto.

Para conferir o estado dos dados e do pipeline:

```bash
poetry run dvc status
```

**Observação:** o armazenamento remoto DVC configurado durante o desenvolvimento é local. Por isso, a recuperação dos dados em outra máquina é realizada pelo script de download, e não pelo comando `dvc pull`.

## 7. Execução do pipeline DVC

O pipeline está definido no arquivo `dvc.yaml` e possui duas etapas:

1. **Pré-processamento:** leitura do dataset original e geração do arquivo intermediário.
2. **Treinamento:** treinamento da Regressão Logística e registro do experimento no MLflow.

Para executar o pipeline completo:

```bash
poetry run dvc repro
```

O DVC utiliza o arquivo `dvc.lock` para acompanhar as dependências e as saídas do pipeline, permitindo identificar quando uma etapa precisa ser executada novamente.

O arquivo intermediário é gerado em:

```text
data/processed/online_shoppers_processed.csv
```

## 8. Treinamento dos modelos

### 8.1. Regressão Logística

Para executar o modelo baseline:

```bash
poetry run python main.py
```

### 8.2. Random Forest

Para executar o Random Forest:

```bash
poetry run python -m src.training.train_random_forest
```

Os modelos utilizam divisão estratificada dos dados, com 80% para treinamento e 20% para teste, além de uma semente fixa (`random_state=42`) para garantir a reprodutibilidade dos resultados.

O pré-processamento do Scikit-learn inclui tratamento de valores ausentes, padronização de variáveis numéricas e codificação de variáveis categóricas.

## 9. Resultados obtidos

| Métrica | Regressão Logística | Random Forest |
|---|---:|---:|
| Accuracy | 85,00% | 86,62% |
| Precision | 51,07% | 54,69% |
| Recall | 74,87% | 79,32% |
| F1-score | 60,72% | 64,74% |
| ROC-AUC | 89,62% | 92,40% |

O **Random Forest apresentou os melhores resultados** nas métricas avaliadas e foi selecionado como modelo final.

Além das métricas, foi gerada uma matriz de confusão para apoiar a análise dos resultados.

## 10. MLflow e Model Registry

O MLflow é utilizado para registrar os parâmetros, as métricas e os artefatos dos experimentos.

O melhor modelo foi registrado no Model Registry com as seguintes informações:

- **Nome:** `modelo_propensao_compra`
- **Algoritmo:** Random Forest
- **Versão inicial:** 1

### Iniciar a interface do MLflow

O projeto foi executado utilizando um banco SQLite local para o rastreamento e o registro dos modelos.

Para iniciar a interface:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Depois, acesse:

http://127.0.0.1:5000

Na interface, utilize as seções **Experiments** e **Model registry** para visualizar os resultados.

O banco SQLite e os artefatos locais do MLflow não são versionados no GitHub. Para gerar novos experimentos e registrar o modelo em outra máquina, execute novamente os scripts de treinamento.

## 11. Execução com Docker

O projeto possui um `Dockerfile` configurado para instalar as dependências e executar o treinamento.

Para construir a imagem:

```bash
docker build -t tech-challenge-fase-2 .
```

Para executar utilizando o Docker Compose:

```bash
docker compose up --build
```

O arquivo `compose.yaml` configura a execução do treinamento e os volumes utilizados para os dados e os registros do MLflow.

**Importante:** na configuração atual, o Docker utiliza os dados disponíveis na pasta local `data/`. Portanto, o dataset deve ser recuperado antes da execução do contêiner.

## 12. Testes automatizados

O projeto possui testes desenvolvidos com Pytest para verificar importações, integridade dos dados e funcionamento básico de um modelo de classificação.

Para executar os testes:

```bash
poetry run python -m pytest -v
```

Os testes que utilizam o dataset dependem da recuperação prévia do arquivo CSV.

## 13. Reprodutibilidade

A reprodutibilidade do projeto é apoiada pelos seguintes recursos:

- Dependências gerenciadas pelo Poetry e registradas no `poetry.lock`.
- Código-fonte organizado em módulos.
- Seeds fixadas para treinamento e divisão dos dados.
- Dataset versionado pelo DVC.
- Pipeline documentado em `dvc.yaml` e