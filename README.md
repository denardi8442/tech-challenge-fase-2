# Tech Challenge — Fase 2

## Previsão de Intenção de Compra em E-commerce

Projeto desenvolvido individualmente como parte do Tech Challenge da Pós-Graduação em Machine Learning Engineering da FIAP.

O objetivo é desenvolver um pipeline de Machine Learning para prever a intenção de compra de usuários em um ambiente de e-commerce, utilizando informações relacionadas ao comportamento de navegação.

O projeto contempla preparação dos dados, treinamento e avaliação de modelos, versionamento de dados, gerenciamento de dependências, containerização, rastreamento de experimentos e registro do melhor modelo.

## 1. Objetivo

Desenvolver um modelo de classificação capaz de identificar a propensão de compra de usuários de um e-commerce.

A variável-alvo é `Revenue`, que indica se uma sessão de navegação resultou em uma compra.

Foram implementados dois algoritmos:

- **Regressão Logística:** utilizada como modelo baseline.
- **Random Forest:** utilizado para comparação de desempenho e seleção do modelo final.

## 2. Tecnologias utilizadas

- **Python 3.13:** linguagem de programação.
- **Scikit-learn:** preparação dos dados, treinamento e avaliação dos modelos.
- **Pandas:** leitura e manipulação dos dados.
- **MLflow:** rastreamento de experimentos, métricas e registro de modelos.
- **DVC:** versionamento dos dados e gerenciamento do pipeline.
- **Poetry:** gerenciamento de dependências e ambientes virtuais.
- **Docker:** execução da aplicação em contêiner.
- **Git e GitHub:** versionamento e disponibilização do código-fonte.
- **Pytest:** execução de testes automatizados.
- **GitHub Actions:** execução automática dos testes em ambiente independente.

## 3. Dataset

Foi utilizado o **Online Shoppers Purchasing Intention Dataset**, conjunto de dados voltado à análise do comportamento de navegação de consumidores em plataformas de comércio eletrônico.

O dataset contém **12.330 registros**, e a variável `Revenue` representa o resultado da sessão de navegação quanto à realização de uma compra.

O arquivo original é controlado pelo DVC e está disponível publicamente em uma release do GitHub.

**Download do dataset:**

https://github.com/denardi8442/tech-challenge-fase-2/releases/tag/dataset-v1

**Arquivo:**

`online_shoppers_intention.csv`

O dataset possui aproximadamente 1,02 MB.

## 4. Estrutura do projeto

```text
tech-challenge-fase-2/
├── .dvc/
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv.dvc
│   └── processed/
├── reports/
│   └── matriz_confusao_random_forest.png
├── src/
│   ├── config.py
│   ├── data/
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_random_forest.py
│   └── evaluation/
│       └── evaluate.py
├── tests/
│   ├── test_data.py
│   ├── test_imports.py
│   └── test_model.py
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

Os arquivos de dados processados, os bancos locais do MLflow e os modelos gerados durante o treinamento não são necessariamente armazenados diretamente no GitHub. Esses elementos podem ser produzidos novamente por meio dos comandos documentados.

## 5. Instalação do projeto

### 5.1. Pré-requisitos

Para executar o projeto, são necessários:

- Git.
- Python 3.13.
- Poetry.
- Conexão com a internet para instalar as dependências e baixar o dataset.
- Docker, caso seja utilizada a execução em contêiner.

### 5.2. Clonar o repositório

Abra um terminal e execute:

```bash
git clone https://github.com/denardi8442/tech-challenge-fase-2.git
cd tech-challenge-fase-2
```

### 5.3. Instalar as dependências

Execute:

```bash
poetry install --no-root
```

O parâmetro `--no-root` permite instalar as dependências sem tentar instalar o próprio projeto como um pacote Python.

As bibliotecas e suas versões são gerenciadas pelos arquivos `pyproject.toml` e `poetry.lock`.

Esse procedimento foi validado em uma nova clonagem do repositório.

## 6. Recuperação do dataset

O dataset original não é armazenado diretamente no histórico convencional do Git.

Para recuperá-lo, execute:

```bash
poetry run python download_dataset.py
```

O script realiza o download a partir da release pública do GitHub e salva o arquivo em:

```text
data/raw/online_shoppers_intention.csv
```

Caso o arquivo já exista, o download será ignorado.

### 6.1. Versionamento com DVC

O arquivo `online_shoppers_intention.csv.dvc` contém as informações utilizadas pelo DVC para identificar a versão do dataset.

Para verificar o estado dos dados e do pipeline:

```bash
poetry run dvc status
```

O armazenamento DVC utilizado durante o desenvolvimento foi local. Para permitir a reprodução do projeto por terceiros, o dataset foi disponibilizado em uma release pública e pode ser recuperado pelo script `download_dataset.py`.

**Importante:** a recuperação do dataset em outro computador não depende do comando `dvc pull`, pois não foi configurado um armazenamento DVC remoto compartilhado.

## 7. Pré-processamento dos dados

O pré-processamento é realizado pelo módulo `src/data/preprocessing.py`.

O processo contempla:

- Leitura do dataset original.
- Verificação da variável-alvo `Revenue`.
- Preparação das variáveis utilizadas no treinamento.
- Separação estratificada dos dados de treinamento e teste.
- Tratamento de valores ausentes.
- Padronização das variáveis numéricas.
- Codificação das variáveis categóricas.

A divisão dos dados utiliza 80% para treinamento e 20% para teste, com `random_state=42`.

As transformações utilizadas pelos modelos são organizadas com `Pipeline` e `ColumnTransformer` do Scikit-learn.

## 8. Pipeline DVC

O pipeline está definido no arquivo `dvc.yaml` e possui duas etapas principais.

**Pré-processamento:** realiza a leitura do dataset original e gera o arquivo:

```text
data/processed/online_shoppers_processed.csv
```

**Treinamento:** utiliza o dataset preparado para treinar a Regressão Logística e registrar os resultados no MLflow.

### 8.1. Executar o pipeline

Após instalar as dependências e baixar o dataset, execute:

```bash
poetry run dvc repro
```

O DVC verifica as dependências e executa as etapas necessárias.

Para executar novamente as etapas, mesmo quando o DVC considerar que não houve alterações:

```bash
poetry run dvc repro --force
```

O arquivo `dvc.lock` registra informações sobre as dependências e saídas do pipeline, contribuindo para a rastreabilidade das execuções.

**Observação:** o Random Forest é executado separadamente e não faz parte das duas etapas atualmente declaradas no `dvc.yaml`.

## 9. Treinamento dos modelos

### 9.1. Regressão Logística

A Regressão Logística foi utilizada como baseline para estabelecer uma referência inicial de desempenho.

Para executar o treinamento:

```bash
poetry run python main.py
```

O modelo utiliza:

- `max_iter=1000`
- `random_state=42`
- `class_weight="balanced"`

O treinamento registra parâmetros, métricas e o modelo no MLflow.

### 9.2. Random Forest

O Random Forest foi utilizado como segundo algoritmo para comparar os resultados com o baseline.

Para executar:

```bash
poetry run python -m src.training.train_random_forest
```

O modelo utiliza:

- `n_estimators=200`
- `max_depth=10`
- `random_state=42`
- `class_weight="balanced"`
- `n_jobs=-1`

O treinamento calcula as métricas de avaliação, registra os resultados no MLflow e cria uma versão do modelo no Model Registry.

**Importante:** execute primeiro o pré-processamento pelo DVC, pois os scripts de treinamento utilizam o arquivo preparado.

## 10. Avaliação e resultados

Foram utilizadas as seguintes métricas:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Matriz de confusão.

### 10.1. Comparação dos modelos

| Métrica | Regressão Logística | Random Forest |
|---|---:|---:|
| Accuracy | 85,00% | 86,62% |
| Precision | 51,07% | 54,69% |
| Recall | 74,87% | 79,32% |
| F1-score | 60,72% | 64,74% |
| ROC-AUC | 89,62% | 92,40% |

O **Random Forest apresentou os melhores resultados em todas as métricas avaliadas**, sendo selecionado como modelo final.

O F1-score passou de 60,72% para 64,74%, enquanto o ROC-AUC aumentou de 89,62% para 92,40%.

Esses resultados indicam uma melhoria na capacidade de classificação e discriminação entre sessões com e sem compra.

### 10.2. Matriz de confusão

A matriz de confusão do Random Forest está disponível em:

```text
reports/matriz_confusao_random_forest.png
```

Ela permite analisar os acertos e erros do modelo, incluindo falsos positivos e falsos negativos.

## 11. MLflow e Model Registry

O MLflow foi utilizado para registrar os experimentos, os parâmetros, as métricas e os artefatos dos modelos.

Os experimentos foram organizados com o nome:

`tech-challenge-fase-2`

O Random Forest foi registrado no **MLflow Model Registry** com as seguintes informações:

- **Nome:** `modelo_propensao_compra`
- **Algoritmo:** Random Forest
- **Versão inicial:** 1

### 11.1. Configurar o armazenamento do MLflow

Para reproduzir o treinamento utilizando um banco SQLite local, configure a variável de ambiente `MLFLOW_TRACKING_URI`.

No PowerShell do Windows:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
```

No Linux ou macOS:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

Em seguida, execute os scripts de treinamento.

### 11.2. Abrir a interface do MLflow

Execute:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse no navegador:

http://127.0.0.1:5000

Na interface do MLflow, selecione **Model training** para acessar as seções **Experiments** e **Model registry**.

Em **Model registry**, será possível visualizar o modelo `modelo_propensao_compra` e suas versões, desde que ele tenha sido registrado no banco utilizado pela interface.

### 11.3. Reprodutibilidade dos experimentos

O banco SQLite e os artefatos locais do MLflow não são enviados automaticamente ao GitHub.

Para reproduzir os resultados, o avaliador deve executar os treinamentos utilizando a configuração indicada anteriormente.

Esse procedimento foi validado em uma cópia independente do projeto, incluindo a criação do experimento e o registro da primeira versão do Random Forest.

## 12. Execução com Docker

O projeto possui um `Dockerfile` e um arquivo `compose.yaml` para executar o treinamento em contêiner.

### 12.1. Construir a imagem

Com o Docker em execução, utilize:

```bash
docker build -t tech-challenge-fase-2 .
```

### 12.2. Executar com Docker Compose

Após recuperar o dataset, execute:

```bash
docker compose up --build
```

O Docker Compose configura o contêiner de treinamento e os volumes utilizados para disponibilizar os dados e armazenar os registros do MLflow.

Na configuração atual, o contêiner executa o treinamento da Regressão Logística por meio do `main.py`.

**Importante:** o dataset deve estar disponível na pasta `data/raw/` antes de iniciar o contêiner.

O Docker foi validado durante o desenvolvimento do projeto. A reprodução completa dos dois modelos também foi validada diretamente com Poetry e DVC.

## 13. Testes automatizados

O projeto utiliza Pytest para validar funcionalidades básicas.

Os testes estão organizados nos arquivos:

- `tests/test_data.py`: existência, integridade básica e variável-alvo do dataset.
- `tests/test_imports.py`: importação dos módulos do projeto.
- `tests/test_model.py`: treinamento e previsões de um modelo de classificação.

### 13.1. Executar os testes localmente

Após baixar o dataset, execute:

```bash
poetry run python -m pytest -v
```

Durante a validação, **os oito testes foram aprovados**.

### 13.2. GitHub Actions

O repositório possui um workflow de testes automatizados:

```text
.github/workflows/tests.yml
```

O workflow realiza:

1. Download do repositório.
2. Configuração do Python 3.13.
3. Instalação do Poetry.
4. Instalação das dependências.
5. Download do dataset pela release pública.
6. Execução dos testes automatizados.

A execução foi concluída com sucesso no GitHub Actions, demonstrando que os testes funcionam em um ambiente independente do computador utilizado no desenvolvimento.

## 14. Reprodutibilidade

A reprodutibilidade do projeto é apoiada pelos seguintes recursos:

- Dependências gerenciadas pelo Poetry e registradas no `poetry.lock`.
- Código-fonte organizado em módulos.
- Sementes fixadas para treinamento e divisão dos dados.
- Dataset versionado pelo DVC.
- Pipeline documentado em `dvc.yaml` e `dvc.lock`.
- Dataset disponibilizado em uma release pública do GitHub.
- Experimentos e modelos rastreados pelo MLflow.
- Ambiente de execução configurado com Docker.
- Testes automatizados executados pelo GitHub Actions.

### 14.1. Validação em ambiente independente

A reprodutibilidade foi verificada por meio de uma nova clonagem do repositório em uma pasta separada do projeto original.

Foram realizados os seguintes procedimentos:

1. Clonagem do repositório GitHub.
2. Instalação das dependências com `poetry install --no-root`.
3. Download do dataset utilizando `download_dataset.py`.
4. Execução do pré-processamento pelo DVC.
5. Execução do treinamento da Regressão Logística.
6. Execução do treinamento do Random Forest.
7. Registro do Random Forest no MLflow Model Registry.

Os dois modelos reproduziram as mesmas métricas obtidas no desenvolvimento original.

Também foi validada a execução dos testes automatizados no GitHub Actions.

### 14.2. Comandos para reproduzir o projeto

Os comandos abaixo apresentam a sequência recomendada para reproduzir os resultados em um ambiente novo.

**Clonar o projeto:**

```bash
git clone https://github.com/denardi8442/tech-challenge-fase-2.git
cd tech-challenge-fase-2
```

**Instalar as dependências:**

```bash
poetry install --no-root
```

**Baixar o dataset:**

```bash
poetry run python download_dataset.py
```

**Configurar o MLflow:**

No PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
```

No Linux ou macOS:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

**Executar o pipeline DVC:**

```bash
poetry run dvc repro --force
```

**Treinar e registrar o Random Forest:**

```bash
poetry run python -m src.training.train_random_forest
```

**Executar os testes:**

```bash
poetry run python -m pytest -v
```

**Visualizar os resultados no MLflow:**

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse `http://127.0.0.1:5000`.

O pipeline DVC executa o pré-processamento e o treinamento da Regressão Logística. O Random Forest é executado separadamente e registrado no MLflow Model Registry.

## 15. Entrega

O projeto foi desenvolvido individualmente como atividade acadêmica da Fase 2 da Pós-Graduação em Machine Learning Engineering da FIAP.

A entrega contempla o repositório GitHub com o código-fonte, os arquivos de configuração, o pipeline DVC, os testes automatizados e a documentação necessária para reprodução.

Também será preparado um vídeo demonstrativo de aproximadamente cinco minutos, estruturado pelo método STAR, apresentando o problema, as atividades realizadas, as soluções implementadas e os resultados obtidos.

**Repositório GitHub:**

https://github.com/denardi8442/tech-challenge-fase-2

**Dataset:**

https://github.com/denardi8442/tech-challenge-fase-2/releases/tag/dataset-v1

**Status:** Implementação e validação técnica concluídas. Preparação da apresentação final em andamento.
