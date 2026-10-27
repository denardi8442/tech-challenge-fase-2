from pathlib import Path

from dotenv import load_dotenv

# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carrega variáveis de ambiente, quando disponíveis
load_dotenv(BASE_DIR / ".env")

# Diretórios do projeto
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Cria os diretórios necessários
for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
