import os
from pathlib import Path
import logging
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"
OPENAI_SSM_PARAMETER_NAME = "/rag-pipeline/openai-api-key"


def _bootstrap_env() -> None:
    """
    Carrega variáveis de ambiente via `.env` quando existir.

    Se `.env` não existir, assume execução na AWS e tenta buscar a chave da OpenAI
    no SSM Parameter Store em `/rag-pipeline/openai-api-key`.
    """
    # 1) Local dev: tenta carregar `.env` do root do projeto.
    if DOTENV_PATH.exists():
        try:
            from dotenv import load_dotenv
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "python-dotenv não está disponível, mas `config/settings.py` "
                "precisa dele para carregar o arquivo .env."
            ) from exc

        load_dotenv(dotenv_path=DOTENV_PATH, override=False)
        return

class Settings(BaseSettings):
    # OpenAI (opcional, só necessário para embeddings/LLM)
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    
    # AWS S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "vector-stores/"
    
    # Pipeline
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-small"
    vector_db_name: str = "vector_db"
    data_dir: str = "data/raw"
    data_processed: str = "data/processed/ingestion.parquet"
    data_structured: str = "data/processed/razao_contabil.parquet"
    
    model_config = SettingsConfigDict(
        env_file=str(DOTENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

_bootstrap_env()
settings = Settings()
