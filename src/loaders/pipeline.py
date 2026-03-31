#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from ledger_parser import parse_ledger_dataframe
from pdf_loader import PDFLoader
from semantic_indexer import SemanticIndexBuilder, resolve_semantic_paths

"""
Módulo do pipeline de ingestão de dados contábeis.

Este módulo coordena o processo completo de ingestão de PDFs contábeis,
desde o carregamento e parsing dos documentos até a geração de dados
estruturados e índices semânticos para busca textual.

O pipeline executa três etapas principais:
1. Carregamento de PDFs com extração de texto (com OCR quando necessário)
2. Parsing estruturado dos dados contábeis usando expressões regulares
3. Geração opcional de índice vetorial FAISS para matching semântico

Os dados são salvos em arquivos Parquet comprimidos para eficiência.
"""


class IngestionPipeline:
    """
    Pipeline orquestrador para ingestão de dados contábeis de PDFs.

    Esta classe coordena todas as etapas do processamento: carregamento de PDFs,
    extração de texto, parsing estruturado e indexação semântica. Suporta
    configuração flexível de caminhos e parâmetros através de argumentos
    ou variáveis de ambiente.

    Attributes:
        data_dir: Diretório contendo os PDFs de entrada.
        data_processed: Caminho para salvar o Parquet bruto por página.
        data_structured: Caminho para salvar o Parquet estruturado.
        structured_columns: Colunas selecionadas para o output estruturado.
        semantic_* : Configurações para indexação semântica.
        loader: Instância do PDFLoader para carregamento de documentos.
    """

    def __init__(
        self,
        data_dir: str = None,
        data_processed: str = None,
        data_structured: str = None,
        structured_columns: Optional[List[str]] = None,
        semantic_index_path: str | None = None,
        semantic_terms_path: str | None = None,
        semantic_model_name: str | None = None,
        semantic_enabled: bool | None = None,
        semantic_local_files_only: bool | None = None,
    ):
        """
        Inicializa o pipeline com configurações fornecidas.

        Args:
            data_dir: Diretório dos PDFs (padrão: settings.data_dir).
            data_processed: Caminho do Parquet bruto (padrão: settings.data_processed).
            data_structured: Caminho do Parquet estruturado (padrão: settings.data_structured).
            structured_columns: Colunas selecionadas para output estruturado.
            semantic_index_path: Caminho do índice FAISS.
            semantic_terms_path: Caminho do arquivo de termos semânticos.
            semantic_model_name: Nome do modelo de embeddings.
            semantic_enabled: Se deve gerar índice semântico.
            semantic_local_files_only: Usar apenas modelos locais.
        """
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = data_dir or settings.data_dir
        self.data_processed = data_processed or settings.data_processed
        self.data_structured = data_structured or settings.data_structured
        self.structured_columns = structured_columns

        # Resolve caminhos relativos para absolutos
        data_dir_path = Path(self.data_dir)
        if not data_dir_path.is_absolute():
            data_dir_path = project_root / data_dir_path
        self.data_dir = str(data_dir_path)

        data_processed_path = Path(self.data_processed)
        if not data_processed_path.is_absolute():
            data_processed_path = project_root / data_processed_path
        self.data_processed = str(data_processed_path)

        data_structured_path = Path(self.data_structured)
        if not data_structured_path.is_absolute():
            data_structured_path = project_root / data_structured_path
        self.data_structured = str(data_structured_path)

        # Configurações semânticas com fallbacks para variáveis de ambiente
        self.semantic_enabled = (
            semantic_enabled
            if semantic_enabled is not None
            else os.getenv("SEMANTIC_MATCH_ENABLED", "true").lower() == "true"
        )
        self.semantic_model_name = (
            semantic_model_name
            or os.getenv("SEMANTIC_MODEL_NAME")
            or "paraphrase-multilingual-mpnet-base-v2"
        )
        self.semantic_local_files_only = (
            semantic_local_files_only
            if semantic_local_files_only is not None
            else os.getenv("SEMANTIC_LOCAL_FILES_ONLY", "false").lower() == "true"
        )
        self.semantic_index_path, self.semantic_terms_path = resolve_semantic_paths(
            structured_path=self.data_structured,
            semantic_index_path=semantic_index_path,
            semantic_terms_path=semantic_terms_path,
        )

        # Inicializa o loader de PDFs
        self.loader = PDFLoader(self.data_dir)

    def run(self):
        """
        Executa o pipeline completo de ingestão.

        Esta método orquestra todas as etapas:
        1. Carrega PDFs e extrai texto (com OCR se necessário)
        2. Salva dados brutos em Parquet
        3. Aplica parsing estruturado dos lançamentos contábeis
        4. Salva dados estruturados em Parquet
        5. Gera índice semântico se habilitado

        Cada etapa imprime progresso no console.
        """
        print("🚀 Iniciando pipeline de ingestão...\n")
        print("📄 Carregando PDFs...")
        df_pages = self.loader.load()

        # Salva dados brutos por página
        raw_output_path = Path(self.data_processed)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pages.to_parquet(str(raw_output_path), engine="pyarrow", compression="snappy")
        print(f"✅ Parquet bruto (por pagina) salvo em {raw_output_path}")

        print("🧾 Estruturando lancamentos contabeis...")
        df_structured = parse_ledger_dataframe(
            df_pages,
            selected_columns=self.structured_columns,
        )

        # Salva dados estruturados
        structured_output_path = Path(self.data_structured)
        structured_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_structured.to_parquet(
            str(structured_output_path),
            engine="pyarrow",
            compression="snappy",
        )
        print(
            "✅ Parquet estruturado (colunas contabeis) salvo em "
            f"{structured_output_path}"
        )

        print("🧠 Gerando índice semântico para consultas textuais...")
        semantic_builder = SemanticIndexBuilder(
            index_path=self.semantic_index_path,
            terms_path=self.semantic_terms_path,
            model_name=self.semantic_model_name,
            enabled=self.semantic_enabled,
            local_files_only=self.semantic_local_files_only,
        )
        semantic_result = semantic_builder.build(df_structured)
        print(
            "✅ " + semantic_result.message
            if semantic_result.enabled
            else "ℹ️ " + semantic_result.message
        )
