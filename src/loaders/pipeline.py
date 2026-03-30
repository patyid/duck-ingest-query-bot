#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from ledger_parser import parse_ledger_dataframe
from pdf_loader import PDFLoader
from semantic_indexer import SemanticIndexBuilder, resolve_semantic_paths


class IngestionPipeline:
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
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = data_dir or settings.data_dir
        self.data_processed = data_processed or settings.data_processed
        self.data_structured = data_structured or settings.data_structured
        self.structured_columns = structured_columns

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

        self.loader = PDFLoader(self.data_dir)

    def run(self):
        """Executa o pipeline completo."""
        print("🚀 Iniciando pipeline de ingestão...\n")
        print("📄 Carregando PDFs...")
        df_pages = self.loader.load()

        raw_output_path = Path(self.data_processed)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pages.to_parquet(str(raw_output_path), engine="pyarrow", compression="snappy")
        print(f"✅ Parquet bruto (por pagina) salvo em {raw_output_path}")

        print("🧾 Estruturando lancamentos contabeis...")
        df_structured = parse_ledger_dataframe(
            df_pages,
            selected_columns=self.structured_columns,
        )
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
