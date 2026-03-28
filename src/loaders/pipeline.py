#!/usr/bin/env python3
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from ledger_parser import parse_ledger_dataframe
from pdf_loader import PDFLoader


class IngestionPipeline:
    def __init__(
        self,
        data_dir: str = None,
        data_processed: str = None,
        data_structured: str = None,
        structured_columns: Optional[List[str]] = None,
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
