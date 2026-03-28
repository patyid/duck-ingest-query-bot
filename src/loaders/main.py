#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _parse_structured_columns(raw_columns: Optional[str]) -> Optional[List[str]]:
    if raw_columns is None:
        return None
    return [col.strip() for col in raw_columns.split(",")]


def main():
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipeline import IngestionPipeline

    parser = argparse.ArgumentParser(description="Pipeline de injestão de PDFs para duckdb")
    parser.add_argument("--data-dir", default="data/raw", help="Diretório dos PDFs")
    parser.add_argument(
        "--data-processed",
        default="data/processed/ingestion.parquet",
        help="Parquet bruto por página",
    )
    parser.add_argument(
        "--data-structured",
        default="data/processed/razao_contabil.parquet",
        help="Parquet estruturado para query analitica",
    )
    parser.add_argument(
        "--structured-columns",
        default=None,
        help=(
            "Lista de colunas do parquet estruturado separadas por virgula. "
            "Ex.: cabecalho,periodo_inicio,cnpj,conta_codigo,data_lancamento,historico,valor,total_debito"
        ),
    )

    args = parser.parse_args()
    structured_columns = _parse_structured_columns(args.structured_columns)

    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        data_processed=args.data_processed,
        data_structured=args.data_structured,
        structured_columns=structured_columns,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
