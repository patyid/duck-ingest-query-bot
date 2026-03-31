#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional

"""
Script principal para executar o pipeline de ingestão de PDFs.

Este módulo fornece uma interface de linha de comando (CLI) para processar PDFs localizados
em um diretório especificado, extrair dados estruturados e opcionalmente gerar um índice
semântico vetorial. O resultado é armazenado em arquivos Parquet para análise posterior
no DuckDB.

O pipeline inclui:
- Carregamento e parsing de PDFs com suporte a OCR
- Extração de dados contábeis estruturados
- Geração opcional de índice semântico FAISS para busca textual
"""


def _parse_structured_columns(raw_columns: Optional[str]) -> Optional[List[str]]:
    """
    Converte uma string de colunas separadas por vírgula em uma lista de strings.

    Args:
        raw_columns (Optional[str]): String contendo nomes de colunas separados por vírgula,
                                   ou None se não especificado.

    Returns:
        Optional[List[str]]: Lista de nomes de colunas limpos, ou None se entrada for None.
    """
    if raw_columns is None:
        return None
    return [col.strip() for col in raw_columns.split(",")]


def main():
    """
    Função principal que configura o parser de argumentos e executa o pipeline de ingestão.

    Esta função:
    - Configura o caminho do projeto no sys.path
    - Define os argumentos da CLI com valores padrão
    - Instancia e executa o IngestionPipeline com os parâmetros fornecidos
    """
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipeline import IngestionPipeline

    # Configura o parser de argumentos da linha de comando
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
    parser.add_argument(
        "--semantic-index-path",
        default=None,
        help="Caminho de saída do índice vetorial FAISS (default: data/processed/semantic_terms.faiss).",
    )
    parser.add_argument(
        "--semantic-terms-path",
        default=None,
        help="Caminho do arquivo JSON com termos/metadados do índice semântico.",
    )
    parser.add_argument(
        "--semantic-model-name",
        default=None,
        help="Modelo Sentence Transformers para indexação semântica.",
    )
    parser.add_argument(
        "--semantic-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Habilita/desabilita criação do índice semântico.",
    )
    parser.add_argument(
        "--semantic-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Força uso apenas de modelos locais (sem download).",
    )

    # Faz o parsing dos argumentos fornecidos
    args = parser.parse_args()
    structured_columns = _parse_structured_columns(args.structured_columns)

    # Instancia e executa o pipeline de ingestão com os parâmetros configurados
    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        data_processed=args.data_processed,
        data_structured=args.data_structured,
        structured_columns=structured_columns,
        semantic_index_path=args.semantic_index_path,
        semantic_terms_path=args.semantic_terms_path,
        semantic_model_name=args.semantic_model_name,
        semantic_enabled=args.semantic_enabled,
        semantic_local_files_only=args.semantic_local_files_only,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
