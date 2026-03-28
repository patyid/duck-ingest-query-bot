import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


HEADER_TEXT = "RAZÃO POR CONTA CONTÁBIL"

PERIODO_RE = re.compile(
    r"Período base:\s*(\d{2}/\d{2}/\d{4})\s*a\s*(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)
CNPJ_RE = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")
CONTA_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{2})\s+(.+)$")
DATA_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TOTAL_DEBITO_RE = re.compile(r"^Total débito:\s*([-\d\.,]+)$", re.IGNORECASE)
VALOR_ONLY_RE = re.compile(r"^(-?\d{1,3}(?:\.\d{3})*,\d{2})$")
VALOR_END_RE = re.compile(r"(.+?)\s*(-?\d{1,3}(?:\.\d{3})*,\d{2})$")

STRUCTURED_COLUMNS = [
    "cabecalho",
    "periodo_inicio",
    "periodo_fim",
    "cnpj",
    "conta_codigo",
    "conta_nome",
    "data_lancamento",
    "historico",
    "valor",
    "total_debito",
    "arquivo",
    "source",
    "pagina",
]


def parse_ledger_dataframe(
    df_pages: pd.DataFrame, selected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    resolved_columns = validate_structured_columns(selected_columns)
    rows: List[Dict[str, Any]] = []
    for page_row in df_pages.to_dict(orient="records"):
        rows.extend(_parse_page(page_row))
    return pd.DataFrame(rows, columns=resolved_columns)


def validate_structured_columns(selected_columns: Optional[List[str]]) -> List[str]:
    if selected_columns is None:
        return STRUCTURED_COLUMNS

    cleaned = [col.strip() for col in selected_columns if col and col.strip()]
    if not cleaned:
        raise ValueError(
            "Nenhuma coluna valida foi informada em --structured-columns."
        )

    invalid = [col for col in cleaned if col not in STRUCTURED_COLUMNS]
    if invalid:
        raise ValueError(
            "Coluna(s) invalida(s) em --structured-columns: "
            f"{', '.join(invalid)}. "
            "Colunas permitidas: "
            f"{', '.join(STRUCTURED_COLUMNS)}"
        )

    # remove duplicidade mantendo ordem de entrada
    unique_columns = list(dict.fromkeys(cleaned))
    return unique_columns


def _parse_page(page_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    page_content = str(page_row.get("page_content") or "")
    metadata = page_row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    lines = [line.strip() for line in page_content.splitlines() if line.strip()]
    if not lines:
        return []

    cabecalho = HEADER_TEXT if HEADER_TEXT in page_content else ""
    periodo_inicio, periodo_fim = _extract_periodo(page_content)
    cnpj = _extract_cnpj(page_content)

    source = _as_string(metadata.get("source"))
    arquivo = _as_string(metadata.get("file_path") or source)
    pagina = _as_int(metadata.get("page"))

    current_conta_codigo = ""
    current_conta_nome = ""
    current_data = ""
    current_historico_parts: List[str] = []

    page_rows: List[Dict[str, Any]] = []
    current_conta_row_indexes: List[int] = []

    for line in lines:
        if line in {"Data", "Histórico", "Valor"}:
            continue
        if line in {"1/4", "2/4", "3/4", "4/4"}:
            continue
        if re.match(r"^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}$", line):
            continue

        conta_match = CONTA_RE.match(line)
        if conta_match:
            current_conta_codigo = conta_match.group(1)
            current_conta_nome = conta_match.group(2).strip()
            current_conta_row_indexes = []
            current_data = ""
            current_historico_parts = []
            continue

        total_match = TOTAL_DEBITO_RE.match(line)
        if total_match:
            total_debito_num = _parse_currency_br(total_match.group(1))
            for idx in current_conta_row_indexes:
                page_rows[idx]["total_debito"] = total_debito_num
            current_data = ""
            current_historico_parts = []
            continue

        if DATA_RE.match(line):
            current_data = line
            current_historico_parts = []
            continue

        if not current_data:
            continue

        valor_only = VALOR_ONLY_RE.match(line)
        if valor_only:
            _append_row(
                page_rows=page_rows,
                current_conta_row_indexes=current_conta_row_indexes,
                cabecalho=cabecalho,
                periodo_inicio=periodo_inicio,
                periodo_fim=periodo_fim,
                cnpj=cnpj,
                conta_codigo=current_conta_codigo,
                conta_nome=current_conta_nome,
                data_lancamento=current_data,
                historico=" ".join(current_historico_parts).strip(),
                valor_str=valor_only.group(1),
                arquivo=arquivo,
                source=source,
                pagina=pagina,
            )
            current_data = ""
            current_historico_parts = []
            continue

        valor_end = VALOR_END_RE.match(line)
        if valor_end:
            historico_prefix = valor_end.group(1).strip()
            if historico_prefix:
                current_historico_parts.append(historico_prefix)
            _append_row(
                page_rows=page_rows,
                current_conta_row_indexes=current_conta_row_indexes,
                cabecalho=cabecalho,
                periodo_inicio=periodo_inicio,
                periodo_fim=periodo_fim,
                cnpj=cnpj,
                conta_codigo=current_conta_codigo,
                conta_nome=current_conta_nome,
                data_lancamento=current_data,
                historico=" ".join(current_historico_parts).strip(),
                valor_str=valor_end.group(2),
                arquivo=arquivo,
                source=source,
                pagina=pagina,
            )
            current_data = ""
            current_historico_parts = []
            continue

        current_historico_parts.append(line)

    return page_rows


def _append_row(
    page_rows: List[Dict[str, Any]],
    current_conta_row_indexes: List[int],
    cabecalho: str,
    periodo_inicio: str,
    periodo_fim: str,
    cnpj: str,
    conta_codigo: str,
    conta_nome: str,
    data_lancamento: str,
    historico: str,
    valor_str: str,
    arquivo: str,
    source: str,
    pagina: Optional[int],
) -> None:
    row = {
        "cabecalho": cabecalho,
        "periodo_inicio": periodo_inicio,
        "periodo_fim": periodo_fim,
        "cnpj": cnpj,
        "conta_codigo": conta_codigo,
        "conta_nome": conta_nome,
        "data_lancamento": data_lancamento,
        "historico": historico,
        "valor": _parse_currency_br(valor_str),
        "total_debito": None,
        "arquivo": arquivo,
        "source": source,
        "pagina": pagina,
    }
    page_rows.append(row)
    current_conta_row_indexes.append(len(page_rows) - 1)


def _extract_periodo(text: str) -> Tuple[str, str]:
    match = PERIODO_RE.search(text)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def _extract_cnpj(text: str) -> str:
    match = CNPJ_RE.search(text)
    return match.group(0) if match else ""


def _parse_currency_br(value: str) -> Optional[float]:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    normalized = v.replace(".", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
