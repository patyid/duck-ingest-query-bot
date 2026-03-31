import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

"""
Módulo para parsing de dados de razão contábil a partir de texto extraído de PDFs.

Este módulo implementa um parser baseado em expressões regulares e máquina de estados
para extrair informações estruturadas de documentos contábeis. O algoritmo processa
linha por linha o texto de cada página, identificando cabeçalhos, períodos, CNPJs,
contas contábeis, datas de lançamento, históricos e valores.

O parsing segue uma lógica de estado onde:
- Cada conta contábil (código e nome) inicia um novo contexto
- Datas de lançamento agrupam lançamentos subsequentes
- Históricos podem se estender por múltiplas linhas até encontrar um valor
- Valores são extraídos no formato brasileiro (1.234,56)
- Totais de débito são associados retroativamente aos lançamentos da conta

Expressões regulares são usadas para identificar padrões específicos de cada campo.
"""

# Constantes para identificação de elementos no texto
HEADER_TEXT = "RAZÃO POR CONTA CONTÁBIL"  # Texto que identifica o cabeçalho do documento

# Regex para extrair o período base (formato: "Período base: DD/MM/YYYY a DD/MM/YYYY")
PERIODO_RE = re.compile(
    r"Período base:\s*(\d{2}/\d{2}/\d{4})\s*a\s*(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

# Regex para extrair CNPJ (formato: XX.XXX.XXX/XXXX-XX)
CNPJ_RE = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")

# Regex para identificar linhas de conta contábil (formato: XX.XX.XX Nome da Conta)
CONTA_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{2})\s+(.+)$")

# Regex para identificar datas de lançamento (formato: DD/MM/YYYY)
DATA_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")

# Regex para identificar totais de débito (formato: "Total débito: 1.234,56")
TOTAL_DEBITO_RE = re.compile(r"^Total débito:\s*([-\d\.,]+)$", re.IGNORECASE)

# Regex para valores que aparecem sozinhos em uma linha (formato brasileiro: -1.234,56)
VALOR_ONLY_RE = re.compile(r"^(-?\d{1,3}(?:\.\d{3})*,\d{2})$")

# Regex para linhas que terminam com valor (texto seguido de valor)
VALOR_END_RE = re.compile(r"(.+?)\s*(-?\d{1,3}(?:\.\d{3})*,\d{2})$")

# Colunas padrão do DataFrame estruturado resultante
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
    """
    Converte um DataFrame de páginas em um DataFrame estruturado de lançamentos contábeis.

    Processa cada página do DataFrame de entrada, aplicando o parsing linha por linha
    para extrair informações estruturadas de razão contábil. O resultado é um DataFrame
    com colunas padronizadas contendo dados de lançamentos, contas, valores, etc.

    Args:
        df_pages (pd.DataFrame): DataFrame com colunas 'page_content' e 'metadata'
                               contendo o texto extraído e metadados de cada página.
        selected_columns (Optional[List[str]]): Lista de colunas desejadas no output.
                                               Se None, usa todas as colunas padrão.

    Returns:
        pd.DataFrame: DataFrame estruturado com os lançamentos contábeis extraídos.
    """
    resolved_columns = validate_structured_columns(selected_columns)
    rows: List[Dict[str, Any]] = []
    for page_row in df_pages.to_dict(orient="records"):
        rows.extend(_parse_page(page_row))
    return pd.DataFrame(rows, columns=resolved_columns)


def validate_structured_columns(selected_columns: Optional[List[str]]) -> List[str]:
    """
    Valida e processa a lista de colunas selecionadas para o output estruturado.

    Verifica se as colunas fornecidas são válidas comparando com a lista de colunas
    permitidas. Remove duplicatas mantendo a ordem original e lança erro se colunas
    inválidas forem encontradas.

    Args:
        selected_columns (Optional[List[str]]): Lista de nomes de colunas desejadas.

    Returns:
        List[str]: Lista validada e deduplicada de nomes de colunas.

    Raises:
        ValueError: Se colunas inválidas forem fornecidas ou lista vazia.
    """
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
    """
    Processa uma única página de texto extraído, aplicando o algoritmo de parsing.

    Esta função implementa a máquina de estados principal do parser:
    - Extrai informações globais da página (cabeçalho, período, CNPJ)
    - Processa linha por linha mantendo estado de conta atual, data e histórico
    - Identifica padrões de conta, data, valor e total usando regex
    - Constrói lista de dicionários representando lançamentos estruturados

    Args:
        page_row (Dict[str, Any]): Dicionário com 'page_content' (str) e 'metadata' (dict)
                                 da página a ser processada.

    Returns:
        List[Dict[str, Any]]: Lista de lançamentos extraídos da página, cada um como dict.
    """
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

    # Variáveis de estado para acompanhar o contexto atual durante o parsing
    current_conta_codigo = ""  # Código da conta atual (ex: "01.01.01")
    current_conta_nome = ""    # Nome da conta atual
    current_data = ""          # Data de lançamento atual
    current_historico_parts: List[str] = []  # Partes do histórico sendo acumuladas

    page_rows: List[Dict[str, Any]] = []  # Lista de lançamentos da página
    current_conta_row_indexes: List[int] = []  # Índices dos lançamentos da conta atual

    for line in lines:
        # Pula linhas de cabeçalho de tabela
        if line in {"Data", "Histórico", "Valor"}:
            continue
        # Pula indicadores de página do PDF
        if line in {"1/4", "2/4", "3/4", "4/4"}:
            continue
        # Pula timestamps de geração do PDF
        if re.match(r"^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}$", line):
            continue

        # Verifica se é uma linha de conta contábil (reinicia contexto)
        # Verifica se é uma linha de conta contábil (reinicia contexto)
        conta_match = CONTA_RE.match(line)
        if conta_match:
            current_conta_codigo = conta_match.group(1)
            current_conta_nome = conta_match.group(2).strip()
            current_conta_row_indexes = []  # Reinicia índices para nova conta
            current_data = ""
            current_historico_parts = []
            continue

        # Verifica se é um total de débito (atribui retroativamente aos lançamentos da conta)
        # Verifica se é um total de débito (atribui retroativamente aos lançamentos da conta)
        total_match = TOTAL_DEBITO_RE.match(line)
        if total_match:
            total_debito_num = _parse_currency_br(total_match.group(1))
            for idx in current_conta_row_indexes:
                page_rows[idx]["total_debito"] = total_debito_num
            current_data = ""
            current_historico_parts = []
            continue

        # Verifica se é uma data de lançamento (reinicia histórico)
        # Verifica se é uma data de lançamento (reinicia histórico)
        if DATA_RE.match(line):
            current_data = line
            current_historico_parts = []
            continue

        # Pula linhas antes da primeira data
        if not current_data:
            continue

        # Verifica se é um valor sozinho (lançamento sem histórico adicional)
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

        # Verifica se linha termina com valor (histórico seguido de valor)
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

        # Se não é valor, acumula como parte do histórico
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
    """
    Adiciona um novo lançamento à lista de linhas da página.

    Cria um dicionário com todas as informações estruturadas do lançamento
    e o adiciona à lista page_rows. Também registra o índice na lista de
    lançamentos da conta atual para associação posterior de totais.

    Args:
        page_rows: Lista onde o novo lançamento será adicionado.
        current_conta_row_indexes: Lista de índices dos lançamentos da conta atual.
        cabecalho: Texto do cabeçalho do documento.
        periodo_inicio: Data de início do período.
        periodo_fim: Data de fim do período.
        cnpj: CNPJ da empresa.
        conta_codigo: Código da conta contábil.
        conta_nome: Nome da conta contábil.
        data_lancamento: Data do lançamento.
        historico: Descrição do lançamento.
        valor_str: Valor como string no formato brasileiro.
        arquivo: Caminho do arquivo de origem.
        source: Fonte dos dados.
        pagina: Número da página.
    """
    row = {
        "cabecalho": cabecalho,
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
    """
    Extrai as datas de início e fim do período do texto usando regex.

    Args:
        text: Texto completo da página onde buscar o período.

    Returns:
        Tupla com (data_inicio, data_fim) ou ("", "") se não encontrado.
    """
    match = PERIODO_RE.search(text)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def _extract_cnpj(text: str) -> str:
    """
    Extrai o CNPJ do texto usando regex.

    Args:
        text: Texto completo da página onde buscar o CNPJ.

    Returns:
        CNPJ encontrado ou string vazia se não encontrado.
    """
    match = CNPJ_RE.search(text)
    return match.group(0) if match else ""


def _parse_currency_br(value: str) -> Optional[float]:
    """
    Converte um valor monetário brasileiro (1.234,56) para float.

    Remove separadores de milhar (.) e converte vírgula para ponto decimal.

    Args:
        value: Valor como string no formato brasileiro.

    Returns:
        Valor como float ou None se conversão falhar.
    """
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
    """
    Converte valor para int de forma segura.

    Args:
        value: Valor a ser convertido.

    Returns:
        Valor como int ou None se conversão falhar.
    """
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_string(value: Any) -> str:
    """
    Converte valor para string de forma segura.

    Args:
        value: Valor a ser convertido.

    Returns:
        Valor como string (nunca None).
    """
    if value is None:
        return ""
    return str(value)
