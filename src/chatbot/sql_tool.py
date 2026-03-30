from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
from langchain_core.tools import tool

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama

try:  # pragma: no cover
    import faiss
except Exception:  # pragma: no cover
    faiss = None

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class SQLToolResult:
    question: str
    sql: str
    answer: str
    row_count: int
    generated_sqls: list[str]
    debug_trace: list[str]


class SQLTool:
    """SQL Agent com tools dedicadas: schema, generate, validate, execute e fix."""

    def __init__(self, parquet_path: str, database_path: str = ":memory:") -> None:
        # Resolve o caminho do parquet com base no root do projeto para evitar
        # problemas quando o app é iniciado de diretórios diferentes.
        self.parquet_path = self._resolve_path(parquet_path)

        # Conexão principal do DuckDB. Em modo de estudo usamos ":memory:" e
        # criamos uma VIEW sobre o parquet para simplificar consultas.
        self.conn = duckdb.connect(database=database_path, read_only=False)

        # Estado auxiliar para exibir no chat a SQL realmente executada e a
        # quantidade de linhas retornadas.
        self._last_executed_sql = ""
        self._last_row_count = 0
        self._generated_sqls: list[str] = []
        self._debug_trace: list[str] = []
        self._semantic_enabled = (os.getenv("SEMANTIC_MATCH_ENABLED", "true").lower() == "true")
        self._semantic_local_files_only = (
            os.getenv("SEMANTIC_LOCAL_FILES_ONLY", "true").lower() == "true"
        )
        self._semantic_model_name_from_env = os.getenv("SEMANTIC_MODEL_NAME")
        self._semantic_model_name = (
            self._semantic_model_name_from_env or "paraphrase-multilingual-mpnet-base-v2"
        )
        self._semantic_index_path = self._resolve_semantic_artifact_path(
            os.getenv("SEMANTIC_INDEX_PATH"),
            default_filename="semantic_terms.faiss",
        )
        self._semantic_terms_path = self._resolve_semantic_artifact_path(
            os.getenv("SEMANTIC_TERMS_PATH"),
            default_filename="semantic_terms.json",
        )
        self._semantic_model: Any = None
        self._semantic_index: Any = None
        self._semantic_terms: list[str] = []
        self._register_views()
        self._init_semantic_matcher()

        # Inicialização do agente em 4 blocos: modelo, tools, prompt e agente.
        self.llm = self._build_llm()
        self.tools = self._build_tools()
        self.sql_system_prompt = self._build_system_prompt()
        self.sql_agent = self._build_agent()

    def _resolve_semantic_artifact_path(
        self,
        configured_path: str | None,
        default_filename: str,
    ) -> str:
        if configured_path:
            path = Path(configured_path)
            if not path.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                path = project_root / path
            return str(path)
        parquet_parent = Path(self.parquet_path).parent
        return str(parquet_parent / default_filename)

    def _build_agent(self):
        # Import tardio para devolver erro amigável se o pacote `langchain`
        # não estiver instalado no ambiente virtual.
        try:
            from langchain.agents import create_agent
        except Exception as exc:
            raise RuntimeError(
                "Pacote `langchain` não encontrado. "
                "Instale dependências com: pip install -r requirements.txt"
            ) from exc

        self.sql_agent = create_agent(
            self.llm,
            self.tools,
            system_prompt=self.sql_system_prompt,
        )
        return self.sql_agent

    def _resolve_path(self, raw_path: str) -> str:
        # Mantém compatibilidade com caminhos relativos em config/.env.
        project_root = Path(__file__).resolve().parents[2]
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        return str(path)

    def _register_views(self) -> None:
        # O chatbot sempre consulta a VIEW `lancamentos`.
        # Ela encapsula o parquet e evita repetir `read_parquet(...)` em toda SQL.
        escaped_path = self.parquet_path.replace("'", "''")
        self.conn.execute(
            "CREATE OR REPLACE VIEW lancamentos AS "
            f"SELECT * FROM read_parquet('{escaped_path}');",
        )

    def _build_llm(self):
        # Setup do modelo no mesmo formato do exemplo de referência.
        return ChatOllama(
            model=os.getenv("LLM") or os.getenv("llm") or "qwen3",
            base_url="http://localhost:11434",
            temperature=0,
        )

    def schema_description(self) -> str:
        # Converte o DESCRIBE em texto para alimentar prompts e inspeção.
        schema_df = self.conn.execute("DESCRIBE lancamentos;").fetchdf()
        lines = []
        for _, row in schema_df.iterrows():
            lines.append(f"- {row['column_name']} ({row['column_type']})")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        schema = self.schema_description()
        return f"""Você é um analista SQL especialista em DuckDB.

Schema disponível:
{schema}

Seu fluxo obrigatório para responder perguntas:
1. Use `get_database_schema` para conferir schema.
2. Use `generate_sql_query` para gerar SQL.
3. Use `validate_sql_query` para validar segurança e formato.
4. Use `execute_sql_query` para executar SQL validada.
5. Se houver erro, use `fix_sql_error` e tente novamente (ate 3 tentativas).
6. Entregue resposta final objetiva em português do Brasil.

Regras:
- Trabalhe apenas com a tabela `lancamentos`.
- Nunca faça comandos destrutivos.
- Quando necessário, use WHERE, GROUP BY e ORDER BY.
- Para filtros por texto (ex.: telefone, internet, energia), considere `conta_nome` e `historico`.
- Para "valor/total de débito ...", prefira `SUM(total_debito)` como métrica principal.
- Para termos textuais com variação morfológica (ex.: telefone, telefonia, telefônica),
  prefira buscar por radical estável com `LIKE` (ex.: `'%telefon%'`) em vez de match literal.
- Se não houver resultado, informe explicitamente.
"""

    def _reset_debug_state(self) -> None:
        # Limpa todo o estado por pergunta antes de iniciar uma nova execução.
        # Isso evita "vazamento" de SQL/trace entre mensagens consecutivas no chat.
        self._last_executed_sql = ""
        self._last_row_count = 0
        self._generated_sqls = []
        self._debug_trace = []

    def _add_debug(self, message: str) -> None:
        self._debug_trace.append(message)

    def _track_generated_sql(self, sql: str, source: str) -> None:
        # Armazena apenas SQL saneada para o modo debug do Streamlit.
        # O campo `source` identifica de onde veio a query (generate/fix).
        clean_sql = self._cleanup_sql(sql)
        if clean_sql:
            self._generated_sqls.append(clean_sql)
            self._add_debug(f"{source}: {clean_sql}")

    @staticmethod
    def _dataframe_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return ""

        headers = [str(col) for col in df.columns.tolist()]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        body_rows: list[str] = []
        for row in df.itertuples(index=False, name=None):
            values = []
            for value in row:
                cell = "" if value is None else str(value)
                cell = cell.replace("\n", " ").replace("|", "\\|")
                values.append(cell)
            body_rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header_row, sep_row, *body_rows])

    @staticmethod
    def _stopwords() -> set[str]:
        return {
            "a",
            "ao",
            "as",
            "com",
            "da",
            "das",
            "de",
            "do",
            "dos",
            "e",
            "em",
            "na",
            "nas",
            "no",
            "nos",
            "o",
            "os",
            "para",
            "por",
            "que",
            "qual",
            "quais",
            "quanto",
            "quantos",
            "lista",
            "liste",
            "listar",
            "list",
            "mostra",
            "mostre",
            "mostrar",
            "conta",
            "contas",
            "lancamento",
            "lancamentos",
            "valor",
            "valores",
            "gasto",
            "gastos",
            "total",
            "totais",
            "debito",
            "debitos",
            "credito",
            "creditos",
            "soma",
            "somar",
        }

    @staticmethod
    def _safe_tokenize(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value or "")

    def _init_semantic_matcher(self) -> None:
        # Carrega componente semântico (preferência: índice persistido da ingestão).
        if not self._semantic_enabled:
            return

        if faiss is None or np is None:
            return

        loaded_prebuilt = self._load_prebuilt_semantic_index()
        if loaded_prebuilt and self._load_semantic_model():
            self._add_debug(
                f"semantic_matcher: índice persistido carregado ({len(self._semantic_terms)} termos)."
            )
        return

    def _load_prebuilt_semantic_index(self) -> bool:
        if faiss is None:
            return False
        index_path = Path(self._semantic_index_path)
        terms_path = Path(self._semantic_terms_path)
        if not index_path.exists() or not terms_path.exists():
            return False

        try:
            payload = json.loads(terms_path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else None
            model_name = payload.get("model_name") if isinstance(payload, dict) else None
            if (
                model_name
                and not self._semantic_model_name_from_env
                and isinstance(model_name, str)
            ):
                self._semantic_model_name = model_name
            if not isinstance(terms, list) or not terms:
                return False

            index = faiss.read_index(str(index_path))
            if int(index.ntotal) != len(terms):
                return False

            self._semantic_index = index
            self._semantic_terms = [str(term) for term in terms]
            return True
        except Exception:
            return False

    def _load_semantic_model(self):
        if self._semantic_model is not None:
            return self._semantic_model
        if SentenceTransformer is None:
            return None
        try:
            self._semantic_model = SentenceTransformer(
                self._semantic_model_name,
                local_files_only=self._semantic_local_files_only,
            )
        except Exception:
            self._semantic_model = None
        return self._semantic_model

    def _semantic_expand_terms(self, terms: list[str], top_k: int = 6) -> dict[str, list[str]]:
        # Para cada termo da pergunta, recupera termos semanticamente próximos
        # presentes no vocabulário do dataset.
        if not terms or self._semantic_index is None or np is None:
            return {}
        model = self._load_semantic_model()
        if model is None:
            return {}

        expanded: dict[str, list[str]] = {}
        max_k = min(top_k, len(self._semantic_terms))
        if max_k <= 0:
            return expanded

        for term in terms:
            try:
                query_vec = model.encode(
                    [term],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                query_vec = np.asarray(query_vec, dtype="float32")
                scores, indices = self._semantic_index.search(query_vec, max_k)
            except Exception:
                continue

            matches: list[str] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._semantic_terms):
                    continue
                candidate = self._semantic_terms[idx]
                # Score de cosseno em embeddings normalizados.
                if float(score) < 0.45:
                    continue
                if candidate == term:
                    continue
                if candidate not in matches:
                    matches.append(candidate)
            if matches:
                expanded[term] = matches

        if expanded:
            self._add_debug(f"semantic_expand_terms: {expanded}")
        return expanded

    def get_database_schema(self, table_name: str | None = None) -> str:
        # Mantém escopo controlado: apenas uma tabela de estudo.
        if table_name and table_name.strip().lower() != "lancamentos":
            return "Error: Somente a tabela 'lancamentos' está disponível."
        return self.schema_description()

    @staticmethod
    def _cleanup_sql(query: str) -> str:
        # Remove blocos markdown que o modelo pode incluir por engano.
        clean_query = (query or "").strip()
        clean_query = re.sub(r"```sql\s*", "", clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r"```\s*", "", clean_query)
        return clean_query.strip()

    def generate_sql_query(self, question: str, schema_info: str = "") -> str:
        # Geração de SQL em linguagem natural seguindo regras de segurança.
        schema_to_use = schema_info.strip() or self.get_database_schema()
        prompt = f"""
Com base no schema:
{schema_to_use}

Gere uma SQL DuckDB para responder:
{question}

Regras:
- Use somente SELECT/CTE.
- Use apenas a tabela lancamentos.
- Use colunas que existam no schema.
- Inclua WHERE, GROUP BY e ORDER BY quando fizer sentido.
- Se a pergunta indicar soma/total de débito, use `SUM(total_debito)`.
- Para filtros textuais, use `lower(coalesce(conta_nome,''))` e `lower(coalesce(historico,''))`.
- Para termos que podem variar em escrita (telefone/telefonia/telefônica), prefira o radical no LIKE
  (ex.: `'%telefon%'`) para evitar falso negativo por match literal.
- Retorne apenas SQL pura.
"""
        response = self.llm.invoke(prompt)
        sql = self._cleanup_sql(getattr(response, "content", str(response)))
        self._track_generated_sql(sql, "generate_sql_query")
        return sql

    def validate_sql_query(self, query: str) -> str:
        # Gate de segurança: bloqueia instruções perigosas e mantém o app
        # estritamente em modo leitura, mesmo quando a SQL veio de LLM.
        clean_query = self._cleanup_sql(query)
        self._add_debug(f"validate_sql_query input: {clean_query}")
        if not clean_query:
            return "Error: Query vazia."

        # Bloqueia múltiplas statements para reduzir superfície de risco.
        # Ex.: "SELECT ...; DROP TABLE ..." deve falhar aqui.
        without_trailing_semicolon = clean_query.rstrip(";").strip()
        if ";" in without_trailing_semicolon:
            return "Error: Apenas uma query por execução é permitida."

        # Permite apenas operações de leitura. O objetivo é impedir qualquer
        # mutação de dados, mesmo acidental, em ambiente de exploração.
        normalized = " ".join(without_trailing_semicolon.split()).lower()
        if not normalized.startswith(("select", "with", "show", "describe")):
            return "Error: Somente consultas de leitura são permitidas."

        blocked = [
            "insert",
            "update",
            "delete",
            "alter",
            "drop",
            "create",
            "truncate",
            "copy",
            "attach",
            "detach",
            "call",
        ]
        for keyword in blocked:
            if re.search(rf"\b{keyword}\b", normalized, flags=re.IGNORECASE):
                return f"Error: Operação bloqueada ({keyword.upper()})."

        valid_result = f"Valid: {without_trailing_semicolon}"
        self._add_debug(f"validate_sql_query output: {valid_result}")
        return valid_result

    def execute_sql_query(self, query: str, max_rows: Optional[int] = 200) -> pd.DataFrame:
        # Aceita query crua ou o retorno de validação no formato "Valid: ...".
        # Isso simplifica o encadeamento entre tools no agente.
        clean_query = self._cleanup_sql(query)
        if clean_query.startswith("Valid:"):
            clean_query = clean_query[6:].strip()

        # LIMIT automático para não explodir a UI com milhares de linhas.
        # Exceção: consultas agregadas (SUM/COUNT/AVG/MIN/MAX) não recebem LIMIT,
        # pois normalmente já retornam poucos registros por natureza.
        normalized = " ".join(clean_query.split()).lower()
        if max_rows and " limit " not in normalized and self._should_apply_limit(normalized):
            clean_query = f"{clean_query} LIMIT {int(max_rows)}"

        self._add_debug(f"execute_sql_query: {clean_query}")
        df = self.conn.execute(clean_query).fetchdf()
        self._last_executed_sql = clean_query
        self._last_row_count = len(df)
        return df

    @staticmethod
    def _should_apply_limit(normalized_query: str) -> bool:
        aggregate_tokens = ("sum(", "count(", "avg(", "min(", "max(")
        if any(token in normalized_query for token in aggregate_tokens):
            return False
        return True

    def fix_sql_error(self, original_query: str, error_message: str, question: str) -> str:
        # Prompt de autocorreção acionado quando validação/execução falha.
        schema = self.get_database_schema()
        prompt = f"""
A query abaixo falhou:
Query: {original_query}
Erro: {error_message}
Pergunta original: {question}

Schema:
{schema}

Corrija a SQL mantendo as regras:
- somente leitura (SELECT/CTE)
- apenas tabela lancamentos
- retornar apenas SQL
"""
        response = self.llm.invoke(prompt)
        sql = self._cleanup_sql(getattr(response, "content", str(response)))
        self._track_generated_sql(sql, "fix_sql_error")
        return sql

    def _build_tools(self):
        # Estas funções são expostas ao agente LangChain como "ferramentas".
        # O modelo escolhe a sequência (gerar -> validar -> executar -> corrigir)
        # e cada etapa fica isolada para facilitar depuração e controle.
        @tool
        def get_database_schema(table_name: str = "") -> str:
            """Get database schema information for SQL query generation."""
            return self.get_database_schema(table_name.strip() or None)

        @tool
        def generate_sql_query(question: str, schema_info: str = "") -> str:
            """Generate SQL query from natural language question."""
            return self.generate_sql_query(question=question, schema_info=schema_info)

        @tool
        def validate_sql_query(query: str) -> str:
            """Validate SQL query for safety before execution."""
            return self.validate_sql_query(query)

        @tool
        def execute_sql_query(query: str, max_rows: int = 200) -> str:
            """Execute validated SQL query and return results."""
            df = self.execute_sql_query(query=query, max_rows=max_rows)
            if df.empty:
                return "Query executada com sucesso, mas sem linhas."
            return df.to_string(index=False)

        @tool
        def fix_sql_error(original_query: str, error_message: str, question: str) -> str:
            """Fix SQL query when validation or execution fails."""
            return self.fix_sql_error(
                original_query=original_query,
                error_message=error_message,
                question=question,
            )

        return [
            get_database_schema,
            generate_sql_query,
            validate_sql_query,
            execute_sql_query,
            fix_sql_error,
        ]

    @staticmethod
    def _normalize_text(value: str) -> str:
        raw = (value or "").lower().strip()
        ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        return " ".join(ascii_text.split())

    @staticmethod
    def _normalized_sql_text(column: str) -> str:
        # Normaliza acentos no lado SQL para permitir filtros robustos
        # mesmo quando a pergunta vem sem acento (ex.: "telefonica").
        expr = f"lower(coalesce({column}, ''))"
        replacements = [
            ("á", "a"),
            ("à", "a"),
            ("â", "a"),
            ("ã", "a"),
            ("é", "e"),
            ("ê", "e"),
            ("í", "i"),
            ("ó", "o"),
            ("ô", "o"),
            ("õ", "o"),
            ("ú", "u"),
            ("ç", "c"),
        ]
        for original, normalized in replacements:
            expr = f"replace({expr}, '{original}', '{normalized}')"
        return expr

    @staticmethod
    def _extract_query_terms(normalized_question: str) -> list[str]:
        stopwords = SQLTool._stopwords()
        tokens = SQLTool._safe_tokenize(normalized_question)
        terms: list[str] = []
        for token in tokens:
            # Remove termos de intenção/medida para evitar filtros inválidos como
            # "debitos" no WHERE textual de conta_nome/historico.
            if token.startswith(("debit", "credit", "som", "total", "gast")):
                continue
            if len(token) < 3 or token in stopwords:
                continue
            if token not in terms:
                terms.append(token)
        return terms

    def _try_rule_based_query(self, question: str) -> dict[str, Any] | None:
        normalized = self._normalize_text(question)
        terms = self._extract_query_terms(normalized)
        if not terms:
            return None

        conta_norm = self._normalized_sql_text("conta_nome")
        historico_norm = self._normalized_sql_text("historico")
        semantic_terms = self._semantic_expand_terms(terms)

        term_clauses: list[str] = []
        for term in terms:
            # Usa o próprio termo + similares semânticos no mesmo grupo.
            # Cada grupo segue obrigatório (AND entre grupos; OR dentro do grupo).
            alternatives = [term]
            alternatives.extend(semantic_terms.get(term, []))

            like_parts: list[str] = []
            for alternative in alternatives:
                stem = alternative[:7] if len(alternative) > 7 else alternative
                like_parts.append(f"({conta_norm} LIKE '%{stem}%')")
                like_parts.append(f"({historico_norm} LIKE '%{stem}%')")

            group_sql = " OR ".join(like_parts)
            term_clauses.append(f"({group_sql})")
        text_filter = " AND ".join(term_clauses)

        list_intent = any(
            token in normalized
            for token in ("liste", "listar", "list", "mostre", "mostrar", "quais", "contas")
        )
        total_debito_intent = (
            any(token in normalized for token in ("total", "soma", "somar"))
            and "debito" in normalized
        )

        if total_debito_intent:
            sql = f"""
SELECT
  SUM(total_debito) AS total_debito,
  COUNT(*) AS linhas
FROM lancamentos
WHERE {text_filter}
""".strip()
            self._track_generated_sql(sql, "rule_based")
            df = self.execute_sql_query(sql, max_rows=200)
            if df.empty:
                return None
            total = df.iloc[0].get("total_debito")
            linhas = int(df.iloc[0].get("linhas", 0))
            total_num = float(total) if total is not None else 0.0
            return {
                "answer": f"Total de débito para filtro textual ({', '.join(terms)}): {total_num:.2f} ({linhas} linhas).",
                "sql": self._last_executed_sql,
                "row_count": self._last_row_count,
                "generated_sqls": self._generated_sqls,
                "debug_trace": self._debug_trace,
            }

        if list_intent:
            sql = f"""
SELECT
  data_lancamento,
  conta_nome,
  historico,
  valor,
  total_debito
FROM lancamentos
WHERE {text_filter}
ORDER BY data_lancamento, conta_nome
""".strip()
            self._track_generated_sql(sql, "rule_based")
            df = self.execute_sql_query(sql, max_rows=200)
            if df.empty:
                return {
                    "answer": f"Não encontrei lançamentos para o filtro textual ({', '.join(terms)}).",
                    "sql": self._last_executed_sql,
                    "row_count": self._last_row_count,
                    "generated_sqls": self._generated_sqls,
                    "debug_trace": self._debug_trace,
                }
            table_md = self._dataframe_to_markdown(df)
            return {
                "answer": (
                    f"Encontrei {len(df)} lançamentos para o filtro textual ({', '.join(terms)}):\n\n"
                    f"{table_md}"
                ),
                "sql": self._last_executed_sql,
                "row_count": self._last_row_count,
                "generated_sqls": self._generated_sqls,
                "debug_trace": self._debug_trace,
            }

        return None

    def ask_sql(self, question: str) -> dict[str, Any]:
        # Inicia execução limpa para a pergunta atual e registra no trace.
        self._reset_debug_state()
        self._add_debug(f"Pergunta: {question}")

        rule_based_result = self._try_rule_based_query(question)
        if rule_based_result is not None:
            self._add_debug("Resposta via fallback determinístico.")
            return rule_based_result

        # Consumimos o stream de eventos do agente e ignoramos mensagens
        # intermediárias de tool-call para retornar apenas a resposta final.
        # A SQL efetivamente executada fica em `_last_executed_sql`.
        final_answer = ""

        for event in self.sql_agent.stream({"messages": question}, stream_mode="values"):
            msg = event["messages"][-1]
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                continue
            content = getattr(msg, "content", "")
            if content:
                final_answer = content

        return {
            "answer": final_answer or "Não consegui gerar uma resposta.",
            "sql": self._last_executed_sql,
            "row_count": self._last_row_count,
            "generated_sqls": self._generated_sqls,
            "debug_trace": self._debug_trace,
        }

    def ask(self, question: str) -> SQLToolResult:
        # Método de alto nível usado pelo Streamlit.
        result = self.ask_sql(question)
        return SQLToolResult(
            question=question,
            sql=result.get("sql", ""),
            answer=result.get("answer", ""),
            row_count=int(result.get("row_count", 0)),
            generated_sqls=list(result.get("generated_sqls", [])),
            debug_trace=list(result.get("debug_trace", [])),
        )

    def close(self) -> None:
        # Encerra conexão explícita para liberar recursos.
        self.conn.close()
