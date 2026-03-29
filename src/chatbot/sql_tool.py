from __future__ import annotations

import os
import re
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


@dataclass
class SQLToolResult:
    question: str
    sql: str
    answer: str
    row_count: int


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
        self._register_views()

        # Inicialização do agente em 4 blocos: modelo, tools, prompt e agente.
        self.llm = self._build_llm()
        self.tools = self._build_tools()
        self.sql_system_prompt = self._build_system_prompt()
        self.sql_agent = self._build_agent()

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
5. Se houver erro, use `fix_sql_error` e tente novamente (até 3 tentativas).
6. Entregue resposta final objetiva em português do Brasil.

Regras:
- Trabalhe apenas com a tabela `lancamentos`.
- Nunca faça comandos destrutivos.
- Quando necessário, use WHERE, GROUP BY e ORDER BY.
- Se não houver resultado, informe explicitamente.
"""

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
- Retorne apenas SQL pura.
"""
        response = self.llm.invoke(prompt)
        return self._cleanup_sql(getattr(response, "content", str(response)))

    def validate_sql_query(self, query: str) -> str:
        # Validação defensiva antes de executar qualquer SQL.
        clean_query = self._cleanup_sql(query)
        if not clean_query:
            return "Error: Query vazia."

        # Bloqueia múltiplas statements na mesma execução.
        without_trailing_semicolon = clean_query.rstrip(";").strip()
        if ";" in without_trailing_semicolon:
            return "Error: Apenas uma query por execução é permitida."

        # Aceita somente leitura para reduzir risco em ambiente de estudo.
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

        return f"Valid: {without_trailing_semicolon}"

    def execute_sql_query(self, query: str, max_rows: Optional[int] = 200) -> pd.DataFrame:
        # Executa tanto SQL bruta quanto retorno no formato "Valid: ...".
        clean_query = self._cleanup_sql(query)
        if clean_query.startswith("Valid:"):
            clean_query = clean_query[6:].strip()

        # LIMIT padrão para evitar respostas gigantes no chat.
        normalized = " ".join(clean_query.split()).lower()
        if max_rows and " limit " not in normalized:
            clean_query = f"{clean_query} LIMIT {int(max_rows)}"

        df = self.conn.execute(clean_query).fetchdf()
        self._last_executed_sql = clean_query
        self._last_row_count = len(df)
        return df

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
        return self._cleanup_sql(getattr(response, "content", str(response)))

    def _build_tools(self):
        # Cada @tool encapsula uma etapa do workflow para o agente decidir.
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

    def ask_sql(self, question: str) -> dict[str, Any]:
        # Executa streaming do agente e captura somente a resposta final textual.
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
        }

    def ask(self, question: str) -> SQLToolResult:
        # Método de alto nível usado pelo Streamlit.
        result = self.ask_sql(question)
        return SQLToolResult(
            question=question,
            sql=result.get("sql", ""),
            answer=result.get("answer", ""),
            row_count=int(result.get("row_count", 0)),
        )

    def close(self) -> None:
        # Encerra conexão explícita para liberar recursos.
        self.conn.close()
