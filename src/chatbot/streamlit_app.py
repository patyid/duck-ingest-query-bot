from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from chatbot.sql_tool import SQLTool
from config.settings import settings


st.set_page_config(page_title="Chatbot DuckDB", page_icon="🦆", layout="wide")
st.title("Chatbot DuckDB")
st.caption("Faça perguntas em português sobre os dados contábeis.")


@st.cache_resource
def build_sql_tool(parquet_path: str) -> SQLTool:
    return SQLTool(parquet_path=parquet_path)


if "messages" not in st.session_state:
    st.session_state.messages = []

sql_tool = build_sql_tool(settings.data_structured)

with st.sidebar:
    st.subheader("Configuração")
    st.write(f"Arquivo de dados: `{settings.data_structured}`")

    with st.expander("Schema disponível"):
        st.text(sql_tool.schema_description())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            st.code(message["sql"], language="sql")

question = st.chat_input("Ex.: Qual o total de débito por conta?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando dados..."):
            try:
                response = sql_tool.ask(question)
                st.markdown(response.answer)
                with st.expander(f"SQL executado ({response.row_count} linhas)", expanded=False):
                    st.code(response.sql, language="sql")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sql": response.sql,
                    }
                )
            except Exception as exc:
                error_message = f"Não consegui concluir a consulta: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
