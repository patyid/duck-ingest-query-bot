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

"""
Módulo da aplicação Streamlit para o chatbot DuckDB.

Esta aplicação fornece uma interface de chat interativa onde os usuários podem fazer perguntas
em português sobre dados contábeis armazenados em um arquivo Parquet estruturado.
O chatbot utiliza o SQLTool para gerar e executar consultas SQL no DuckDB,
com suporte opcional a busca semântica e modo debug para visualizar as consultas geradas.

Funcionalidades principais:
- Chat interativo com histórico de mensagens
- Integração com SQLTool para geração de SQL baseada em LLM
- Modo debug para visualizar consultas SQL geradas e traces
- Exibição do schema da base de dados na sidebar
"""

st.set_page_config(page_title="Chatbot DuckDB", page_icon="🦆", layout="wide")
st.title("Chatbot DuckDB")
st.caption("Faça perguntas em português sobre os dados contábeis.")


@st.cache_resource
def build_sql_tool(parquet_path: str) -> SQLTool:
    """
    Cria e retorna uma instância do SQLTool configurada para o arquivo Parquet especificado.

    Esta função é decorada com @st.cache_resource para otimizar o desempenho,
    evitando recriar o SQLTool a cada interação.

    Args:
        parquet_path (str): Caminho absoluto para o arquivo Parquet estruturado
                           contendo os dados contábeis.

    Returns:
        SQLTool: Instância configurada do SQLTool pronta para processar perguntas.
    """
    return SQLTool(parquet_path=parquet_path)


if "messages" not in st.session_state:
    st.session_state.messages = []

# Instancia o SQLTool usando o cache para evitar recriações desnecessárias
sql_tool = build_sql_tool(settings.data_structured)

# Configura a sidebar com informações de configuração e schema
with st.sidebar:
    st.subheader("Configuração")
    st.write(f"Arquivo de dados: `{settings.data_structured}`")
    st.write(f"Modelo LLM: `{sql_tool.llm_description()}`")
    debug_mode = st.checkbox("Modo debug (mostrar SELECTs gerados)", value=False)

    with st.expander("Schema disponível"):
        st.text(sql_tool.schema_description())

# Renderiza o histórico de mensagens do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if debug_mode and message.get("sql"):
            st.code(message["sql"], language="sql")
        if debug_mode and message.get("generated_sqls"):
            with st.expander(f"SELECTs gerados ({len(message['generated_sqls'])})", expanded=False):
                for idx, generated_sql in enumerate(message["generated_sqls"], start=1):
                    st.caption(f"Tentativa {idx}")
                    st.code(generated_sql, language="sql")
        if debug_mode and message.get("debug_trace"):
            with st.expander("Trace debug", expanded=False):
                st.text("\n".join(message["debug_trace"]))

# Captura a entrada do usuário via chat input
question = st.chat_input("Ex.: Qual o total de débito por conta?")
if question:
    # Adiciona a pergunta do usuário ao histórico de mensagens
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Processa a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Consultando dados..."):
            try:
                # Executa a consulta usando o SQLTool
                response = sql_tool.ask(question)
                st.markdown(response.answer)
                if debug_mode:
                    with st.expander(f"SQL executado ({response.row_count} linhas)", expanded=False):
                        st.code(response.sql, language="sql")
                if debug_mode and response.generated_sqls:
                    with st.expander(f"SELECTs gerados ({len(response.generated_sqls)})", expanded=False):
                        for idx, generated_sql in enumerate(response.generated_sqls, start=1):
                            st.caption(f"Tentativa {idx}")
                            st.code(generated_sql, language="sql")
                if debug_mode and response.debug_trace:
                    with st.expander("Trace debug", expanded=False):
                        st.text("\n".join(response.debug_trace))

                # Adiciona a resposta bem-sucedida ao histórico
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sql": response.sql,
                        "generated_sqls": response.generated_sqls,
                        "debug_trace": response.debug_trace,
                    }
                )
            except Exception as exc:
                # Trata erros ocorridos durante a consulta
                error_message = f"Não consegui concluir a consulta: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
