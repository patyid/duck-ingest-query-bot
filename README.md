# Duck Ingest Query Bot

Um pipeline de ingestão focado em transformar PDFs locais em dados estruturados prontos para análise e construção de vetores (duckdb + RAG).

## Overview

- Carrega todos os PDFs encontrados em `data/raw` usando o `PDFLoader` baseado em PyMuPDF.
- Detecta documentos de imagem e, se `unstructured[all-docs]` estiver disponível, aplica OCR para extrair texto.
- Consolida cada página em um `DataFrame` e salva o resultado como um arquivo Parquet comprimido em `data/processed`.

## Requirements

- Python `>=3.9.0,<3.13.0` (sempre alinhe ao ambiente virtual que você usa).
- `pip install -r requirements.txt` traz as dependências principais (LangChain, pandas, pyarrow, faiss, openai, qwen-agent, ollama, streamlit, duckdb etc.).

## Setup

1. Entre na pasta do projeto:
   `cd /home/patriciacafundo/dev/git/duck-ingest-query-bot`
2. Ative o ambiente virtual já criado:
   `source /home/patriciacafundo/dev/venv/chatbotduckdb/bin/activate`
3. Alternativa de ativação por caminho relativo (a partir da pasta do projeto):
   `source ../../venv/chatbotduckdb/bin/activate`
4. Confirme que o Python ativo é do `venv`:
   `which python`
5. O resultado esperado deve ser:
   `/home/patriciacafundo/dev/venv/chatbotduckdb/bin/python`
6. Instale as dependências:
   `pip install -r requirements.txt`
7. Configure variáveis de ambiente em `.env` (o `config/settings.py` tenta carregá-lo) ou exporte-as diretamente.
8. Quando terminar, desative o ambiente com:
   `deactivate`


## Executando o pipeline

1. Coloque os PDFs que deseja processar em `data/raw` (pode organizar em subpastas).
2. Rode `python src/loaders/main.py --data-dir data/raw`.
3. O script imprime o progresso de cada PDF, aplica OCR quando necessário e gera dois Parquets:
   - `data/processed/ingestion.parquet` (bruto por página)
   - `data/processed/razao_contabil.parquet` (estruturado por lançamento contábil)

### Exemplo de execução

```
python src/loaders/main.py --data-dir ./data/raw --data-processed ./data/processed/ingestion.parquet

```

### Query em colunas no DuckDB

```sql
SELECT
  conta_codigo,
  conta_nome,
  data_lancamento,
  historico,
  valor,
  total_debito
FROM 'data/processed/razao_contabil.parquet'
WHERE cnpj = '08.244.460/0001-77'
ORDER BY conta_codigo, data_lancamento;
```

### Selecionando colunas no `main`

Você pode escolher quais colunas vão para o Parquet estruturado com `--structured-columns`:

```bash
python src/loaders/main.py \
  --data-dir data/raw \
  --structured-columns cabecalho,periodo_inicio,periodo_fim,cnpj,conta_codigo,conta_nome,data_lancamento,historico,valor,total_debito
```

Colunas permitidas:
`cabecalho, periodo_inicio, periodo_fim, cnpj, conta_codigo, conta_nome, data_lancamento, historico, valor, total_debito, arquivo, source, pagina`

Se alguma coluna for informada com nome errado, o processo falha com erro e é interrompido.

Se quiser testar apenas um subconjunto de PDFs, aponte `--data-dir` para uma pasta menor e verifique o arquivo Parquet gerado em `data/processed`.

### Observações

- O parser aceita `--data-dir`, `--data-processed`, `--data-structured` e `--structured-columns`.
- `IngestionPipeline` usa o `PDFLoader` e salva o DataFrame final via `pandas.to_parquet(..., engine="pyarrow", compression="snappy")`.


## Estrutura do projeto

- `src/loaders/pdf_loader.py`: responsável por localizar PDFs, fazer o parsing com PyMuPDF e, quando necessário, aplicar OCR com `unstructured.partition.pdf`.
- `src/loaders/pipeline.py`: orquestração da ingestão (loader → storage).
- `src/loaders/main.py`: CLI trivial que instancia o pipeline e dispara `run()`.

## Próximos passos recomendados

1. Adicionar testes, logging estruturado e suporte a argumentos adicionais (e.g., buckets, flags de OCR).
2. Verificar se alguma das dependências como `ollama`, `qwen-agent` e `streamlit` será utilizada no momento ou pode ser removida, já que o projeto só está operando na ingestão.

## Licença

Este repositório não declara formalmente uma licença. Adicione o arquivo `LICENSE` se precisar definir um termo de uso.


source ../../venv/chatbotduckdb/bin/activate
