from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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

"""
Módulo para geração de índices semânticos vetoriais.

Este módulo implementa a criação de índices FAISS para busca semântica
em dados contábeis. Extrai termos relevantes de colunas textuais,
gera embeddings usando Sentence Transformers e constrói um índice
vetorial para matching semântico eficiente.

Funcionalidades:
- Extração e normalização de termos de texto contábil
- Filtragem de stopwords em português
- Geração de embeddings com modelos pré-treinados
- Construção e persistência de índices FAISS
- Resolução automática de caminhos de arquivo
"""

@dataclass
class SemanticIndexBuildResult:
    enabled: bool
    index_path: str
    terms_path: str
    term_count: int
    model_name: str
    message: str


class SemanticIndexBuilder:
    """
    Construtor de índices semânticos para busca vetorial em dados contábeis.

    Esta classe extrai termos de colunas textuais, gera embeddings e cria
    um índice FAISS para matching semântico. Suporta configuração de modelo
    e opções de armazenamento local.
    """

    def __init__(
        self,
        index_path: str,
        terms_path: str,
        model_name: str,
        enabled: bool = True,
        local_files_only: bool = False,
    ) -> None:
        """
        Inicializa o construtor de índices semânticos.

        Args:
            index_path: Caminho onde salvar o índice FAISS.
            terms_path: Caminho onde salvar os metadados dos termos.
            model_name: Nome do modelo Sentence Transformers.
            enabled: Se deve construir o índice.
            local_files_only: Usar apenas modelos locais (sem download).
        """
        self.index_path = index_path
        self.terms_path = terms_path
        self.model_name = model_name
        self.enabled = enabled
        self.local_files_only = local_files_only

    @staticmethod
    def _stopwords() -> set[str]:
        """
        Retorna conjunto de stopwords em português para filtragem de termos.

        Inclui palavras comuns da língua portuguesa e termos específicos
        de consultas contábeis que não agregam valor semântico.

        Returns:
            Conjunto de palavras a serem ignoradas na extração de termos.
        """
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
            "total",
            "totais",
            "debito",
            "credito",
            "soma",
            "somar",
        }

    @staticmethod
    def _normalize_text(value: str) -> str:
        """
        Normaliza texto removendo acentos e caracteres especiais.

        Converte para minúsculas, remove acentos via NFKD normalization,
        mantém apenas caracteres ASCII alfanuméricos e espaços.

        Args:
            value: Texto a ser normalizado.

        Returns:
            Texto normalizado em minúsculas sem acentos.
        """
        raw = (value or "").lower().strip()
        ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        return " ".join(ascii_text.split())

    @staticmethod
    def _safe_tokenize(value: str) -> list[str]:
        """
        Tokeniza texto extraindo apenas sequências alfanuméricas.

        Usa regex para encontrar tokens consistindo de letras minúsculas e dígitos.

        Args:
            value: Texto a ser tokenizado.

        Returns:
            Lista de tokens alfanuméricos encontrados.
        """
        return re.findall(r"[a-z0-9]+", value or "")

    def _extract_terms(self, df_structured: pd.DataFrame) -> list[str]:
        """
        Extrai termos únicos relevantes das colunas textuais do DataFrame.

        Processa colunas 'conta_nome' e 'historico', normaliza o texto,
        remove stopwords e tokens muito curtos, retornando termos únicos.

        Args:
            df_structured: DataFrame com dados estruturados contábeis.

        Returns:
            Lista de termos únicos extraídos para indexação.
        """
        columns = [col for col in ("conta_nome", "historico") if col in df_structured.columns]
        if not columns:
            return []

        stopwords = self._stopwords()
        terms: list[str] = []
        for col in columns:
            for raw in df_structured[col].fillna("").astype(str).tolist():
                normalized = self._normalize_text(raw)
                for token in self._safe_tokenize(normalized):
                    if len(token) < 4 or token in stopwords:
                        continue
                    if token not in terms:
                        terms.append(token)
        return terms

    def build(self, df_structured: pd.DataFrame) -> SemanticIndexBuildResult:
        """
        Constrói o índice semântico a partir dos dados estruturados.

        Extrai termos, gera embeddings usando o modelo configurado,
        cria índice FAISS e salva ambos os arquivos. Retorna resultado
        com status e metadados da operação.

        Args:
            df_structured: DataFrame com dados contábeis estruturados.

        Returns:
            SemanticIndexBuildResult com detalhes da construção do índice.
        """
        if not self.enabled:
            return SemanticIndexBuildResult(
                enabled=False,
                index_path=self.index_path,
                terms_path=self.terms_path,
                term_count=0,
                model_name=self.model_name,
                message="Indexação semântica desabilitada.",
            )

        if SentenceTransformer is None or faiss is None or np is None:
            return SemanticIndexBuildResult(
                enabled=False,
                index_path=self.index_path,
                terms_path=self.terms_path,
                term_count=0,
                model_name=self.model_name,
                message=(
                    "Dependências semânticas não disponíveis. "
                    "Instale sentence-transformers, faiss-cpu e numpy."
                ),
            )

        terms = self._extract_terms(df_structured)
        if not terms:
            return SemanticIndexBuildResult(
                enabled=False,
                index_path=self.index_path,
                terms_path=self.terms_path,
                term_count=0,
                model_name=self.model_name,
                message="Sem termos suficientes para criar índice semântico.",
            )

        try:
            # Carrega o modelo de embeddings
            model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            # Gera embeddings normalizados para os termos
            embeddings = model.encode(
                terms,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype="float32")
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                raise RuntimeError("Embeddings inválidos para indexação.")

            # Cria índice FAISS com produto interno (cosine similarity)
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            index.add(embeddings)

            # Salva índice e metadados
            index_path = Path(self.index_path)
            terms_path = Path(self.terms_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            terms_path.parent.mkdir(parents=True, exist_ok=True)

            faiss.write_index(index, str(index_path))
            terms_payload: dict[str, Any] = {
                "model_name": self.model_name,
                "local_files_only": self.local_files_only,
                "term_count": len(terms),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "terms": terms,
            }
            terms_path.write_text(
                json.dumps(terms_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return SemanticIndexBuildResult(
                enabled=True,
                index_path=str(index_path),
                terms_path=str(terms_path),
                term_count=len(terms),
                model_name=self.model_name,
                message=(
                    f"Índice semântico criado com {len(terms)} termos em {index_path.name}."
                ),
            )
        except Exception as exc:
            return SemanticIndexBuildResult(
                enabled=False,
                index_path=self.index_path,
                terms_path=self.terms_path,
                term_count=0,
                model_name=self.model_name,
                message=f"Falha ao criar índice semântico: {exc}",
            )


def resolve_semantic_paths(
    structured_path: str,
    semantic_index_path: str | None,
    semantic_terms_path: str | None,
) -> tuple[str, str]:
    """
    Resolve os caminhos para os arquivos de índice semântico.

    Se os caminhos não forem fornecidos, usa valores padrão baseados
    no diretório do arquivo estruturado ou variáveis de ambiente.

    Args:
        structured_path: Caminho do arquivo Parquet estruturado.
        semantic_index_path: Caminho do índice FAISS (opcional).
        semantic_terms_path: Caminho do arquivo de termos (opcional).

    Returns:
        Tupla com caminhos resolvidos para (índice, termos).
    """
    base_structured = Path(structured_path)
    base_dir = base_structured.parent

    resolved_index = Path(
        semantic_index_path
        or os.getenv("SEMANTIC_INDEX_PATH")
        or str(base_dir / "semantic_terms.faiss")
    )
    resolved_terms = Path(
        semantic_terms_path
        or os.getenv("SEMANTIC_TERMS_PATH")
        or str(base_dir / "semantic_terms.json")
    )
    return str(resolved_index), str(resolved_terms)
