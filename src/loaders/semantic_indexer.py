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


@dataclass
class SemanticIndexBuildResult:
    enabled: bool
    index_path: str
    terms_path: str
    term_count: int
    model_name: str
    message: str


class SemanticIndexBuilder:
    def __init__(
        self,
        index_path: str,
        terms_path: str,
        model_name: str,
        enabled: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.index_path = index_path
        self.terms_path = terms_path
        self.model_name = model_name
        self.enabled = enabled
        self.local_files_only = local_files_only

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
            "total",
            "totais",
            "debito",
            "credito",
            "soma",
            "somar",
        }

    @staticmethod
    def _normalize_text(value: str) -> str:
        raw = (value or "").lower().strip()
        ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        return " ".join(ascii_text.split())

    @staticmethod
    def _safe_tokenize(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value or "")

    def _extract_terms(self, df_structured: pd.DataFrame) -> list[str]:
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
            model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            embeddings = model.encode(
                terms,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype="float32")
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                raise RuntimeError("Embeddings inválidos para indexação.")

            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            index.add(embeddings)

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
