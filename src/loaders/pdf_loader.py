import os
from typing import Any, Dict, List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

import pandas as pd

"""
Módulo para carregamento e processamento de PDFs.

Este módulo fornece funcionalidades para carregar PDFs de um diretório,
extrair texto de cada página e detectar quando é necessário aplicar OCR
para documentos de imagem. Suporta carregamento recursivo de diretórios
e conversão dos dados para um DataFrame padronizado.

Funcionalidades principais:
- Carregamento de PDFs usando PyMuPDF para texto nativo
- Detecção automática de PDFs de imagem e aplicação de OCR com Unstructured
- Normalização de metadados e conteúdo de página
- Output em DataFrame com schema consistente para processamento posterior
"""

METADATA_SCHEMA_KEYS = [
    "author",
    "creationDate",
    "creationdate_1",
    "creator",
    "file_path",
    "format",
    "keywords",
    "modDate",
    "moddate_1",
    "page",
    "producer",
    "source",
    "subject",
    "title",
    "total_pages",
    "trapped",
]


class PDFLoader:
    def __init__(self, directory: str, use_ocr: bool = True):
        """
        Inicializa o carregador de PDFs.

        Parameters:             
        directory (str): Diretório local (modo local) ou prefixo S3 (modo S3).
        use_ocr (bool, optional): Se True, utiliza ocr para extrair texto de PDFs de imagem. Defaults to True.
        s3_bucket (str, optional): Se informado, lê PDFs de `s3://{s3_bucket}/{directory}`.
        """
        self.directory = directory
        self.use_ocr = use_ocr
        self.has_unstructured = False

        
        if use_ocr:
            try:            
                from unstructured.partition.pdf import partition_pdf
                self.has_unstructured = True
            except ImportError:
                print("⚠️ unstructured não instalado. Instale: pip install unstructured[all-docs]")
                self.has_unstructured = False

    
    def load(self) -> pd.DataFrame:
        """Carrega todos os PDFs do diretório recursivamente."""
        pdfs = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".pdf"):
                    pdfs.append(os.path.join(root, file))
        
        docs = []
        # 1. Load PDFs com PyMuPDF, detectando se são de imagem para aplicar OCR se necessário

        for pdf in pdfs:
            print(f"Processando arquivo: {os.path.basename(pdf)}")
            print(f"📄 Processando: {os.path.basename(pdf)}")
            
            # Tenta PyMuPDF primeiro
            loader = PyMuPDFLoader(pdf)
            temp_docs = loader.load()
            
            # Verifica se extraiu texto significativo
            total_text = sum(len(d.page_content.strip()) for d in temp_docs)
            
            if total_text < 100 and self.use_ocr and self.has_unstructured:
                # Se pouco texto, usa OCR
                print(f"   🖼️  Detectado PDF de imagem, aplicando OCR...")
                temp_docs = self._load_with_ocr(pdf)
            
            docs.extend(temp_docs)
            print(f"   ✓ {len(temp_docs)} páginas processadas")
        
        print(f"\n✓ Total: {len(docs)} páginas de {len(pdfs)} PDFs")

        # 2. Converte para DataFrame no schema esperado do parquet
        records = []
        for idx, doc in enumerate(docs, start=1):
            records.append(
                {
                    "id": int(idx),
                    "metadata": self._normalize_metadata(doc.metadata),
                    "page_content": self._normalize_page_content(doc.page_content),
                    "type": str(getattr(doc, "type", "Document")),
                }
            )

        df = pd.DataFrame(records, columns=["id", "metadata", "page_content", "type"])
        return df
    
    def _load_with_ocr(self, pdf_path: str) -> List[Document]:
        """Usa Unstructured com OCR para PDFs de imagem."""
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",  # Usa OCR
            languages=["por"],   # Português - mude se necessário
        )
        
        # Agrupa por página
        pages = {}
        for element in elements:
            page_num = element.metadata.page_number or 1
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(str(element))
        
        # Cria documentos do LangChain
        from langchain_core.documents import Document
        docs = []
        for page_num, texts in sorted(pages.items()):
            content = "\n".join(texts)
            if content.strip():  # Só adiciona se tiver conteúdo
                docs.append(Document(
                    page_content=content,
                    metadata={"source": pdf_path, "page": page_num}
                ))
        
        return docs

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Padroniza metadata para o schema esperado no parquet."""
        m = metadata or {}

        creation_date = m.get("creationDate") or m.get("creationdate")
        mod_date = m.get("modDate") or m.get("moddate")
        source = m.get("source") or m.get("file_path") or ""
        page = m.get("page")
        total_pages = m.get("total_pages")

        normalized = {
            "author": self._as_string(m.get("author")),
            "creationDate": self._as_string(creation_date),
            "creationdate_1": self._as_string(m.get("creationdate") or creation_date),
            "creator": self._as_string(m.get("creator")),
            "file_path": self._as_string(m.get("file_path") or source),
            "format": self._as_string(m.get("format")),
            "keywords": self._as_string(m.get("keywords")),
            "modDate": self._as_string(mod_date),
            "moddate_1": self._as_string(m.get("moddate") or mod_date),
            "page": self._as_int(page),
            "producer": self._as_string(m.get("producer")),
            "source": self._as_string(source),
            "subject": self._as_string(m.get("subject")),
            "title": self._as_string(m.get("title")),
            "total_pages": self._as_int(total_pages),
            "trapped": self._as_string(m.get("trapped")),
        }

        # Garante ordem e presença de todas as chaves.
        return {key: normalized.get(key) for key in METADATA_SCHEMA_KEYS}

    def _normalize_page_content(self, page_content: Any) -> str:
        """
        Normaliza o conteúdo da página mantendo quebras de linha consistentes.

        Converte quebras de linha do Windows/Mac para Unix e remove espaços extras.

        Args:
            page_content: Conteúdo da página como string ou None.

        Returns:
            Conteúdo normalizado como string.
        """
        text = "" if page_content is None else str(page_content)
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _as_string(value: Any) -> str:
        """
        Converte valor para string de forma segura.

        Args:
            value: Valor a ser convertido.

        Returns:
            String vazia se None, caso contrário str(value).
        """
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _as_int(value: Any):
        """
        Converte valor para int de forma segura.

        Args:
            value: Valor a ser convertido.

        Returns:
            None se valor vazio ou conversão falhar, caso contrário int(value).
        """
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
