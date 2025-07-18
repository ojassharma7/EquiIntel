"""
EquIntel App Module
AI-powered equity research platform
"""

__version__ = "1.0.0"
__author__ = "EquIntel Team"

from .document_parser import DocumentParser, ParsedDocument, DocumentMetadata
from .chunker import DocumentChunker, TextChunk
from .vector_store import VectorStore, VectorRecord
from .rag_engine import RAGEngine, QueryResult, create_llm

__all__ = [
    "DocumentParser",
    "ParsedDocument", 
    "DocumentMetadata",
    "DocumentChunker",
    "TextChunk",
    "VectorStore",
    "VectorRecord",
    "RAGEngine",
    "QueryResult",
    "create_llm"
] 