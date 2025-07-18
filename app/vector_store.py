"""
Vector Store for EquIntel
Uses LanceDB to store document embeddings and metadata
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import lancedb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


class VectorRecord(BaseModel):
    """A record in the vector store"""
    chunk_id: str
    text: str
    embedding: List[float]
    ticker: Optional[str] = None
    doc_type: Optional[str] = None
    quarter: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None
    source: Optional[str] = None
    section: Optional[str] = None
    file_path: Optional[str] = None
    token_count: Optional[int] = None


class VectorStore:
    """LanceDB-based vector store for document embeddings"""
    
    def __init__(self, db_path: str = "embeddings", table_name: str = "documents"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path / "equintel.lancedb"))
        self.table_name = table_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create table if it doesn't exist
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the table exists with proper schema"""
        if self.table_name not in self.db.table_names():
            # Create table with sample data
            sample_record = VectorRecord(
                chunk_id="sample",
                text="Sample text",
                embedding=[0.0] * 384,  # all-MiniLM-L6-v2 dimension
                ticker="SAMPLE",
                doc_type="sample",
                year=2023
            )
            
            self.db.create_table(
                self.table_name,
                data=[sample_record.model_dump()],
                mode="overwrite"
            )
            print(f"Created table '{self.table_name}' in LanceDB")
    
    def get_table(self):
        """Get the LanceDB table"""
        return self.db.open_table(self.table_name)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def add_chunks(self, chunks: List[Dict]) -> int:
        """Add chunks to the vector store"""
        if not chunks:
            return 0
        
        # Prepare records
        records = []
        texts = []
        
        for chunk in chunks:
            texts.append(chunk['text'])
            records.append(VectorRecord(
                chunk_id=chunk['chunk_id'],
                text=chunk['text'],
                embedding=[],  # Will be filled after embedding
                ticker=chunk['metadata'].get('ticker'),
                doc_type=chunk['metadata'].get('doc_type'),
                quarter=chunk['metadata'].get('quarter'),
                year=chunk['metadata'].get('year'),
                date=chunk['metadata'].get('date'),
                source=chunk['metadata'].get('source'),
                section=chunk['metadata'].get('section'),
                file_path=chunk['metadata'].get('file_path'),
                token_count=chunk.get('token_count')
            ))
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Update records with embeddings
        for record, embedding in zip(records, embeddings):
            record.embedding = embedding
        
        # Insert into LanceDB
        table = self.get_table()
        table.add([record.model_dump() for record in records])
        
        print(f"Added {len(records)} chunks to vector store")
        return len(records)
    
    def search(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Build search query
        table = self.get_table()
        
        # Start with vector search
        search_query = table.search(query_embedding).limit(k)
        
        # Add filters if provided
        if filters:
            for key, value in filters.items():
                if value is not None:
                    if isinstance(value, list):
                        search_query = search_query.where(f"{key} IN {value}")
                    else:
                        search_query = search_query.where(f"{key} = '{value}'")
        
        # Execute search
        results = search_query.to_pandas()
        
        # Convert to list of dicts
        return results.to_dict('records')
    
    def search_by_ticker(self, query: str, ticker: str, k: int = 10) -> List[Dict]:
        """Search within a specific ticker"""
        return self.search(query, k=k, filters={'ticker': ticker})
    
    def search_by_doc_type(self, query: str, doc_type: str, k: int = 10) -> List[Dict]:
        """Search within a specific document type"""
        return self.search(query, k=k, filters={'doc_type': doc_type})
    
    def search_by_year(self, query: str, year: int, k: int = 10) -> List[Dict]:
        """Search within a specific year"""
        return self.search(query, k=k, filters={'year': year})
    
    def search_by_section(self, query: str, section: str, k: int = 10) -> List[Dict]:
        """Search within a specific section"""
        return self.search(query, k=k, filters={'section': section})
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        table = self.get_table()
        df = table.to_pandas()
        
        stats = {
            'total_chunks': len(df),
            'unique_tickers': df['ticker'].nunique() if 'ticker' in df.columns else 0,
            'unique_doc_types': df['doc_type'].nunique() if 'doc_type' in df.columns else 0,
            'year_range': {
                'min': int(df['year'].min()) if 'year' in df.columns and not df['year'].isna().all() else None,
                'max': int(df['year'].max()) if 'year' in df.columns and not df['year'].isna().all() else None
            },
            'sections': df['section'].value_counts().to_dict() if 'section' in df.columns else {},
            'doc_types': df['doc_type'].value_counts().to_dict() if 'doc_type' in df.columns else {},
            'tickers': df['ticker'].value_counts().to_dict() if 'ticker' in df.columns else {}
        }
        
        return stats
    
    def delete_by_ticker(self, ticker: str) -> int:
        """Delete all chunks for a specific ticker"""
        table = self.get_table()
        result = table.delete(f"ticker = '{ticker}'")
        print(f"Deleted {result} chunks for ticker {ticker}")
        return result
    
    def delete_by_file_path(self, file_path: str) -> int:
        """Delete all chunks for a specific file"""
        table = self.get_table()
        result = table.delete(f"file_path = '{file_path}'")
        print(f"Deleted {result} chunks for file {file_path}")
        return result
    
    def clear_all(self):
        """Clear all data from the vector store"""
        table = self.get_table()
        table.delete("1=1")  # Delete all records
        print("Cleared all data from vector store")


def load_chunks_from_file(chunks_file: str) -> List[Dict]:
    """Load chunks from a JSON file"""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_chunks_directory(chunks_dir: str, vector_store: VectorStore):
    """Process all chunk files in a directory"""
    chunks_dir = Path(chunks_dir)
    
    for chunks_file in chunks_dir.glob("*_chunks.json"):
        try:
            chunks = load_chunks_from_file(str(chunks_file))
            vector_store.add_chunks(chunks)
        except Exception as e:
            print(f"Error processing {chunks_file}: {e}")


if __name__ == "__main__":
    import sys
    
    vector_store = VectorStore()
    
    if len(sys.argv) > 1:
        chunks_path = sys.argv[1]
        
        if os.path.isdir(chunks_path):
            process_chunks_directory(chunks_path, vector_store)
        else:
            chunks = load_chunks_from_file(chunks_path)
            vector_store.add_chunks(chunks)
        
        # Print stats
        stats = vector_store.get_document_stats()
        print("\nVector Store Statistics:")
        print(json.dumps(stats, indent=2))
    else:
        print("Usage: python vector_store.py <chunks_file_or_directory>") 