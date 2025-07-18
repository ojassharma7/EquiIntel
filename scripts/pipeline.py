#!/usr/bin/env python3
"""
EquIntel Pipeline
Main script to orchestrate the entire document processing workflow
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import time

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from document_parser import DocumentParser
from chunker import DocumentChunker
from vector_store import VectorStore
from rag_engine import RAGEngine, create_llm


class EquIntelPipeline:
    """Main pipeline for processing financial documents"""
    
    def __init__(self, 
                 raw_pdfs_dir: str = "raw_pdfs",
                 parsed_data_dir: str = "parsed_data",
                 embeddings_dir: str = "embeddings",
                 llm_type: str = "openai"):
        
        self.raw_pdfs_dir = Path(raw_pdfs_dir)
        self.parsed_data_dir = Path(parsed_data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        
        # Create directories
        self.raw_pdfs_dir.mkdir(exist_ok=True)
        self.parsed_data_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.parser = DocumentParser(str(self.parsed_data_dir))
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore(str(self.embeddings_dir))
        
        # Initialize LLM
        self.llm = create_llm(llm_type)
        self.rag_engine = RAGEngine(self.vector_store, self.llm)
    
    def parse_documents(self, pdf_dir: str = None) -> List[str]:
        """Parse PDF documents"""
        pdf_dir = pdf_dir or self.raw_pdfs_dir
        
        print(f"🔍 Parsing documents from {pdf_dir}...")
        start_time = time.time()
        
        parsed_files = self.parser.parse_directory(str(pdf_dir))
        
        elapsed = time.time() - start_time
        print(f"✅ Parsed {len(parsed_files)} documents in {elapsed:.2f}s")
        
        return parsed_files
    
    def chunk_documents(self, parsed_dir: str = None) -> List[str]:
        """Chunk parsed documents"""
        parsed_dir = parsed_dir or self.parsed_data_dir
        
        print(f"✂️ Chunking documents from {parsed_dir}...")
        start_time = time.time()
        
        chunked_files = []
        for parsed_file in Path(parsed_dir).glob("*_parsed.json"):
            try:
                with open(parsed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                text = data['text']
                chunks = self.chunker.chunk_document(text, metadata)
                
                output_file = str(parsed_file).replace('_parsed.json', '_chunks.json')
                self.chunker.save_chunks(chunks, output_file)
                chunked_files.append(output_file)
                
            except Exception as e:
                print(f"❌ Error chunking {parsed_file}: {e}")
        
        elapsed = time.time() - start_time
        print(f"✅ Chunked {len(chunked_files)} documents in {elapsed:.2f}s")
        
        return chunked_files
    
    def embed_documents(self, chunks_dir: str = None) -> int:
        """Embed document chunks"""
        chunks_dir = chunks_dir or self.parsed_data_dir
        
        print(f"🧠 Embedding documents from {chunks_dir}...")
        start_time = time.time()
        
        total_chunks = 0
        for chunks_file in Path(chunks_dir).glob("*_chunks.json"):
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                self.vector_store.add_chunks(chunks)
                total_chunks += len(chunks)
                
            except Exception as e:
                print(f"❌ Error embedding {chunks_file}: {e}")
        
        elapsed = time.time() - start_time
        print(f"✅ Embedded {total_chunks} chunks in {elapsed:.2f}s")
        
        return total_chunks
    
    def run_full_pipeline(self) -> Dict:
        """Run the complete pipeline"""
        print("🚀 Starting EquIntel Pipeline...")
        print("=" * 50)
        
        results = {
            'parsed_files': [],
            'chunked_files': [],
            'total_chunks': 0,
            'vector_store_stats': {}
        }
        
        # Step 1: Parse documents
        try:
            results['parsed_files'] = self.parse_documents()
        except Exception as e:
            print(f"❌ Error in parsing step: {e}")
            return results
        
        # Step 2: Chunk documents
        try:
            results['chunked_files'] = self.chunk_documents()
        except Exception as e:
            print(f"❌ Error in chunking step: {e}")
            return results
        
        # Step 3: Embed documents
        try:
            results['total_chunks'] = self.embed_documents()
        except Exception as e:
            print(f"❌ Error in embedding step: {e}")
            return results
        
        # Get final stats
        try:
            results['vector_store_stats'] = self.vector_store.get_document_stats()
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
        
        print("=" * 50)
        print("✅ Pipeline completed successfully!")
        print(f"📊 Final stats: {json.dumps(results['vector_store_stats'], indent=2)}")
        
        return results
    
    def query(self, query: str, **kwargs) -> Dict:
        """Run a query against the processed documents"""
        print(f"🔍 Querying: {query}")
        
        result = self.rag_engine.query(query, **kwargs)
        
        print(f"📝 Answer: {result.answer}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"📚 Sources: {len(result.sources)} chunks")
        
        return result.model_dump()
    
    def get_stats(self) -> Dict:
        """Get statistics about the processed documents"""
        return self.vector_store.get_document_stats()


def main():
    parser = argparse.ArgumentParser(description="EquIntel Pipeline")
    parser.add_argument("--action", choices=["parse", "chunk", "embed", "pipeline", "query", "stats"], 
                       default="pipeline", help="Action to perform")
    parser.add_argument("--pdf-dir", default="raw_pdfs", help="Directory containing PDF files")
    parser.add_argument("--parsed-dir", default="parsed_data", help="Directory for parsed documents")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Directory for embeddings")
    parser.add_argument("--llm-type", default="openai", choices=["openai", "local"], 
                       help="Type of LLM to use")
    parser.add_argument("--query", help="Query to run (for query action)")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--ticker", help="Filter by ticker")
    parser.add_argument("--doc-type", help="Filter by document type")
    parser.add_argument("--year", type=int, help="Filter by year")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EquIntelPipeline(
        raw_pdfs_dir=args.pdf_dir,
        parsed_data_dir=args.parsed_dir,
        embeddings_dir=args.embeddings_dir,
        llm_type=args.llm_type
    )
    
    if args.action == "parse":
        pipeline.parse_documents()
    
    elif args.action == "chunk":
        pipeline.chunk_documents()
    
    elif args.action == "embed":
        pipeline.embed_documents()
    
    elif args.action == "pipeline":
        pipeline.run_full_pipeline()
    
    elif args.action == "query":
        if not args.query:
            print("❌ Query is required for query action")
            return
        
        filters = {}
        if args.ticker:
            filters['ticker'] = args.ticker
        if args.doc_type:
            filters['doc_type'] = args.doc_type
        if args.year:
            filters['year'] = args.year
        
        result = pipeline.query(args.query, k=args.k, filters=filters if filters else None)
        
        # Save result
        output_file = f"query_result_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Result saved to {output_file}")
    
    elif args.action == "stats":
        stats = pipeline.get_stats()
        print("📊 Vector Store Statistics:")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main() 