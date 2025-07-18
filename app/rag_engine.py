"""
RAG Engine for EquIntel
Combines vector search with LLM generation for answering financial queries
"""

import os
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
import openai
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel

from .vector_store import VectorStore


class QueryResult(BaseModel):
    """Result of a RAG query"""
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    metadata: Dict[str, Any]


class LocalLLM(LLM):
    """Wrapper for local LLM using llama-cpp-python"""
    
    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self._llm = None
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the local LLM"""
        if self._llm is None:
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=4
                )
            except ImportError:
                raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        
        response = self._llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=stop
        )
        return response['choices'][0]['text'].strip()
    
    @property
    def _llm_type(self) -> str:
        return "local_llama"


class OpenAILLM(LLM):
    """Wrapper for OpenAI API"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, well-reasoned answers based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                stop=stop
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
    @property
    def _llm_type(self) -> str:
        return "openai"


class RAGEngine:
    """RAG engine for financial document analysis"""
    
    def __init__(self, vector_store: VectorStore, llm: Optional[LLM] = None):
        self.vector_store = vector_store
        self.llm = llm
        
        # Default prompt template for financial analysis
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a financial analyst assistant. Use the following context to answer the question. 
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer the question based on the context provided. Be specific and cite relevant information from the context when possible.
"""
        )
    
    def retrieve_relevant_chunks(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant chunks from vector store"""
        return self.vector_store.search(query, k=k, filters=filters)
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context for LLM"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[Source {i}: {chunk.get('ticker', 'Unknown')} - {chunk.get('doc_type', 'Unknown')}"
            if chunk.get('year'):
                source_info += f" ({chunk['year']})"
            if chunk.get('section'):
                source_info += f" - {chunk['section']}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        if not self.llm:
            return "No LLM configured. Please set up a local or OpenAI LLM."
        
        prompt = self.prompt_template.format(context=context, question=query)
        return self.llm._call(prompt)
    
    def query(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> QueryResult:
        """Main query method"""
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(query, k=k, filters=filters)
        
        if not chunks:
            return QueryResult(
                query=query,
                answer="No relevant documents found in the database.",
                sources=[],
                confidence=0.0,
                metadata={"chunks_retrieved": 0}
            )
        
        # Format context
        context = self.format_context(chunks)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        # Prepare sources
        sources = []
        for chunk in chunks:
            sources.append({
                'chunk_id': chunk['chunk_id'],
                'ticker': chunk.get('ticker'),
                'doc_type': chunk.get('doc_type'),
                'year': chunk.get('year'),
                'section': chunk.get('section'),
                'file_path': chunk.get('file_path'),
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })
        
        # Calculate confidence (simple heuristic based on number of relevant chunks)
        confidence = min(1.0, len(chunks) / k)
        
        return QueryResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={
                "chunks_retrieved": len(chunks),
                "filters_applied": filters
            }
        )
    
    def query_by_ticker(self, query: str, ticker: str, k: int = 10) -> QueryResult:
        """Query within a specific ticker"""
        return self.query(query, k=k, filters={'ticker': ticker})
    
    def query_by_doc_type(self, query: str, doc_type: str, k: int = 10) -> QueryResult:
        """Query within a specific document type"""
        return self.query(query, k=k, filters={'doc_type': doc_type})
    
    def query_by_year(self, query: str, year: int, k: int = 10) -> QueryResult:
        """Query within a specific year"""
        return self.query(query, k=k, filters={'year': year})
    
    def query_by_section(self, query: str, section: str, k: int = 10) -> QueryResult:
        """Query within a specific section"""
        return self.query(query, k=k, filters={'section': section})


def create_llm(llm_type: str = "openai", **kwargs) -> LLM:
    """Create LLM instance based on type"""
    if llm_type == "openai":
        return OpenAILLM(**kwargs)
    elif llm_type == "local":
        return LocalLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def save_query_result(result: QueryResult, output_file: str):
    """Save query result to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Initialize LLM (default to OpenAI, can be overridden)
    llm_type = os.getenv("LLM_TYPE", "openai")
    llm = create_llm(llm_type)
    
    # Initialize RAG engine
    rag_engine = RAGEngine(vector_store, llm)
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = rag_engine.query(query)
        
        print(f"\nQuery: {result.query}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sources: {len(result.sources)} chunks retrieved")
        
        # Save result
        output_file = f"query_result_{int(time.time())}.json"
        save_query_result(result, output_file)
        print(f"Result saved to {output_file}")
    else:
        print("Usage: python rag_engine.py <your query>")
        print("Example: python rag_engine.py 'What are the main risks mentioned in NVIDIA earnings calls?'") 