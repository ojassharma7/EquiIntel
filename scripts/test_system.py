#!/usr/bin/env python3
"""
EquIntel System Test
Tests all components of the EquIntel system
"""

import sys
import os
from pathlib import Path
import json

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))


def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from app.document_parser import DocumentParser, ParsedDocument
        from app.chunker import DocumentChunker, TextChunk
        from app.vector_store import VectorStore, VectorRecord
        from app.rag_engine import RAGEngine, QueryResult, create_llm
        print("✅ All app modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_document_parser():
    """Test document parser functionality"""
    print("📄 Testing document parser...")
    
    try:
        parser = DocumentParser()
        
        # Create a test document
        test_text = """
        NVIDIA CORPORATION
        10-K ANNUAL REPORT
        FISCAL YEAR ENDED JANUARY 29, 2023
        
        ITEM 1. BUSINESS
        
        NVIDIA Corporation is a world leader in visual computing.
        We pioneered a supercharged form of computing loved by the most demanding computer users.
        
        RISK FACTORS
        
        Our business is subject to numerous risks and uncertainties.
        We depend on third parties to manufacture our products.
        """
        
        # Test metadata extraction
        metadata = parser.extract_metadata_from_content(test_text, "NVIDIA_10K_2023.pdf")
        
        assert metadata.ticker == "NVIDIA"
        assert metadata.doc_type == "10-K"
        assert metadata.year == 2023
        
        print("✅ Document parser test passed")
        return True
        
    except Exception as e:
        print(f"❌ Document parser test failed: {e}")
        return False


def test_chunker():
    """Test document chunker functionality"""
    print("✂️ Testing document chunker...")
    
    try:
        chunker = DocumentChunker()
        
        test_text = """
        This is a test document with multiple sentences. 
        It contains information about NVIDIA Corporation.
        The company is a leader in AI and graphics processing.
        They have faced various challenges in the semiconductor industry.
        """
        
        metadata = {"ticker": "NVDA", "doc_type": "test"}
        chunks = chunker.chunk_document(test_text, metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.metadata["ticker"] == "NVDA" for chunk in chunks)
        
        print("✅ Document chunker test passed")
        return True
        
    except Exception as e:
        print(f"❌ Document chunker test failed: {e}")
        return False


def test_vector_store():
    """Test vector store functionality"""
    print("🗄️ Testing vector store...")
    
    try:
        vector_store = VectorStore()
        
        # Test embedding generation
        test_text = "This is a test document about NVIDIA."
        embedding = vector_store.embed_text(test_text)
        
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Test adding chunks
        test_chunks = [
            {
                "chunk_id": "test_1",
                "text": "NVIDIA is a leader in AI computing.",
                "metadata": {"ticker": "NVDA", "doc_type": "test"}
            }
        ]
        
        vector_store.add_chunks(test_chunks)
        
        # Test search
        results = vector_store.search("AI computing", k=5)
        assert len(results) > 0
        
        print("✅ Vector store test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_rag_engine():
    """Test RAG engine functionality"""
    print("🧠 Testing RAG engine...")
    
    try:
        from app.vector_store import VectorStore
        from app.rag_engine import RAGEngine, create_llm
        
        vector_store = VectorStore()
        
        # Create a mock LLM for testing
        class MockLLM:
            def _call(self, prompt, stop=None):
                return "This is a test response from the LLM."
        
        rag_engine = RAGEngine(vector_store, MockLLM())
        
        # Test query
        result = rag_engine.query("What is NVIDIA?")
        
        assert hasattr(result, 'query')
        assert hasattr(result, 'answer')
        assert hasattr(result, 'sources')
        assert hasattr(result, 'confidence')
        
        print("✅ RAG engine test passed")
        return True
        
    except Exception as e:
        print(f"❌ RAG engine test failed: {e}")
        return False


def test_pipeline():
    """Test the complete pipeline"""
    print("🔄 Testing complete pipeline...")
    
    try:
        from scripts.pipeline import EquIntelPipeline
        
        pipeline = EquIntelPipeline()
        
        # Test pipeline initialization
        assert pipeline.parser is not None
        assert pipeline.chunker is not None
        assert pipeline.vector_store is not None
        assert pipeline.rag_engine is not None
        
        print("✅ Pipeline test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def test_sample_data():
    """Test with sample data"""
    print("📊 Testing with sample data...")
    
    try:
        # Create sample data
        sample_files = [
            "raw_pdfs/NVIDIA_10K_2023.txt",
            "raw_pdfs/NVIDIA_Earnings_Q4_2023.txt"
        ]
        
        # Check if sample files exist
        existing_files = [f for f in sample_files if Path(f).exists()]
        
        if existing_files:
            print(f"✅ Found {len(existing_files)} sample files")
            return True
        else:
            print("⚠️ No sample files found, run download_sample_data.py first")
            return False
            
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 EquIntel System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_document_parser,
        test_chunker,
        test_vector_store,
        test_rag_engine,
        test_pipeline,
        test_sample_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EquIntel is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Add PDF documents to raw_pdfs/")
        print("2. Run: python scripts/pipeline.py --action pipeline")
        print("3. Or start UI: streamlit run ui/app.py")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        print("💡 Try running: python scripts/setup.py")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 