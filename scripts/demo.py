#!/usr/bin/env python3
"""
EquIntel Demo
Demonstrates the EquIntel system with sample data
"""

import sys
import os
from pathlib import Path
import time

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from app.document_parser import DocumentParser
from app.chunker import DocumentChunker
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine, create_llm


def create_sample_data():
    """Create sample financial documents"""
    print("📝 Creating sample financial documents...")
    
    raw_pdfs_dir = Path("raw_pdfs")
    raw_pdfs_dir.mkdir(exist_ok=True)
    
    # Sample NVIDIA 10-K content
    nvidia_10k = """
    NVIDIA CORPORATION
    10-K ANNUAL REPORT
    FISCAL YEAR ENDED JANUARY 29, 2023
    
    ITEM 1. BUSINESS
    
    NVIDIA Corporation ("NVIDIA," "we," "us," "our," or the "Company") is a world leader in visual computing. We pioneered a supercharged form of computing loved by the most demanding computer users in the world — scientists, designers, artists, and gamers. It was the first to bring programmable shading to consumers, and it is the only company with substantial expertise in both the visual and artificial intelligence, or AI, sides of computer graphics.
    
    Our two platforms address four large and growing markets — Gaming, Data Center, Professional Visualization, and Automotive. We serve these markets with our GPU, Tegra processor, and related software, which are based on our proprietary technologies.
    
    RISK FACTORS
    
    Our business is subject to numerous risks and uncertainties, including those described in this section, that could adversely affect our business, financial condition, results of operations, and cash flows. These risks include, but are not limited to, the following:
    
    • We depend on third parties to manufacture, assemble, package, and test our products, and if these third parties are unable to do so on a timely basis or at all, our business could be harmed.
    • Our business is highly dependent on the success of our GPU products, and if we are unable to successfully develop and introduce new products, our business could be harmed.
    • We face intense competition in our markets, and if we are unable to compete effectively, our business could be harmed.
    • We are subject to various environmental, health and safety laws and regulations, and if we fail to comply with these laws and regulations, we could be subject to fines, penalties, and other sanctions.
    
    FINANCIAL PERFORMANCE
    
    For fiscal 2023, we reported revenue of $26.97 billion, an increase of 61% from fiscal 2022. Our Data Center segment revenue was $15.0 billion, an increase of 41% from fiscal 2022, driven by strong demand for our AI infrastructure products.
    """
    
    # Sample NVIDIA earnings call content
    nvidia_earnings = """
    NVIDIA CORPORATION
    Q4 2023 EARNINGS CALL TRANSCRIPT
    FEBRUARY 22, 2023
    
    PARTICIPANTS
    Colette Kress - Executive Vice President and Chief Financial Officer
    Jensen Huang - Founder, President and Chief Executive Officer
    
    OPERATOR: Good day, and welcome to the NVIDIA Fourth Quarter Fiscal Year 2023 Earnings Conference Call. All participants will be in listen-only mode. After today's presentation, there will be an opportunity to ask questions.
    
    JENSEN HUANG: Thank you, operator. Good afternoon, everyone. Thank you for joining us today. I'm Jensen Huang, NVIDIA's founder and CEO.
    
    We had a strong finish to fiscal 2023, with record revenue of $6.05 billion, up 2% sequentially and down 21% year-over-year. Gaming revenue was $1.83 billion, down 46% year-over-year. Data Center revenue was $3.62 billion, up 11% year-over-year.
    
    The AI revolution is driving exponential growth in computing requirements, and NVIDIA is at the center of it. Our Data Center platform is powering the AI revolution across every industry.
    
    We're seeing strong demand for our AI infrastructure, particularly for large language models and generative AI applications. Our H100 GPU is in high demand, and we're working to increase supply to meet customer needs.
    
    Looking ahead, we expect continued strong growth in our Data Center business as AI adoption accelerates across industries.
    
    COLETTE KRESS: Thank you, Jensen. Let me provide some additional financial details.
    
    Our gross margin for the quarter was 66.1%, up from 65.4% in the previous quarter. Operating expenses were $1.6 billion, up 2% sequentially. We ended the quarter with $13.3 billion in cash and marketable securities.
    
    For the first quarter of fiscal 2024, we expect revenue to be approximately $6.5 billion, plus or minus 2%. We expect gross margin to be approximately 66.5%, plus or minus 50 basis points.
    """
    
    # Sample Apple 10-K content
    apple_10k = """
    APPLE INC.
    10-K ANNUAL REPORT
    FISCAL YEAR ENDED SEPTEMBER 24, 2022
    
    ITEM 1. BUSINESS
    
    Apple Inc. ("Apple," the "Company," "we," "us," and "our") designs, manufactures, and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52 or 53-week period that ends on the last Saturday of September.
    
    The Company's business strategy leverages its unique ability to design and develop its own operating systems, hardware, application software, and services to provide its customers products and solutions with innovative design, superior ease-of-use, and seamless integration. As part of its strategy, the Company continues to expand its platform for the discovery and delivery of digital content and applications through its Digital Content and Services segment, which includes the App Store, Apple Music, Apple TV+, Apple Arcade, Apple Fitness+, Apple News+, Apple Card, Apple Books, iCloud, Apple Pay, and licensing and other services.
    
    RISK FACTORS
    
    The Company's business, financial condition, and operating results can be affected by a number of factors, whether currently known or unknown, including but not limited to those described below, any one or more of which could, directly or indirectly, cause the Company's actual financial condition and operating results to vary materially from past or anticipated financial condition and operating results.
    
    • The Company's business depends substantially on the Company's ability to continue to develop and offer new innovative products and services in a timely manner.
    • The Company's business depends on the Company's ability to obtain components and finished products from third-party suppliers and manufacturing partners.
    • The Company's business depends on the Company's ability to compete effectively in the highly competitive personal computer, mobile communication, media device, and digital content and services markets.
    
    FINANCIAL PERFORMANCE
    
    Net sales for fiscal 2022 were $394.3 billion, an increase of 8% from fiscal 2021. iPhone net sales were $205.5 billion, an increase of 7% from fiscal 2021. Services net sales were $78.1 billion, an increase of 14% from fiscal 2021.
    """
    
    # Write sample files
    with open(raw_pdfs_dir / "NVIDIA_10K_2023.txt", 'w') as f:
        f.write(nvidia_10k)
    
    with open(raw_pdfs_dir / "NVIDIA_Earnings_Q4_2023.txt", 'w') as f:
        f.write(nvidia_earnings)
    
    with open(raw_pdfs_dir / "APPLE_10K_2022.txt", 'w') as f:
        f.write(apple_10k)
    
    print("✅ Created sample documents")
    return [
        str(raw_pdfs_dir / "NVIDIA_10K_2023.txt"),
        str(raw_pdfs_dir / "NVIDIA_Earnings_Q4_2023.txt"),
        str(raw_pdfs_dir / "APPLE_10K_2022.txt")
    ]


def run_demo():
    """Run the complete EquIntel demo"""
    print("🚀 EquIntel Demo")
    print("=" * 50)
    
    # Step 1: Create sample data
    sample_files = create_sample_data()
    
    # Step 2: Initialize components
    print("\n🔧 Initializing components...")
    parser = DocumentParser()
    chunker = DocumentChunker()
    vector_store = VectorStore()
    
    # Step 3: Process documents
    print("\n📄 Processing documents...")
    for file_path in sample_files:
        print(f"  Processing {Path(file_path).name}...")
        
        # Parse document
        parsed_doc = parser.parse_pdf(file_path)
        
        # Chunk document
        chunks = chunker.chunk_document(parsed_doc.text, parsed_doc.metadata.model_dump())
        
        # Add to vector store
        vector_store.add_chunks([chunk.model_dump() for chunk in chunks])
    
    # Step 4: Initialize RAG engine
    print("\n🧠 Initializing RAG engine...")
    
    # Create a mock LLM for demo
    class DemoLLM:
        def _call(self, prompt, stop=None):
            return "Based on the provided context, I can answer your question about the financial documents. The information shows various aspects of the companies' business operations, risks, and financial performance."
    
    rag_engine = RAGEngine(vector_store, DemoLLM())
    
    # Step 5: Run demo queries
    print("\n🔍 Running demo queries...")
    
    demo_queries = [
        "What are the main risks mentioned in NVIDIA documents?",
        "How has NVIDIA's Data Center business performed?",
        "What are Apple's main business segments?",
        "Compare the revenue performance of NVIDIA and Apple"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        result = rag_engine.query(query, k=5)
        
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sources: {len(result.sources)} chunks retrieved")
        
        # Show top source
        if result.sources:
            top_source = result.sources[0]
            print(f"Top source: {top_source.get('ticker', 'Unknown')} - {top_source.get('doc_type', 'Unknown')}")
    
    # Step 6: Show statistics
    print("\n📊 System Statistics")
    print("-" * 40)
    
    stats = vector_store.get_document_stats()
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Unique tickers: {stats.get('unique_tickers', 0)}")
    print(f"Document types: {stats.get('unique_doc_types', 0)}")
    
    if stats.get('tickers'):
        print("\nCompanies in database:")
        for ticker, count in stats['tickers'].items():
            print(f"  {ticker}: {count} chunks")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 Next steps:")
    print("1. Add your own PDF documents to raw_pdfs/")
    print("2. Run: python scripts/pipeline.py --action pipeline")
    print("3. Start the web UI: streamlit run ui/app.py")


if __name__ == "__main__":
    run_demo() 