"""
EquIntel Streamlit UI
Web interface for the EquIntel financial document analysis platform
"""

import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path
import time

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from document_parser import DocumentParser
from chunker import DocumentChunker
from vector_store import VectorStore
from rag_engine import RAGEngine, create_llm


def init_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def initialize_components():
    """Initialize pipeline components"""
    if st.session_state.pipeline is None:
        try:
            # Initialize components
            vector_store = VectorStore()
            llm = create_llm("openai")  # Default to OpenAI
            rag_engine = RAGEngine(vector_store, llm)
            
            st.session_state.vector_store = vector_store
            st.session_state.rag_engine = rag_engine
            
            return True
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return False
    return True


def main():
    st.set_page_config(
        page_title="EquIntel - AI-Powered Equity Research",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 EquIntel - AI-Powered Equity Research Analyst")
    st.markdown("Local-first AI-powered equity research platform using RAG + LLM")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📤 Upload Documents", "🔍 Query Documents", "📊 Statistics"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload Documents":
        show_upload_page()
    elif page == "🔍 Query Documents":
        show_query_page()
    elif page == "📊 Statistics":
        show_statistics_page()


def show_dashboard():
    """Show the main dashboard"""
    st.header("🏠 Dashboard")
    
    # Initialize components
    if not initialize_components():
        st.error("Failed to initialize components. Please check your configuration.")
        return
    
    # Get stats
    try:
        stats = st.session_state.vector_store.get_document_stats()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", stats.get('total_chunks', 0))
        
        with col2:
            st.metric("Unique Tickers", stats.get('unique_tickers', 0))
        
        with col3:
            st.metric("Document Types", stats.get('unique_doc_types', 0))
        
        with col4:
            year_range = stats.get('year_range', {})
            if year_range.get('min') and year_range.get('max'):
                st.metric("Year Range", f"{year_range['min']}-{year_range['max']}")
            else:
                st.metric("Year Range", "N/A")
        
        # Display document types
        if stats.get('doc_types'):
            st.subheader("📄 Document Types")
            doc_types_df = pd.DataFrame(
                list(stats['doc_types'].items()),
                columns=['Document Type', 'Count']
            )
            st.bar_chart(doc_types_df.set_index('Document Type'))
        
        # Display tickers
        if stats.get('tickers'):
            st.subheader("🏢 Companies")
            tickers_df = pd.DataFrame(
                list(stats['tickers'].items()),
                columns=['Ticker', 'Chunks']
            )
            st.dataframe(tickers_df.sort_values('Chunks', ascending=False))
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")


def show_upload_page():
    """Show the document upload page"""
    st.header("📤 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload financial documents (10-Ks, 10-Qs, earnings calls, etc.)"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files selected")
        
        # Show file details
        file_details = []
        for file in uploaded_files:
            file_details.append({
                'Name': file.name,
                'Size': f"{file.size / 1024:.1f} KB",
                'Type': file.type
            })
        
        st.dataframe(pd.DataFrame(file_details))
        
        # Process button
        if st.button("🚀 Process Documents", type="primary"):
            process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    with st.spinner("Processing documents..."):
        try:
            # Save uploaded files
            raw_pdfs_dir = Path("raw_pdfs")
            raw_pdfs_dir.mkdir(exist_ok=True)
            
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = raw_pdfs_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(str(file_path))
            
            # Initialize parser
            parser = DocumentParser()
            
            # Parse documents
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(saved_files):
                status_text.text(f"Parsing {Path(file_path).name}...")
                try:
                    parser.parse_and_save(file_path)
                except Exception as e:
                    st.error(f"Error parsing {file_path}: {e}")
                
                progress_bar.progress((i + 1) / len(saved_files))
            
            status_text.text("Chunking documents...")
            
            # Chunk documents
            chunker = DocumentChunker()
            for parsed_file in Path("parsed_data").glob("*_parsed.json"):
                try:
                    with open(parsed_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metadata = data.get('metadata', {})
                    text = data['text']
                    chunks = chunker.chunk_document(text, metadata)
                    
                    output_file = str(parsed_file).replace('_parsed.json', '_chunks.json')
                    chunker.save_chunks(chunks, output_file)
                    
                except Exception as e:
                    st.error(f"Error chunking {parsed_file}: {e}")
            
            status_text.text("Embedding documents...")
            
            # Embed documents
            if not initialize_components():
                st.error("Failed to initialize vector store")
                return
            
            total_chunks = 0
            for chunks_file in Path("parsed_data").glob("*_chunks.json"):
                try:
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    
                    st.session_state.vector_store.add_chunks(chunks)
                    total_chunks += len(chunks)
                    
                except Exception as e:
                    st.error(f"Error embedding {chunks_file}: {e}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            st.success(f"Successfully processed {len(saved_files)} documents with {total_chunks} chunks!")
            
            # Refresh stats
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing documents: {e}")


def show_query_page():
    """Show the query page"""
    st.header("🔍 Query Documents")
    
    if not initialize_components():
        st.error("Failed to initialize components. Please check your configuration.")
        return
    
    # Query input
    query = st.text_area(
        "Enter your financial question:",
        placeholder="e.g., What are the main risks mentioned in NVIDIA earnings calls?",
        height=100
    )
    
    # Filters
    st.subheader("🔧 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker_filter = st.text_input("Ticker Symbol", placeholder="e.g., NVDA")
    
    with col2:
        doc_type_filter = st.selectbox(
            "Document Type",
            ["", "10-K", "10-Q", "8-K", "earnings_call", "annual_report", "quarterly_report"]
        )
    
    with col3:
        year_filter = st.number_input("Year", min_value=1990, max_value=2030, value=None)
    
    # Query parameters
    k = st.slider("Number of chunks to retrieve", min_value=1, max_value=20, value=10)
    
    # Submit button
    if st.button("🔍 Search", type="primary") and query:
        run_query(query, k, ticker_filter, doc_type_filter, year_filter)


def run_query(query, k, ticker_filter, doc_type_filter, year_filter):
    """Run a query and display results"""
    with st.spinner("Searching documents..."):
        try:
            # Build filters
            filters = {}
            if ticker_filter:
                filters['ticker'] = ticker_filter
            if doc_type_filter:
                filters['doc_type'] = doc_type_filter
            if year_filter:
                filters['year'] = year_filter
            
            # Run query
            result = st.session_state.rag_engine.query(
                query, 
                k=k, 
                filters=filters if filters else None
            )
            
            # Display results
            st.subheader("📝 Answer")
            st.write(result.answer)
            
            # Confidence and metadata
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{result.confidence:.2f}")
            with col2:
                st.metric("Sources", len(result.sources))
            
            # Display sources
            if result.sources:
                st.subheader("📚 Sources")
                
                for i, source in enumerate(result.sources, 1):
                    with st.expander(f"Source {i}: {source.get('ticker', 'Unknown')} - {source.get('doc_type', 'Unknown')}"):
                        st.write(f"**File:** {source.get('file_path', 'Unknown')}")
                        if source.get('year'):
                            st.write(f"**Year:** {source['year']}")
                        if source.get('section'):
                            st.write(f"**Section:** {source['section']}")
                        st.write(f"**Preview:** {source.get('text_preview', 'No preview available')}")
            
            # Save result
            if st.button("💾 Save Result"):
                output_file = f"query_result_{int(time.time())}.json"
                with open(output_file, 'w') as f:
                    json.dump(result.model_dump(), f, indent=2)
                st.success(f"Result saved to {output_file}")
            
        except Exception as e:
            st.error(f"Error running query: {e}")


def show_statistics_page():
    """Show detailed statistics"""
    st.header("📊 Statistics")
    
    if not initialize_components():
        st.error("Failed to initialize components. Please check your configuration.")
        return
    
    try:
        stats = st.session_state.vector_store.get_document_stats()
        
        # Display all statistics
        st.json(stats)
        
        # Download stats
        if st.button("📥 Download Statistics"):
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                label="Download JSON",
                data=stats_json,
                file_name="equintel_stats.json",
                mime="application/json"
            )
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")


if __name__ == "__main__":
    main() 