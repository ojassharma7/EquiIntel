# 📊 EquIntel – AI-Powered Equity Research Analyst (RAG + LLM)

**EquIntel** is a local-first, AI-powered equity research platform that leverages cutting-edge Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques to analyze financial documents, research reports, and market insights — all from your own database of PDFs.

Think of it as your very own BloombergGPT-lite that runs *entirely locally*.

---

## 🚀 Features

- ✅ **Upload & Ingest PDFs** (10-Ks, 10-Qs, earnings calls, analyst reports)
- 🧠 **Local or API-based LLMs** (OpenAI, local models via llama-cpp-python)
- 🔍 **Semantic Search with LanceDB** — find relevant info across hundreds of documents
- 🧾 **Smart Metadata Extraction** — auto-detect ticker, doc type, date, sections
- 💬 **RAG-based Q&A Interface** — ask financial questions and get contextual answers
- 📊 **Web UI with Streamlit** — beautiful interface for document upload and querying
- 🔒 **100% Local-First** — your data stays on your machine

---

## 🏗️ Tech Stack

| Component        | Tool/Library                          |
|------------------|----------------------------------------|
| LLM              | `OpenAI API` / `llama-cpp-python`      |
| Embeddings       | `sentence-transformers`               |
| Vector Store     | `LanceDB`                             |
| PDF Parsing      | `PyMuPDF`, `pdfplumber`, `unstructured` |
| RAG Framework    | `LangChain`                           |
| UI               | `Streamlit`                           |
| Metadata Handling| Custom parser with regex patterns     |

---

## 🧱 Project Structure

```
EquIntel/
├── app/                    # Core application modules
│   ├── document_parser.py  # PDF parsing and metadata extraction
│   ├── chunker.py         # Text chunking for embeddings
│   ├── vector_store.py    # LanceDB vector storage
│   └── rag_engine.py      # RAG query engine
├── ui/                    # Web interface
│   └── app.py            # Streamlit UI
├── scripts/               # Utility scripts
│   ├── pipeline.py       # Main processing pipeline
│   ├── setup.py          # Environment setup
│   └── download_sample_data.py
├── raw_pdfs/             # Input PDF files
├── parsed_data/          # Extracted text and metadata
├── embeddings/           # LanceDB vector store
├── metadata/             # KPI time series (future)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🗃️ LanceDB Vector Storage

We use **[LanceDB](https://github.com/lancedb/lancedb)** as our vector database to store:
- Embeddings of parsed financial documents
- Structured metadata: ticker, doc type, section, date, etc.
- Enables efficient semantic + filtered retrieval

### Schema
```python
{
  "chunk_id": str,
  "text": str,
  "embedding": List[float],
  "ticker": str,
  "doc_type": str,
  "quarter": str,
  "year": int,
  "section": str,
  "source": str
}
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/equintel.git
cd equintel

# Run setup script
python scripts/setup.py
```

### 2. Configure API Keys

Edit the `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
LLM_TYPE=openai  # or 'local' for local models
```

### 3. Add Documents

Place your PDF files in the `raw_pdfs/` directory:
```bash
cp your_documents/*.pdf raw_pdfs/
```

### 4. Process Documents

```bash
# Run the full pipeline
python scripts/pipeline.py --action pipeline

# Or run individual steps
python scripts/pipeline.py --action parse
python scripts/pipeline.py --action chunk
python scripts/pipeline.py --action embed
```

### 5. Query Documents

```bash
# Command line query
python scripts/pipeline.py --action query --query "What are the main risks mentioned in NVIDIA documents?"

# Or start the web interface
streamlit run ui/app.py
```

---

## 🧠 Example Queries

- **"Summarize NVIDIA's AI focus over the last 3 earnings calls"**
- **"Compare capex growth of TSM and INTC over 2 quarters"**
- **"List companies with declining gross margins in 2023"**
- **"What are the main regulatory risks mentioned in Meta's 10-K?"**
- **"How has Apple's revenue from services changed over time?"**

---

## 🔧 Advanced Usage

### Command Line Interface

```bash
# Process documents with custom parameters
python scripts/pipeline.py --action pipeline --pdf-dir custom_pdfs

# Query with filters
python scripts/pipeline.py --action query \
  --query "AI investments" \
  --ticker NVDA \
  --doc-type earnings_call \
  --year 2023

# Get statistics
python scripts/pipeline.py --action stats
```

### Programmatic Usage

```python
from app import VectorStore, RAGEngine, create_llm

# Initialize components
vector_store = VectorStore()
llm = create_llm("openai")
rag_engine = RAGEngine(vector_store, llm)

# Run a query
result = rag_engine.query("What are the main risks mentioned?")
print(result.answer)
```

### Local LLM Setup

For local models using llama-cpp-python:

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Download a GGUF model (e.g., Mistral 7B)
# Place it in your models directory

# Update .env
LLM_TYPE=local
LOCAL_MODEL_PATH=/path/to/your/model.gguf
```

---

## 🔒 Privacy & Local Execution

EquIntel is 100% local-first:

- **No cloud calls** unless explicitly configured for OpenAI API
- **Your data stays on your machine** - no data sent to external servers
- **Suitable for internal, proprietary, or institutional research**
- **Compliant with data privacy regulations**

---

## 📊 Web Interface

The Streamlit web interface provides:

- **📤 Document Upload**: Drag-and-drop PDF upload
- **🔍 Query Interface**: Natural language question answering
- **📊 Dashboard**: Statistics and document overview
- **🔧 Filters**: Filter by ticker, document type, year
- **📚 Source Citations**: View source documents for answers

Start the interface:
```bash
streamlit run ui/app.py
```

---

## 🛠️ Development

### Adding New Document Types

Extend the `DocumentParser` class to support new document formats:

```python
class CustomDocumentParser(DocumentParser):
    def extract_metadata_from_content(self, text: str, filename: str):
        # Add custom metadata extraction logic
        pass
```

### Custom Embedding Models

Replace the default embedding model:

```python
from sentence_transformers import SentenceTransformer

# Use a different model
embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

### Adding New LLM Providers

Extend the LLM interface:

```python
class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Implement your LLM integration
        pass
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/equintel/issues)
- **Documentation**: [Wiki](https://github.com/your-username/equintel/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/equintel/discussions)
