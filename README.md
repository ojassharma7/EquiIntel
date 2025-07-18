# 📊 EquIntel – AI-Powered Equity Research Analyst (RAG + LLM)

**EquiIntel** is a local-first, AI-powered equity research platform that leverages cutting-edge Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques to analyze financial documents, research reports, and market insights — all from your own database of PDFs.

Think of it as your very own BloombergGPT-lite that runs *entirely locally*.

---

## 🚀 Features

- ✅ **Upload & Ingest PDFs** (research reports, IMF docs, 10-Ks, earnings reports)
- 🧠 **Local LLM (Mistral 7B)** for secure & fast generation
- 🔍 **Semantic Search with FAISS** — find relevant info across hundreds of documents
- 🧾 **Metadata Tagging** — categorize by sector, source, date, etc.
- 💬 **RAG-based Q&A Interface** — ask financial questions and get contextual answers
- 📊 **Pluggable Analytics Layer** (coming soon!) — sentiment tracking, equity scores, view comparisons

---

## 🏗️ Tech Stack

| Component        | Tool/Library                          |
|------------------|----------------------------------------|
| LLM              | `llama-cpp-python` (Mistral 7B GGUF)   |
| Embeddings       | `sentence-transformers`               |
| Vector Store     | `FAISS`                               |
| PDF Parsing      | `PyPDF2`, `LangChain PDF Loaders`     |
| RAG Framework    | `LangChain`                           |
| UI               | `Streamlit` (for later phases)        |
| Metadata Handling| Custom parser based on filename logic |

---

## 🧱 Project Structure

# 📊 EquIntel – AI-Powered Equity Research Analyst

**EquIntel** is a local-first AI-powered equity research tool that leverages advanced LLMs and RAG techniques to analyze financial documents such as 10-Ks, earnings calls, and analyst reports.

Think of it as your private BloombergGPT — built for accuracy, privacy, and power.

---

## 🔧 Architecture Overview

- 📁 Parse and ingest financial PDFs (10-Ks, 10-Qs, earnings calls)
- ✂️ Chunk text into semantically meaningful units
- 🧠 Embed chunks using sentence-transformers or OpenAI embeddings
- 📦 Store vectors + metadata in **LanceDB**
- 🔍 Perform vector search with contextual filtering (e.g., ticker, year, section)
- 🤖 Use RAG to answer complex queries using a local or API-based LLM

---

## 🗃️ LanceDB for Vector Storage

We use **[LanceDB](https://github.com/lancedb/lancedb)** as our vector database to store:
- Embeddings of parsed financial documents
- Structured metadata: ticker, doc type, section, date, etc.
- Enables efficient semantic + filtered retrieval

---

## 🧱 LanceDB Schema

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
📥 Getting Started
bash
Copy
Edit
git clone https://github.com/your-org/equintel.git
cd equintel
pip install -r requirements.txt
python pipeline/download_and_embed.py  # Build your dataset
🧠 Example Queries
"Summarize Nvidia’s AI focus over the last 3 earnings calls."

"Compare capex growth of TSM and INTC over 2 quarters."

"List companies with declining gross margins in 2023."

🔒 Privacy & Local Execution
EquIntel is 100% local-first:

No cloud calls unless explicitly configured

Suitable for internal, proprietary, or institutional research use

📜 License
MIT

yaml
Copy
Edit

---

Would you like me to write the full `lancedb` storage code next, including:
- Vector insertion
- Metadata tagging
- Query filter with `where` clauses?

Or update your repo with that automatically?
