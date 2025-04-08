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

more to come...
