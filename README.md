# RAG Personal AI Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about your own PDFs. Upload a document, it's chunked and embedded locally, and a Hugging Face LLM answers your questions grounded in the retrieved passages — with sources cited.

Built with LangChain, FAISS, Hugging Face Transformers, and Streamlit.

## How it works

1. Upload a PDF through the Streamlit UI.
2. The document is split into overlapping chunks (`RecursiveCharacterTextSplitter`) and embedded with `all-MiniLM-L6-v2`.
3. Chunks are indexed in a FAISS vector store for similarity search.
4. On each question, the top-matching chunks are retrieved and passed to the selected LLM (`google/flan-t5-base` or `gpt2`) via a LangChain `RetrievalQA` chain.
5. The answer is displayed along with the source chunks it was grounded in.

Model loading and retriever building are cached per-file (keyed by content hash), so re-asking questions on the same document is fast.

## Setup & run locally

```bash
git clone https://github.com/syedashraf2211/rag-personal-ai.git
cd rag-personal-ai

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file if you want to use a gated/private Hugging Face model:

```
HF_TOKEN=your_huggingface_token
```

Run the app:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a PDF, and ask questions about it.

## Notebook version

`rag-assistant-2.ipynb` walks through the same pipeline step-by-step (loading, chunking, embedding, retrieval, generation) for experimenting with different models, chunk sizes, or vector stores outside the Streamlit UI.

## Tech stack

Python · LangChain · Hugging Face Transformers · FAISS · Streamlit
