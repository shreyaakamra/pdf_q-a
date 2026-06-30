# PDF Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about uploaded PDF documents using document embeddings, FAISS vector search, and a Hugging Face language model.

## How it works
1. PDF is split into chunks and converted into embeddings
2. Relevant chunks are retrieved based on the user's question
3. A language model generates an answer grounded in the retrieved context

## Setup
\`\`\`
pip install -r requirements.txt
python app.py
\`\`\`

## Tech stack
- LangChain (document loading, chunking, vector store)
- sentence-transformers (embeddings)
- FAISS (vector search)
- Hugging Face Transformers (flan-t5-large for generation)
- Gradio (UI)
