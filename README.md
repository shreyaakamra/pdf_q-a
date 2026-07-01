# PDF Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions
about uploaded PDF documents using document embeddings, FAISS vector
search, and a Hugging Face language model.

## How it works
1. PDF is uploaded and split into chunks (~600 characters each)
2. Each chunk is converted into a vector embedding
3. When a question is asked, the most relevant chunks are retrieved
4. A language model generates an answer grounded in those chunks
5. If the answer isn't in the document, the system says "I don't know"

## Setup
pip install -r requirements.txt
python app.py

## Tech stack
- LangChain (document loading, chunking, retrieval)
- sentence-transformers/all-MiniLM-L6-v2 (embeddings)
- FAISS (vector similarity search)
- google/flan-t5-large (answer generation)
- Gradio (interactive UI)

## Performance
- Processing time: ~5-30s depending on document length
- Works best with typed/printed PDFs
- Tested on documents up to 50+ pages

## Known Limitations
- Tables in PDFs may extract poorly due to layout flattening
- List-based questions may return incomplete answers (model size limitation)
- Handwritten/scanned PDFs produce noisy answers due to OCR quality
- Best suited for factual, definition-style questions
- Requires GPU for reasonable processing speed
