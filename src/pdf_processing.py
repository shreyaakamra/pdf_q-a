from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def process_pdf(filepath, embeddings, chunk_size=600, chunk_overlap=80):
    """Takes a PDF file path, returns a searchable vector store."""
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    total_chars = sum(len(p.page_content.strip()) for p in pages)
    if total_chars < 50:
        raise ValueError("This PDF appears to have no extractable text (possibly scanned/image-based).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
