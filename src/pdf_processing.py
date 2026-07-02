from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def process_pdf(filepath, embeddings, chunk_size=600, chunk_overlap=80):
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    total_chars = sum(len(p.page_content.strip()) for p in pages)
    if total_chars < 50:
        raise ValueError(f"'{filepath}' appears to have no extractable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source_file"] = filepath.split("/")[-1].split("\\")[-1]

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def process_multiple_pdfs(filepaths, embeddings, chunk_size=600, chunk_overlap=80):
    combined_vectorstore = None
    results = []

    for filepath in filepaths:
        try:
            vectorstore, chunk_count = process_pdf(
                filepath, embeddings, chunk_size, chunk_overlap
            )
            filename = filepath.split("/")[-1].split("\\")[-1]
            results.append(f"✅ {filename}: {chunk_count} chunks")

            if combined_vectorstore is None:
                combined_vectorstore = vectorstore
            else:
                combined_vectorstore.merge_from(vectorstore)

        except ValueError as e:
            results.append(f"⚠️ {str(e)}")
        except Exception as e:
            filename = filepath.split("/")[-1].split("\\")[-1]
            results.append(f"❌ {filename}: {str(e)}")

    return combined_vectorstore, results