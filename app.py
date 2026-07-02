import gradio as gr
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.pdf_processing import process_multiple_pdfs
from src.llm import answer_question
from src import config

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
print("Embedding model loaded.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
USE_API = config.USE_API and GROQ_API_KEY is not None

tokenizer = None
model = None
if not USE_API:
    print("No API key found, loading local model...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL)
    print("Local model loaded.")
else:
    print(f"Using Groq API ({config.API_MODEL}) for generation.")

state = {"vectorstore": None}


def upload_and_process(files):
    if not files:
        return "No files uploaded."
    try:
        filepaths = [f.name for f in files]
        vectorstore, results = process_multiple_pdfs(filepaths, embeddings)

        if vectorstore is None:
            return "❌ No valid PDFs could be processed.\n" + "\n".join(results)

        state["vectorstore"] = vectorstore
        summary = "\n".join(results)
        return f"Processing complete:\n{summary}\n\nYou can now ask questions."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_question(question):
    if not question.strip():
        return "Please enter a question.", ""
    if state["vectorstore"] is None:
        return "Please upload at least one PDF first.", ""
    try:
        answer, chunks = answer_question(
            state["vectorstore"], question,
            tokenizer=tokenizer, model=model,
            k=config.RETRIEVAL_K,
            use_api=USE_API,
            api_key=GROQ_API_KEY,
            api_model=config.API_MODEL
        )
        sources = "\n\n---\n\n".join([
            f"Source: {chunk.metadata.get('source_file', 'document')}\n"
            f"{chunk.page_content[:300]}..."
            for chunk in chunks[:3]
        ])
        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", ""


with gr.Blocks(title="PDF Q&A System") as demo:
    gr.Markdown("""
    # 📄 PDF Q&A System
    Upload one or more PDFs and ask questions about their content.
    Answers are grounded in the documents — the system will say
    **"I don't know"** if your question isn't covered.

    ⚠️ First upload may take 30-60 seconds.
    """)

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(
                label="Upload PDFs (you can select multiple)",
                file_types=[".pdf"],
                file_count="multiple"
            )
            upload_status = gr.Textbox(
                label="Processing Status",
                interactive=False,
                lines=5
            )

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a question about your documents",
                placeholder="e.g. What is the main topic?",
                lines=2
            )
            ask_button = gr.Button("Get Answer", variant="primary")

    with gr.Row():
        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=4
            )
            sources_output = gr.Textbox(
                label="Source chunks",
                interactive=False,
                lines=6
            )

    pdf_input.change(
        upload_and_process,
        inputs=pdf_input,
        outputs=upload_status
    )
    ask_button.click(
        ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    question_input.submit(
        ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

demo.launch()