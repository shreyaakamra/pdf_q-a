import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.pdf_processing import process_pdf
from src.llm import answer_question
from src import config

print("Loading models...")
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL)
print("Models loaded.")

current_vectorstore = {"store": None}

def upload_pdf(file):
    try:
        vectorstore = process_pdf(file.name, embeddings, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        current_vectorstore["store"] = vectorstore
        return "PDF processed successfully. You can now ask questions."
    except Exception as e:
        return f"Error: {str(e)}"

def ask(question):
    if current_vectorstore["store"] is None:
        return "Please upload a PDF first."
    answer, _ = answer_question(current_vectorstore["store"], question, tokenizer, model, k=config.RETRIEVAL_K)
    return answer

with gr.Blocks() as demo:
    gr.Markdown("# PDF Q&A System")
    pdf_input = gr.File(label="Upload PDF")
    upload_status = gr.Textbox(label="Status")
    pdf_input.change(upload_pdf, inputs=pdf_input, outputs=upload_status)

    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")
    question_input.submit(ask, inputs=question_input, outputs=answer_output)

demo.launch()
