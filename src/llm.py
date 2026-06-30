def build_prompt(question, retrieved_chunks):
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    return f"""Use the following context to answer the question. If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""


def generate_text(prompt, tokenizer, model, max_input_length=1024, max_output_length=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question(vectorstore, question, tokenizer, model, k=5):
    retrieved_chunks = vectorstore.similarity_search(question, k=k)
    prompt = build_prompt(question, retrieved_chunks)
    answer = generate_text(prompt, tokenizer, model)
    return answer, retrieved_chunks
