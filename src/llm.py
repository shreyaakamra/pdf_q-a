import os


def build_prompt(question, retrieved_chunks):
    context = "\n\n".join([
        f"[From: {chunk.metadata.get('source_file', 'document')}]\n{chunk.page_content}"
        for chunk in retrieved_chunks
    ])
    return f"""Use the following context to answer the question. If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""


def generate_text_api(prompt, api_key, model="llama3-8b-8192"):
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    return response.choices[0].message.content


def generate_text_local(prompt, tokenizer, model,
                        max_input_length=1024, max_output_length=256):
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=max_input_length
    )
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question(vectorstore, question, tokenizer=None, model=None,
                    k=5, use_api=False, api_key=None,
                    api_model="llama3-8b-8192"):
    retrieved_chunks = vectorstore.similarity_search(question, k=k)
    prompt = build_prompt(question, retrieved_chunks)

    if use_api and api_key:
        answer = generate_text_api(prompt, api_key, api_model)
    else:
        answer = generate_text_local(prompt, tokenizer, model)

    return answer, retrieved_chunks