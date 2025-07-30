#!/usr/bin/env python
# coding: utf-8

# In[1]:

import gradio as gr
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA dataset
qa_list = []
questions = []

with open("answer.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            qa_list.append(item)
            questions.append(item["question"])
        except:
            continue

# Precompute embeddings
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Chatbot function
def chatbot(user_question):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, question_embeddings, top_k=1)
    best_match = hits[0][0]
    idx = best_match["corpus_id"]
    matched_qa = qa_list[idx]
    return f"**Answer:** {matched_qa['answer']}\n\n(Matched question: _{matched_qa['question']}_)"

# Gradio UI
demo = gr.Blocks()

with demo:
    gr.Markdown("# 🤖 Aditya's AI Mentor Chatbot")
    gr.Markdown("Ask questions about AI, your projects, or the knowledge base!")

    question = gr.Textbox(label="Your Question")
    answer = gr.Markdown()
    submit = gr.Button("Get Answer")
    submit.click(chatbot, inputs=question, outputs=answer)

# Launch server (Render will auto-detect the port)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
