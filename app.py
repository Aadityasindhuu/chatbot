#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ✅ Cell 1: Install required packages (Run this first)
get_ipython().system('pip install -q gradio sentence-transformers')


# In[2]:


import gradio as gr
import json
from sentence_transformers import SentenceTransformer, util


# In[3]:


model = SentenceTransformer("all-MiniLM-L6-v2")


# In[4]:


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


# In[5]:


question_embeddings = model.encode(questions, convert_to_tensor=True)


# In[6]:


# ✅ Cell 6: Define the chatbot logic
def chatbot(user_question):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, question_embeddings, top_k=1)
    best_match = hits[0][0]
    idx = best_match["corpus_id"]
    matched_qa = qa_list[idx]
    
    return f"**Answer:** {matched_qa['answer']}\n\n(Matched question: _{matched_qa['question']}_)"


# In[8]:


with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Aditya's AI Mentor Chatbot")
    gr.Markdown("Ask questions about AI, your projects, or the knowledge base!")

    question = gr.Textbox(label="Your Question")
    answer = gr.Markdown()
    submit = gr.Button("Get Answer")

    submit.click(chatbot, inputs=question, outputs=answer)

demo.launch(share=True)


# In[ ]:




