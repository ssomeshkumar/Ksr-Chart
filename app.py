import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import process
import requests
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# Load embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# Extract Q&A from PDF
@st.cache_data
def extract_qa(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        pattern = r"\n?\s*(\d{1,3})\.\s+(.+?)\n\s*➤\s+(.+?)(?=\n\s*\d{1,3}\.|$)"
        matches = re.findall(pattern, text, flags=re.DOTALL)

        questions, answers = [], []
        for _, q, a in matches:
            questions.append(q.strip())
            answers.append(a.strip())

        return questions, answers
    except Exception as e:
        st.error(f"❌ Failed to extract from PDF: {e}")
        return [], []

# Build FAISS index
@st.cache_resource
def build_index(questions):
    if not questions:
        raise ValueError("❌ No questions to embed.")
    embeddings = model.encode(questions)
    if len(embeddings) == 0 or len(embeddings[0]) == 0:
        raise ValueError("❌ Invalid embeddings.")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings

# Call OpenRouter API
def call_llm_api(context, question):
    prompt = f"Answer the user's question using only the college handbook info below:\n\n{context}\n\nUser question: {question}\nAnswer:"

    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.getenv("OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for college admission and policy FAQs."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ API Error: No valid choices.\n{data}"
    except Exception as e:
        return f"⚠️ API Exception: {e}"

# Streamlit UI
st.set_page_config(page_title="KSRCE FAQ Bot", layout="centered")
st.title("🎓 KSRCE FAQ Bot")
st.write("Ask anything about KSRCE admissions, policies, or placements.")

# Load PDF Q&A
questions, answers = extract_qa("ksrfaq1.pdf")

if not questions:
    st.error("❌ No questions extracted from the PDF.")
    st.stop()


# Build FAISS
index, embeddings = build_index(questions)

# Input
user_query = st.text_input("📩 Ask your question:")

if user_query:
    # Typo-tolerant fuzzy match (RapidFuzz)
    best_match, score, idx = process.extractOne(user_query, questions, score_cutoff=60)
    
    if best_match:
        matched_q = questions[idx]
        matched_a = answers[idx]
        context = f"Q: {matched_q}\nA: {matched_a}"
        answer = call_llm_api(context, user_query)

        st.markdown("### ✅ Answer")
        st.write(answer)

       
    else:
        st.warning("❓ Sorry, I couldn’t understand your question well. Try rephrasing.")
