"""
Hyundai RAG Chatbot - Demo App
--------------------------------
RAG pipeline: Knowledge base (.txt) -> Chunking -> Embeddings (free, local)
-> ChromaDB (vector store) -> Retrieval -> Groq LLM (generation) -> Answer

Run with: streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
KB_FILE = "hyundai_knowledge_base.txt"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks so context isn't cut off
GROQ_MODEL = "openai/gpt-oss-120b"   # current recommended model on Groq (free tier)
COLLECTION_NAME = "hyundai_kb"

st.set_page_config(page_title="Hyundai Assistant", page_icon="🚗")

# ---------------------------------------------------------------------------
# STEP 1: Load & chunk the knowledge base (runs once, cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Knowledge base load ho rahi hai...")
def build_vector_store():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on section markers first (keeps each car model / FAQ block intact)
    raw_sections = text.split("=================================================================")
    sections = [s.strip() for s in raw_sections if s.strip()]

    # Further split any very long section into smaller chunks with overlap
    chunks = []
    for section in sections:
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            start = 0
            while start < len(section):
                end = start + CHUNK_SIZE
                chunks.append(section[start:end])
                start += CHUNK_SIZE - CHUNK_OVERLAP

    # Free local embedding model (no API cost, runs on CPU)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.Client()  # in-memory vector DB (fine for a demo)
    # Fresh collection each run to avoid stale data across app reloads
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


def retrieve_context(collection, query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]


# ---------------------------------------------------------------------------
# STEP 2: Call Groq LLM with retrieved context
# ---------------------------------------------------------------------------
def get_groq_client():
    # Priority: Streamlit Cloud secrets -> env variable -> manual sidebar input
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_api_key")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def generate_answer(client, query, context_chunks):
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "Tum Hyundai dealership ke liye ek helpful customer-support chatbot ho. "
        "Sirf neeche diye gaye CONTEXT ke aadhar par jawab do. "
        "Agar context me answer nahi hai, to politely bolo ki yeh jaankari "
        "available nahi hai aur dealership se contact karne ko bolo. "
        "Jawab crisp aur friendly rakho. User jis language (Hindi/English/Hinglish) "
        "me poochta hai usi me jawab do."
    )

    user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            "⚠️ LLM se jawab lene me error aaya. Details: "
            f"`{str(e)}`\n\nAgar yeh 'model decommissioned' ya 'invalid_request' bol raha hai, "
            "to console.groq.com/docs/models pe jaake current model ka naam check karo aur "
            "app.py me GROQ_MODEL update karo."
        )


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.title("🚗 Hyundai Assistant (RAG Demo)")
st.caption("Creta, Venue, Verna, Alcazar aur baaki models ke baare me kuch bhi poochho.")

# Sidebar: API key input (if not set as environment variable)
with st.sidebar:
    st.header("Setup")
    key_already_set = os.environ.get("GROQ_API_KEY") or (
        "GROQ_API_KEY" in st.secrets if hasattr(st, "secrets") else False
    )
    if not key_already_set:
        key_input = st.text_input("Groq API Key", type="password")
        if key_input:
            st.session_state["groq_api_key"] = key_input
    else:
        st.success("Groq API key already configured.")
    st.markdown("[Free Groq API key yahan se lo →](https://console.groq.com/keys)")
    st.divider()
    st.markdown("**Sample questions:**")
    st.markdown("- Creta ki price kya hai?\n- Creta aur Venue me difference?\n- EV options kya hain?\n- Family ke liye best car kaunsi hai?")

# Build (or load cached) vector store
collection = build_vector_store()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Apna sawaal likho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        client = get_groq_client()
        if client is None:
            answer = "⚠️ Pehle sidebar me apni Groq API key daalo."
        else:
            with st.spinner("Soch raha hoon..."):
                chunks = retrieve_context(collection, prompt, top_k=3)
                answer = generate_answer(client, prompt, chunks)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
