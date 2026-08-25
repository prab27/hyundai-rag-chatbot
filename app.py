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

# Known model names to catch keyword matches that semantic search sometimes
# misses on short queries like "price of creta"
MODEL_KEYWORDS = [
    "creta electric", "creta", "venue", "verna", "exter", "alcazar",
    "i20 n line", "i20", "ioniq 5", "ioniq 6", "ioniq 9", "ioniq",
    "tucson", "grand i10 nios", "nios", "aura", "bluelink",
]


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
    return collection, chunks


def retrieve_context(collection, all_chunks, query, top_k=4):
    # 1) Semantic search (catches paraphrased / conceptual questions)
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved = list(results["documents"][0])

    # 2) Keyword boost: if the query mentions a specific model by name,
    # make sure every chunk containing that model name is included too.
    # This fixes cases like "price of creta" where embeddings alone
    # sometimes miss the exact section.
    query_lower = query.lower()
    for keyword in MODEL_KEYWORDS:
        if keyword in query_lower:
            for chunk in all_chunks:
                if keyword in chunk.lower() and chunk not in retrieved:
                    retrieved.append(chunk)

    return retrieved




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
        "Jawab crisp aur friendly rakho. "
        "IMPORTANT: User jis language/script me sawaal poochta hai (Hindi/English/Hinglish/Devanagari), "
        "usi language aur usi script me jawab do — agar user Roman/Hinglish me poochta hai to jawab "
        "bhi Roman script me do, Devanagari me mat likho."
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
collection, all_chunks = build_vector_store()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None


def run_rag(query):
    """Runs retrieval + generation for a given query, returns the answer text."""
    client = get_groq_client()
    if client is None:
        return "⚠️ Pehle sidebar me apni Groq API key daalo."
    chunks = retrieve_context(collection, all_chunks, query, top_k=4)
    return generate_answer(client, query, chunks)


def regenerate_from(user_index):
    """Re-runs the RAG pipeline for the user message at user_index and
    replaces everything after it (i.e. the old assistant answer)."""
    query = st.session_state.messages[user_index]["content"]
    st.session_state.messages = st.session_state.messages[: user_index + 1]
    with st.spinner("Soch raha hoon..."):
        answer = run_rag(query)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# Render existing chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and st.session_state.editing_index == i:
            # Edit mode for this user message
            edited_text = st.text_area(
                "Sawaal edit karo:", value=msg["content"], key=f"edit_box_{i}"
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Save & Resend", key=f"save_{i}"):
                    st.session_state.messages[i]["content"] = edited_text
                    st.session_state.editing_index = None
                    regenerate_from(i)
            with col2:
                if st.button("❌ Cancel", key=f"cancel_{i}"):
                    st.session_state.editing_index = None
                    st.rerun()
        else:
            st.markdown(msg["content"])
            if msg["role"] == "user":
                if st.button("✏️ Edit", key=f"edit_btn_{i}"):
                    st.session_state.editing_index = i
                    st.rerun()
            elif msg["role"] == "assistant":
                if st.button("🔄 Regenerate", key=f"regen_btn_{i}"):
                    # user_index is the message right before this assistant reply
                    regenerate_from(i - 1)

# Chat input for new questions
if prompt := st.chat_input("Apna sawaal likho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            answer = run_rag(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
