"""
Hyundai RAG Chatbot - Demo App
--------------------------------
RAG pipeline: Knowledge base (.txt) -> Chunking -> Embeddings (free, local)
-> ChromaDB (vector store) -> Retrieval -> Groq LLM (generation) -> Answer

Run with: streamlit run app.py
"""

import os
import urllib.parse
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

# In a real production system, each car's image comes from the company's own
# CDN/S3 bucket (e.g. https://cdn.hyundai.com/creta/hero.jpg), stored alongside
# the rest of that model's structured data. For this demo we use stable,
# freely-licensed Wikimedia Commons images so the links won't break.
MODEL_IMAGES = {
    "creta electric": "Hyundai_Creta_Electric_SU2_EV_PE_(3).jpg",
    "creta": "Hyundai_Creta_India.jpg",
    "venue": "Hyundai_Venue.jpg",
    "verna": "HYUNDAI_VERNA_(RC).jpg",
    "i20": "Hyundai_i20_(BC3)_IMG_4165.jpg",
    "alcazar": "2021_Hyundai_Alcazar_2.0_Signature_(India)_front_view.png",
    "ioniq 5": "Hyundai_Ioniq_5.jpg",
}


def get_model_image(query):
    """Detect a car model in the query and return (image_url, caption) or None."""
    query_lower = query.lower()
    for keyword in MODEL_KEYWORDS:
        if keyword in query_lower and keyword in MODEL_IMAGES:
            filename = MODEL_IMAGES[keyword]
            encoded = urllib.parse.quote(filename, safe="()_.-")
            url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}?width=500"
            caption = f"Hyundai {keyword.title()}"
            return url, caption
    return None, None


# ---------------------------------------------------------------------------
# LIVE WEB SEARCH FALLBACK
# ---------------------------------------------------------------------------
# Real production chatbots don't rely only on a static knowledge base for
# time-sensitive info (today's offers, latest launch news, current prices).
# Here we fall back to a live web search when either:
#   (a) the query itself signals it wants "current/live" info, or
#   (b) our RAG answer came back empty-handed (bot said it doesn't know)
LIVE_INFO_TRIGGERS = [
    "today", "aaj", "current", "abhi", "latest", "newest", "is week",
    "is month", "recent", "offer", "discount", "news",
]

NOT_FOUND_PHRASES = [
    "don't have", "dont have", "not available", "no information",
    "reach out to", "nearest hyundai dealership", "contact the dealership",
    "please reach out", "i'm sorry, but i don't",
]


def needs_live_search(query, rag_answer):
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in LIVE_INFO_TRIGGERS):
        return True
    answer_lower = rag_answer.lower()
    if any(phrase in answer_lower for phrase in NOT_FOUND_PHRASES):
        return True
    return False


def web_search_fallback(query):
    """Free, no-API-key web search (DuckDuckGo) used only when the static
    knowledge base can't answer. Returns a list of short text snippets."""
    try:
        from ddgs import DDGS
        results = DDGS().text(f"Hyundai India {query}", max_results=4)
        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            if body:
                snippets.append(f"{title}: {body}")
        return snippets
    except Exception:
        return []


def friendly_groq_error(e):
    """Turn a raw Groq/HTTP exception into a clear, non-technical message."""
    msg = str(e).lower()
    if "401" in msg or "invalid_api_key" in msg or "unauthorized" in msg:
        return "⚠️ Groq API key invalid lag rahi hai. Sidebar/Secrets me sahi key check karo."
    if "429" in msg or "rate limit" in msg:
        return "⚠️ Thoda zyada traffic ho gaya (rate limit). Kuch second baad phir try karo."
    if "model_decommissioned" in msg or "decommissioned" in msg:
        return (
            "⚠️ Yeh AI model ab discontinue ho chuka hai. "
            "console.groq.com/docs/models pe current model dekh ke app.py me GROQ_MODEL update karo."
        )
    if "timeout" in msg or "connection" in msg:
        return "⚠️ Network/connection issue aaya. Ek baar phir try karo."
    return f"⚠️ Kuch galat ho gaya. Technical details: `{str(e)[:200]}`"


def generate_answer_with_web(client, query, web_snippets):
    context_text = "\n\n---\n\n".join(web_snippets)
    system_prompt = (
        "Tum Hyundai dealership ke liye ek helpful customer-support chatbot ho. "
        "Neeche diye gaye LIVE WEB SEARCH RESULTS ke aadhar par jawab do — yeh "
        "current/real-time info hai. Jawab crisp aur friendly rakho. "
        "User jis language/script me sawaal poochta hai usi me jawab do "
        "(Roman/Hinglish me poocha hai to Roman script me hi jawab do, Devanagari mat likho). "
        "Agar in results me bhi answer na mile to politely bolo dealership se "
        "confirm kar lein kyunki yeh info fast-changing hai."
    )
    user_prompt = f"LIVE WEB SEARCH RESULTS:\n{context_text}\n\nQUESTION: {query}"
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as e:
        return friendly_groq_error(e)




@st.cache_resource(show_spinner="Knowledge base load ho rahi hai...")
def build_vector_store():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on section markers first (keeps each car model / FAQ block intact)
    raw_sections = text.split("=================================================================")
    sections = [s.strip() for s in raw_sections if s.strip()]

    # Further split any very long section into smaller chunks — but split on
    # LINE boundaries, never mid-line. This matters a lot for this dataset:
    # the comparison table and FAQ section have long lines/rows that must
    # stay intact, otherwise a car's price row (or a Q without its A) could
    # get cut in half and the bot would answer with garbled/wrong data.
    chunks = []
    for section in sections:
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
            continue

        lines = section.split("\n")
        current_chunk_lines = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1  # +1 for the newline
            if current_len + line_len > CHUNK_SIZE and current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                # start next chunk with a small overlap: repeat the last
                # couple of lines so context isn't lost at the boundary
                overlap_lines = current_chunk_lines[-2:] if len(current_chunk_lines) >= 2 else current_chunk_lines
                current_chunk_lines = list(overlap_lines)
                current_len = sum(len(l) + 1 for l in current_chunk_lines)
            current_chunk_lines.append(line)
            current_len += line_len
        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

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


def generate_answer(client, query, context_chunks, chat_history=None):
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "Tum Hyundai dealership ke liye ek helpful customer-support chatbot ho. "
        "Sirf neeche diye gaye CONTEXT ke aadhar par jawab do. "
        "Agar context me answer nahi hai, to politely bolo ki yeh jaankari "
        "available nahi hai aur dealership se contact karne ko bolo. "
        "Jawab crisp aur friendly rakho. "
        "IMPORTANT: User jis language/script me sawaal poochta hai (Hindi/English/Hinglish/Devanagari), "
        "usi language aur usi script me jawab do — agar user Roman/Hinglish me poochta hai to jawab "
        "bhi Roman script me do, Devanagari me mat likho. "
        "IMPORTANT: Neeche CONVERSATION HISTORY di gayi hai — agar user follow-up sawaal poochhe "
        "(jaise 'aur detail do', 'iske baare me aur batao', 'price kya hai iski'), to history dekh ke "
        "samjho wo kis cheez ke baare me baat kar raha hai, aur usi topic ka detail do."
    )

    # Build the messages list: system prompt + recent chat history + current question
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        # Only send the last few turns to keep the prompt small and relevant
        for turn in chat_history[-6:]:
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["content"]})

    user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as e:
        return friendly_groq_error(e)


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
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.editing_index = None
        st.rerun()
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


FOLLOWUP_TRIGGERS = [
    "aur detail", "aur batao", "iske baare", "uske baare", "isme", "usme",
    "iska", "uska", "iski", "uski", "isse", "usse", "more detail",
    "tell me more", "explain more", "aur bata",
]


def build_retrieval_query(query, chat_history):
    """If this looks like a short follow-up ('aur detail do'), pull in the
    last user message so retrieval still knows which car/topic we're on.
    But if the new query already names its own model (e.g. user switches
    from Creta to just 'venue'), treat it as a fresh topic and don't merge."""
    query_lower = query.lower()
    mentions_own_model = any(kw in query_lower for kw in MODEL_KEYWORDS)
    has_followup_trigger = any(trigger in query_lower for trigger in FOLLOWUP_TRIGGERS)
    is_short = len(query.split()) <= 4

    if mentions_own_model and not has_followup_trigger:
        return query  # already has a clear subject of its own, no need to merge

    if (has_followup_trigger or is_short) and chat_history:
        last_user_msgs = [m["content"] for m in chat_history if m["role"] == "user"]
        if last_user_msgs:
            return f"{last_user_msgs[-1]} {query}"
    return query


def run_rag(query, chat_history=None):
    """Runs retrieval + generation for a given query.
    Returns (answer_text, image_url_or_None, image_caption_or_None, source_label)."""
    client = get_groq_client()
    if client is None:
        return "⚠️ Pehle sidebar me apni Groq API key daalo.", None, None, None

    # For follow-ups like "aur detail do", widen the retrieval query using
    # the previous user message so we fetch chunks about the right topic
    retrieval_query = build_retrieval_query(query, chat_history)

    # Step 1: Try the static knowledge base first (fast, free, no rate limits)
    chunks = retrieve_context(collection, all_chunks, retrieval_query, top_k=4)
    answer = generate_answer(client, query, chunks, chat_history=chat_history)
    source_label = "📚 Knowledge base"

    # Step 2: If the query wants live info, or KB came up empty, fall back
    # to a real-time web search
    if needs_live_search(query, answer):
        snippets = web_search_fallback(retrieval_query)
        if snippets:
            answer = generate_answer_with_web(client, query, snippets)
            source_label = "🌐 Live web search"
        # if web search itself returns nothing, we just keep the KB answer

    image_url, image_caption = get_model_image(query)
    if not image_url:
        # Raw query had no model name (pure follow-up like "aur detail do") —
        # fall back to the merged retrieval query which pulls in the topic
        image_url, image_caption = get_model_image(retrieval_query)
    return answer, image_url, image_caption, source_label


def regenerate_from(user_index):
    """Re-runs the RAG pipeline for the user message at user_index and
    replaces everything after it (i.e. the old assistant answer)."""
    query = st.session_state.messages[user_index]["content"]
    history_before = st.session_state.messages[:user_index]
    st.session_state.messages = st.session_state.messages[: user_index + 1]
    with st.spinner("Soch raha hoon..."):
        answer, image_url, image_caption, source_label = run_rag(query, chat_history=history_before)
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "image": image_url, "image_caption": image_caption,
        "source": source_label,
    })
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
                    if edited_text.strip():
                        st.session_state.messages[i]["content"] = edited_text.strip()
                        st.session_state.editing_index = None
                        regenerate_from(i)
                    else:
                        st.warning("Khaali sawaal save nahi ho sakta.")
            with col2:
                if st.button("❌ Cancel", key=f"cancel_{i}"):
                    st.session_state.editing_index = None
                    st.rerun()
        else:
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("image"):
                st.image(msg["image"], caption=msg.get("image_caption"), width=400)
            if msg["role"] == "assistant" and msg.get("source"):
                st.caption(f"Source: {msg['source']}")
            if msg["role"] == "user":
                if st.button("✏️ Edit", key=f"edit_btn_{i}"):
                    st.session_state.editing_index = i
                    st.rerun()
            elif msg["role"] == "assistant":
                if st.button("🔄 Regenerate", key=f"regen_btn_{i}"):
                    # user_index is the message right before this assistant reply
                    regenerate_from(i - 1)

# Chat input for new questions
raw_prompt = st.chat_input("Apna sawaal likho...")
if raw_prompt and raw_prompt.strip():
    prompt = raw_prompt.strip()
    history_before = list(st.session_state.messages)  # snapshot before adding new prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            answer, image_url, image_caption, source_label = run_rag(prompt, chat_history=history_before)
        st.markdown(answer)
        if image_url:
            st.image(image_url, caption=image_caption, width=400)
        if source_label:
            st.caption(f"Source: {source_label}")

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "image": image_url, "image_caption": image_caption,
        "source": source_label,
    })
    st.rerun()
