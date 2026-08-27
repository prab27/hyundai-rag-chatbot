"""
Hyundai RAG Chatbot - Demo App
--------------------------------
RAG pipeline: Knowledge base (.txt) -> Chunking -> Embeddings (free, local)
-> ChromaDB (vector store) -> Retrieval -> Groq LLM (generation) -> Answer

Run with: streamlit run app.py
"""

import os
import json
import uuid
import urllib.parse
from datetime import datetime
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ---------------------------------------------------------------------------
# CHAT HISTORY STORAGE (file-based, persists across restarts)
# ---------------------------------------------------------------------------
# In a real product each user logs in and their chats live in a database keyed
# by user id. For this demo we persist all chats to a local JSON file so the
# history survives page refreshes and app restarts, just like Claude/ChatGPT.
HISTORY_FILE = "chat_history.json"


def load_all_chats():
    """Return the dict of {chat_id: {title, created_at, messages}}."""
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_all_chats(chats):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def make_chat_title(messages):
    """Use the first user message as the chat's title (like ChatGPT)."""
    for m in messages:
        if m["role"] == "user":
            title = m["content"].strip()
            return title[:40] + ("..." if len(title) > 40 else "")
    return "New chat"


def persist_current_chat():
    """Save the in-progress conversation into the history file."""
    if not st.session_state.messages:
        return
    chats = load_all_chats()
    chats[st.session_state.current_chat_id] = {
        "title": make_chat_title(st.session_state.messages),
        "created_at": st.session_state.get("current_chat_created", datetime.now().isoformat()),
        "messages": st.session_state.messages,
    }
    save_all_chats(chats)


# ---------------------------------------------------------------------------
# LEAD GENERATION (capture interested customers for dealership follow-up)
# ---------------------------------------------------------------------------
# A "lead" is a potential customer's contact info + what car they're interested
# in. The dealership sales team uses these to follow up and close a sale.
# We save every lead to a file, and optionally email it to the dealership.
LEADS_FILE = "leads.json"


def load_all_leads():
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_lead(name, phone, car_interest, message, source="form"):
    """Append a new lead to the leads file and try to email the dealership."""
    leads = load_all_leads()
    lead = {
        "name": name.strip(),
        "phone": phone.strip(),
        "car_interest": car_interest.strip(),
        "message": message.strip(),
        "source": source,  # "form" or "chat" (smart detection)
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    leads.append(lead)
    try:
        with open(LEADS_FILE, "w", encoding="utf-8") as f:
            json.dump(leads, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    email_lead_to_dealership(lead)
    return lead


def email_lead_to_dealership(lead):
    """Optionally email a lead to the dealership.
    Only runs if SMTP settings are configured in Streamlit secrets.
    Fails silently (lead is still saved to file) if not configured."""
    try:
        smtp_host = st.secrets.get("SMTP_HOST")
        smtp_user = st.secrets.get("SMTP_USER")
        smtp_pass = st.secrets.get("SMTP_PASS")
        dealer_email = st.secrets.get("DEALER_EMAIL")
        if not all([smtp_host, smtp_user, smtp_pass, dealer_email]):
            return False  # email not configured — that's fine, lead is saved

        import smtplib
        from email.mime.text import MIMEText

        body = (
            f"New lead from Hyundai chatbot:\n\n"
            f"Name: {lead['name']}\n"
            f"Phone: {lead['phone']}\n"
            f"Interested in: {lead['car_interest']}\n"
            f"Message: {lead['message']}\n"
            f"Source: {lead['source']}\n"
            f"Time: {lead['created_at']}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"New Hyundai Lead: {lead['name']} ({lead['car_interest']})"
        msg["From"] = smtp_user
        msg["To"] = dealer_email

        port = int(st.secrets.get("SMTP_PORT", 587))
        with smtplib.SMTP(smtp_host, port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception:
        return False  # any email failure shouldn't break the app


# Words that signal a customer is ready to buy / wants to be contacted
BUYING_INTENT_TRIGGERS = [
    "book", "booking", "kharid", "khareed", "buy", "purchase", "test drive",
    "test-drive", "emi", "loan", "finance", "down payment", "price quote",
    "quotation", "on road price", "on-road", "contact me", "call me",
    "interested", "delivery", "exchange", "offer chahiye",
]


def detect_buying_intent(text):
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in BUYING_INTENT_TRIGGERS)

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




@st.cache_resource(show_spinner="Loading knowledge base...")
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

    # 3) Showroom/location boost: if the user asks about showrooms, dealers,
    # addresses, timings, or names a Bangalore area, pull in the showroom
    # chunks so the bot can answer with real dealership data.
    SHOWROOM_TRIGGERS = [
        "showroom", "dealer", "dealership", "address", "location", "timing",
        "phone", "contact", "near me", "bangalore", "bengaluru", "test drive",
        "service center", "service centre",
        "vasanth nagar", "residency", "bilekahalli", "bannerghatta",
        "mysore road", "basavanagudi", "rajajinagar", "sankey", "malleshwaram",
        "electronic city", "hebbal", "bommasandra", "hsr", "kudlu",
    ]
    if any(trigger in query_lower for trigger in SHOWROOM_TRIGGERS):
        for chunk in all_chunks:
            chunk_lower = chunk.lower()
            if ("showroom" in chunk_lower or "hyundai -" in chunk_lower
                    or "phone:" in chunk_lower) and chunk not in retrieved:
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
st.caption("Ask me anything about Creta, Venue, Verna, Alcazar and other Hyundai models.")

# ---------------------------------------------------------------------------
# SESSION STATE INIT (must run before sidebar/history uses it)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "current_chat_created" not in st.session_state:
    st.session_state.current_chat_created = datetime.now().isoformat()


def start_new_chat():
    """Save the current chat, then reset to a fresh empty conversation."""
    persist_current_chat()
    st.session_state.messages = []
    st.session_state.editing_index = None
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.current_chat_created = datetime.now().isoformat()


def load_chat(chat_id):
    """Save the current chat, then load a past chat from history."""
    persist_current_chat()
    chats = load_all_chats()
    if chat_id in chats:
        st.session_state.messages = chats[chat_id]["messages"]
        st.session_state.current_chat_id = chat_id
        st.session_state.current_chat_created = chats[chat_id].get("created_at", datetime.now().isoformat())
        st.session_state.editing_index = None


def delete_chat(chat_id):
    chats = load_all_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_all_chats(chats)
    # If we deleted the chat we're currently viewing, reset to a fresh one
    if chat_id == st.session_state.current_chat_id:
        st.session_state.messages = []
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_created = datetime.now().isoformat()


# Sidebar: API key input + chat history
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
    st.markdown("[Get a free Groq API key here →](https://console.groq.com/keys)")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    # ---- Chat history list (like ChatGPT / Claude) ----
    st.markdown("**Chat history**")
    all_chats = load_all_chats()
    if not all_chats:
        st.caption("No past chats yet.")
    else:
        # Show most recent first
        sorted_chats = sorted(
            all_chats.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        )
        for cid, chat in sorted_chats:
            col1, col2 = st.columns([5, 1])
            with col1:
                label = chat.get("title", "Untitled")
                # Mark the currently open chat
                if cid == st.session_state.current_chat_id:
                    label = "▶ " + label
                if st.button(label, key=f"open_{cid}", use_container_width=True):
                    load_chat(cid)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{cid}"):
                    delete_chat(cid)
                    st.rerun()

    st.divider()
    # ---- Lead capture form ----
    st.markdown("**📋 Get a callback**")
    st.caption("Interested? Leave your details and our team will call you.")
    with st.form("lead_form", clear_on_submit=True):
        lead_name = st.text_input("Your name")
        lead_phone = st.text_input("Phone number")
        lead_car = st.selectbox(
            "Interested in",
            ["", "Creta", "Venue", "Verna", "Alcazar", "Exter", "i20",
             "Creta Electric", "Ioniq 5", "Not sure yet"],
        )
        lead_msg = st.text_area("Message (optional)", height=70)
        submitted = st.form_submit_button("Submit", use_container_width=True)
        if submitted:
            if lead_name.strip() and lead_phone.strip():
                save_lead(lead_name, lead_phone, lead_car or "Not specified",
                          lead_msg, source="form")
                st.success("Thanks! Our team will contact you soon. ✅")
            else:
                st.warning("Please enter at least your name and phone number.")

    st.divider()
    # ---- Dealership dashboard: view captured leads ----
    with st.expander("🗂️ Dealership: view leads"):
        leads = load_all_leads()
        if not leads:
            st.caption("No leads captured yet.")
        else:
            st.caption(f"Total leads: {len(leads)}")
            for ld in reversed(leads[-15:]):  # show latest 15
                st.markdown(
                    f"**{ld['name']}** · {ld['phone']}  \n"
                    f"Car: {ld['car_interest']} · via {ld['source']}  \n"
                    f"_{ld.get('message', '') or 'No message'}_  \n"
                    f"<span style='color:gray;font-size:11px'>{ld['created_at']}</span>",
                    unsafe_allow_html=True,
                )
                st.divider()

    st.divider()
    st.markdown("**Sample questions:**")
    st.markdown("- What is the price of Creta?\n- Difference between Creta and Venue?\n- What are the EV options?\n- Which is the best car for a family?")

# Build (or load cached) vector store
collection, all_chunks = build_vector_store()

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
        return "⚠️ Please enter your Groq API key in the sidebar first.", None, None, None

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
    with st.spinner("Thinking..."):
        answer, image_url, image_caption, source_label = run_rag(query, chat_history=history_before)
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "image": image_url, "image_caption": image_caption,
        "source": source_label,
    })
    persist_current_chat()
    st.rerun()


# Render existing chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and st.session_state.editing_index == i:
            # Edit mode for this user message
            edited_text = st.text_area(
                "Edit your question:", value=msg["content"], key=f"edit_box_{i}"
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

# Smart lead prompt: shown inline when the customer showed buying intent
if st.session_state.get("show_lead_prompt"):
    with st.container():
        st.info("🚗 Lagta hai aap interested hain! Apna contact chhod dijiye, "
                "hamari team aapko call karke aage ki process me help karegi.")
        with st.form("inline_lead_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                il_name = st.text_input("Name")
            with c2:
                il_phone = st.text_input("Phone")
            il_car = st.text_input("Which car are you interested in?")
            cola, colb = st.columns([1, 1])
            with cola:
                il_submit = st.form_submit_button("📞 Request callback", use_container_width=True)
            with colb:
                il_skip = st.form_submit_button("Skip", use_container_width=True)
            if il_submit:
                if il_name.strip() and il_phone.strip():
                    save_lead(il_name, il_phone, il_car or "Not specified",
                              "Captured from chat (buying intent)", source="chat")
                    st.session_state.show_lead_prompt = False
                    st.success("Thanks! Our team will call you shortly. ✅")
                    st.rerun()
                else:
                    st.warning("Please enter your name and phone number.")
            if il_skip:
                st.session_state.show_lead_prompt = False
                st.rerun()

# Chat input for new questions
raw_prompt = st.chat_input("Type your question...")
if raw_prompt and raw_prompt.strip():
    prompt = raw_prompt.strip()
    history_before = list(st.session_state.messages)  # snapshot before adding new prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
    # Smart lead detection: if the customer shows buying intent, flag it so
    # an inline "leave your contact" prompt appears after the answer.
    if detect_buying_intent(prompt):
        st.session_state.show_lead_prompt = True
    persist_current_chat()
    st.rerun()
