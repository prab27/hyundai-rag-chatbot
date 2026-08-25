# Hyundai RAG Chatbot — Demo Setup

## Files
- `app.py` — main Streamlit RAG chatbot app
- `hyundai_knowledge_base.txt` — knowledge base (must be in the same folder as app.py)
- `requirements.txt` — Python dependencies

## Steps to run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key**
   - Go to https://console.groq.com/keys
   - Sign up (free) and create an API key

3. **Set the API key** (either option works)

   Option A — environment variable:
   ```bash
   export GROQ_API_KEY="your_key_here"      # Mac/Linux
   set GROQ_API_KEY=your_key_here           # Windows CMD
   ```

   Option B — paste it directly into the sidebar text box when the app opens
   (no env variable needed).

4. **Make sure `hyundai_knowledge_base.txt` is in the same folder as `app.py`**

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Browser me automatically khul jayega — usually at http://localhost:8501

## How it works (for interview explanation)
1. Knowledge base text file ko sections me split kiya jata hai (chunking)
2. Har chunk ko `all-MiniLM-L6-v2` model se embedding (vector) me convert kiya jata hai — yeh free, local model hai, koi API cost nahi
3. Saare embeddings ChromaDB (vector database) me store hote hain
4. User jab query karta hai, uski query bhi embed hoti hai aur ChromaDB se sabse relevant 3 chunks retrieve kiye jaate hain (semantic search)
5. Retrieved chunks + user ka question Groq LLM (Llama 3.3 70B) ko diya jata hai
6. LLM sirf diye gaye context ke aadhar par answer generate karta hai (hallucination kam hoti hai)

## Notes
- Vector store in-memory hai (Chroma `Client()`), toh restart karne pe dobara build hoga — demo ke liye yeh fine hai
- Agar production me deploy karna ho to `chromadb.PersistentClient()` use karo taaki data disk pe save rahe
- Hosting ke liye: Streamlit Community Cloud (free), Render, ya Railway pe deploy kar sakte ho
