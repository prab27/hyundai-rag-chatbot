# LIVE Website Deploy Karne Ke Steps (Free, 10 min)

Hum **Streamlit Community Cloud** use karenge — bilkul free, aur ek public
URL milta hai (jaise `https://hyundai-assistant.streamlit.app`) jo tu
kisi ko bhi bhej sakta hai, wo browser me khol ke bot se chat kar sakta hai.

---

## STEP 1: GitHub par code daalo (5 min)

1. https://github.com pe account banao (agar nahi hai)
2. New repository banao — naam do jaise `hyundai-rag-chatbot`
   - Public rakho
3. In 3 files ko us repo me upload karo (GitHub website pe "Add file" →
   "Upload files" se seedha kar sakte ho, koi git command nahi chahiye):
   - `app.py`
   - `hyundai_knowledge_base.txt`
   - `requirements.txt`
4. "Commit changes" pe click karo

---

## STEP 2: Groq API key lo (2 min)

1. https://console.groq.com/keys pe jao
2. Free account banao (Google se sign-in ho jata hai)
3. "Create API Key" pe click karo, key copy kar lo (sirf ek baar dikhti hai)

---

## STEP 3: Streamlit Cloud pe deploy karo (3 min)

1. https://share.streamlit.io pe jao
2. GitHub se sign in karo
3. "Create app" → "Yup, I have an app" pe click karo
4. Apna repository select karo (`hyundai-rag-chatbot`)
5. Main file path: `app.py`
6. **IMPORTANT**: "Advanced settings" me jaake "Secrets" section me yeh daalo:
   ```
   GROQ_API_KEY = "yahan_apni_groq_key_paste_karo"
   ```
7. "Deploy" pe click karo

2-3 minute wait karo (pehli baar dependencies install hone me time lagta
hai) — uske baad tera bot live ho jayega ek URL jaisa:
`https://your-app-name.streamlit.app`

---

## STEP 4: Test karo

Us URL ko browser me kholo, sample questions try karo:
- "Creta ki price kya hai?"
- "Venue aur Creta me difference?"
- "EV models kya hain?"

URL ko copy karke kahin bhi share kar sakta hai — koi bhi browser me
khol ke chatbot use kar sakta hai, koi installation nahi chahiye unko.

---

## Agar "website pe embed" karna ho (existing Hyundai dealership website me)

Streamlit app standalone website hai. Agar isko kisi **existing website**
(jaise dealership ka WordPress/HTML site) me chat widget ki tarah embed
karna ho, to iframe use kar sakte ho:

```html
<iframe
  src="https://your-app-name.streamlit.app?embed=true"
  width="400"
  height="600"
  style="position: fixed; bottom: 20px; right: 20px; border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 9999;">
</iframe>
```

Yeh code website ke HTML me (body ke end me) daal do — floating chat
widget jaisa dikhega bottom-right corner me.

---

## Troubleshooting

- **App "sleep" ho jaye kuch der baad**: Streamlit Cloud free tier apps
  inactivity ke baad sleep ho jaate hain — koi bhi request aane pe 10-20
  second me wapas wake ho jaate hain. Interview demo ke liye, demo se
  5 min pehle URL khol ke ek baar activate kar lena.
- **"Module not found" error**: `requirements.txt` sahi se upload hui hai
  ya nahi check karo.
- **Groq key error**: Secrets me key format sahi hai ya nahi check karo
  (quotes ke saath: `GROQ_API_KEY = "gsk_..."`)
