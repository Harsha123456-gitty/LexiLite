# 🧠 LexiLite – Controlled Medical Text Simplifier

**LexiLite** is a lightweight web app that simplifies medical text using ChatGPT, classic lexical simplification (LS) techniques, and real-time readability evaluation.

## 🚀 Features
- Detects complex words using `spaCy` + `textstat`
- Generates simpler synonyms via ChatGPT (GPT-3.5)
- Filters and ranks replacements using BERT embeddings + MLM probability
- Performs controlled rewriting using only verified substitutions
- Displays before/after readability scores

## 🧰 Tech Stack
- Python, Streamlit
- OpenAI (GPT-3.5 via API)
- HuggingFace Transformers, SBERT
- spaCy, textstat, torch

## 🧪 How to Use (Streamlit Cloud Recommended)

1. **Fork or clone this repo**
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and deploy a new app
3. Set `app.py` as the entry file
4. Add your OpenAI API key in the **Secrets** panel like this:

```toml
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
