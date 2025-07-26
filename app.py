import streamlit as st
import openai
import spacy
import textstat
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForMaskedLM
import torch
import os

# Load models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_mlm.eval()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🧠 LexiLite – Medical Text Simplifier")
st.write("Simplify complex medical language into plain, readable text using GPT and classical NLP techniques.")

text_input = st.text_area("Paste your medical text here:")

@st.cache_data
def detect_complex_words(text):
    doc = nlp(text)
    complex = []
    for token in doc:
        if token.is_alpha and not token.is_stop and len(token.text) > 6 and textstat.syllable_count(token.text) >= 3:
            complex.append(token.text)
    return list(set(complex))

def get_synonyms(word):
    prompt = f"Give 3 simpler alternatives for the word '{word}' that a 10-year-old can understand."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].split(',')

def filter_synonyms(original, candidates):
    original_vec = embedding_model.encode(original, convert_to_tensor=True)
    filtered = []
    for c in candidates:
        c = c.strip()
        sim = util.cos_sim(original_vec, embedding_model.encode(c, convert_to_tensor=True)).item()
        if sim > 0.5:
            filtered.append((c, sim))
    return sorted(filtered, key=lambda x: -x[1])

def rank_by_mlm(sentence, word, candidates):
    results = []
    for c, sim in candidates:
        masked = sentence.replace(word, "[MASK]")
        inputs = tokenizer(masked, return_tensors="pt")
        mask_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
        with torch.no_grad():
            logits = bert_mlm(**inputs).logits
        score = torch.softmax(logits[0, mask_index, :], dim=1)[0][tokenizer.convert_tokens_to_ids(c)].item()
        results.append((c, sim, score))
    return sorted(results, key=lambda x: -x[2])

def simplify_text(text, replacements):
    subst = ", ".join([f"{k} -> {v}" for k, v in replacements.items()])
    prompt = f"Rewrite this medical text using the following substitutions only: {subst}.\n\nText: {text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

if st.button("Simplify Text") and text_input:
    st.subheader("🔍 Step 1: Complex Word Detection")
    complex_words = detect_complex_words(text_input)
    st.write("Detected:", complex_words)

    st.subheader("🔄 Step 2–4: Substitution & Ranking")
    substitutions = {}
    for word in complex_words:
        st.markdown(f"**{word}**")
        try:
            syns = get_synonyms(word)
            filtered = filter_synonyms(word, syns)
            ranked = rank_by_mlm(text_input, word, filtered)
            if ranked:
                best = ranked[0][0]
                st.write(f"→ `{best}` (Score: {ranked[0][2]:.4f})")
                substitutions[word] = best
            else:
                st.write("No good alternatives found.")
        except Exception as e:
            st.warning(f"Error simplifying {word}: {e}")

    st.subheader("✍️ Step 5: Final Simplification")
    simplified = simplify_text(text_input, substitutions)
    st.text_area("Simplified Output", value=simplified, height=200)

    st.subheader("📊 Step 6: Readability Scores")
    orig_fk = textstat.flesch_kincaid_grade(text_input)
    simp_fk = textstat.flesch_kincaid_grade(simplified)
    st.write(f"Original FKGL: {orig_fk:.2f}")
    st.write(f"Simplified FKGL: {simp_fk:.2f}")
