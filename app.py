from pathlib import Path
import pickle
import re

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")
st.title("Fake News Detection")
st.write("Enter a news article or headline to get a machine-learning prediction.")
st.caption("Educational project using TF-IDF and Logistic Regression. Predictions are not a substitute for professional fact-checking.")

model_path = MODEL_DIR / "model.pkl"
vectorizer_path = MODEL_DIR / "vectorizer.pkl"

if not model_path.exists() or not vectorizer_path.exists():
    st.warning("The model has not been trained yet. Add Fake.csv and True.csv to the data folder, then run `python train_model.py`.")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

news_text = st.text_area(
    "News text",
    height=220,
    placeholder="Paste the news headline or article here..."
)

if st.button("Analyze News", type="primary", use_container_width=True):
    if not news_text.strip():
        st.info("Please enter some news text first.")
    else:
        cleaned = clean_text(news_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        fake_score = float(probabilities[0]) * 100
        real_score = float(probabilities[1]) * 100

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

        st.subheader("Model scores")
        st.write(f"Fake: {fake_score:.1f}%")
        st.progress(int(round(fake_score)))
        st.write(f"Real: {real_score:.1f}%")
        st.progress(int(round(real_score)))

        st.caption("These percentages are model scores, not verified probabilities that a claim is true or false.")
