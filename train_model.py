from pathlib import Path
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Put Fake.csv and True.csv inside the data folder."
        )

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # The common Kaggle dataset contains a 'text' column.
    # If it is absent, combine available title/text columns.
    def get_news_text(frame):
        columns = [c for c in ["title", "text"] if c in frame.columns]
        if not columns:
            raise ValueError("CSV must contain a 'text' column or a 'title'/'text' column.")
        return frame[columns].fillna("").astype(str).agg(" ".join, axis=1)

    fake_data = pd.DataFrame({"text": get_news_text(fake), "label": 0})
    true_data = pd.DataFrame({"text": get_news_text(true), "label": 1})

    data = pd.concat([fake_data, true_data], ignore_index=True)
    data["text"] = data["text"].apply(clean_text)
    data = data[data["text"].str.len() > 0]
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    print("Loading dataset...")
    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    print("Converting text into TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
    with open(MODEL_DIR / "model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("\nModel and vectorizer saved in the model folder.")


if __name__ == "__main__":
    main()
