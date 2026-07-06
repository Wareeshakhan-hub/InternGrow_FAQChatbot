"""
preprocess.py
Stage 2 - Text Preprocessing for the FAQ Chatbot (InternGrow Task 2)

Cleans and normalizes text so that user queries can be reliably matched
against the FAQ dataset using cosine similarity later in Stage 3.

Steps applied to every piece of text:
1. Lowercase
2. Remove punctuation/special characters
3. Tokenize (split into words)
4. Remove stopwords (common words like "the", "is", "a")
5. Lemmatize (reduce words to their base/dictionary form)
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Make sure required NLTK resources are available.
# (Safe to call every run - it skips download if already present.)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase + strip punctuation/digits/extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    Returns a single space-joined string (ready for TF-IDF vectorization).
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    lemmatized = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)


def load_and_preprocess_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the FAQ CSV and adds a new column 'processed_question'
    containing the preprocessed version of each question.
    """
    df = pd.read_csv(csv_path)
    df["processed_question"] = df["question"].apply(preprocess)
    return df


if __name__ == "__main__":
    # Quick manual test when running this file directly
    df = load_and_preprocess_dataset("data/faqs.csv")
    print(df[["question", "processed_question"]].head(10).to_string(index=False))

    print("\n--- Single query test ---")
    sample_query = "How can I reset my password please?"
    print("Original :", sample_query)
    print("Processed:", preprocess(sample_query))
