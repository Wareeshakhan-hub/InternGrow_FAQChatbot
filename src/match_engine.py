"""
match_engine.py
Stage 3 - TF-IDF + Cosine Similarity Matching Engine (InternGrow Task 2, base version)

This is the core "brain" of the base chatbot:
1. Preprocess and vectorize all FAQ questions using TF-IDF.
2. When a user asks something, preprocess + vectorize their query the same way.
3. Compute cosine similarity between the query vector and every FAQ vector.
4. Return the best-matching answer if the similarity score clears a threshold,
   otherwise return a fallback "I don't understand" message.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess, load_and_preprocess_dataset


class FAQMatcher:
    def __init__(self, csv_path: str, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.df = load_and_preprocess_dataset(csv_path)

        # Fit TF-IDF on all preprocessed FAQ questions.
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(self.df["processed_question"])

    def get_best_match(self, user_query: str):
        """
        Returns a dict with: answer, matched_question, category, score, matched (bool)
        """
        processed_query = preprocess(user_query)

        # If preprocessing removes everything (e.g. query was just stopwords/symbols)
        if processed_query.strip() == "":
            return self._no_match_response()

        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.faq_vectors).flatten()

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.similarity_threshold:
            return self._no_match_response(best_score)

        row = self.df.iloc[best_idx]
        return {
            "matched": True,
            "answer": row["answer"],
            "matched_question": row["question"],
            "category": row["category"],
            "score": round(float(best_score), 3),
        }

    def _no_match_response(self, score: float = 0.0):
        return {
            "matched": False,
            "answer": "I'm sorry, I don't have an answer for that yet. Could you rephrase your question, or contact the IT helpdesk directly at support@company.com?",
            "matched_question": None,
            "category": None,
            "score": round(float(score), 3),
        }


if __name__ == "__main__":
    matcher = FAQMatcher("data/faqs.csv", similarity_threshold=0.3)

    test_queries = [
        "How do I reset my password?",          # exact match
        "I can't remember my password, help",   # paraphrased match
        "vpn is not connecting",                # paraphrased match
        "how to fix slow wifi",                 # paraphrased match
        "what is the capital of France",        # should NOT match (irrelevant)
        "printer is not printing anything",     # paraphrased match
    ]

    for q in test_queries:
        result = matcher.get_best_match(q)
        print(f"\nUser Query   : {q}")
        print(f"Matched?     : {result['matched']}  (score={result['score']})")
        if result["matched"]:
            print(f"Matched FAQ  : {result['matched_question']}")
            print(f"Category     : {result['category']}")
        print(f"Answer       : {result['answer']}")
