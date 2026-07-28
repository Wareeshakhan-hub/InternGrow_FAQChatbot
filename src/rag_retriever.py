"""
rag_retriever.py
Stage 6 - RAG Upgrade: Embedding-based Semantic Retrieval

This replaces TF-IDF + Cosine Similarity (keyword matching) with
sentence-transformer embeddings (meaning-based matching).

Why this is better:
- TF-IDF only looks at shared words. "vpn is not connecting" and
  "vpn keeps disconnecting" share the word "vpn" but TF-IDF can't tell
  these mean almost the OPPOSITE thing in places, or fail to link a query
  that shares NO words with the right FAQ (e.g. "internet is dead at my
  desk" vs "WiFi is very slow").
- Sentence embeddings encode meaning, so paraphrases and related concepts
  are matched correctly even with completely different words.

Run this file directly to see a demo:
    python3 src/rag_retriever.py
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util


class RAGRetriever:
    def __init__(self, csv_path: str, model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 3, _model=None):
        """
        _model: optional pre-built model object (used for offline/mock testing
        without needing to download real weights).
        """
        self.df = pd.read_csv(csv_path)
        self.top_k = top_k
        self.model = _model if _model is not None else SentenceTransformer(model_name)

        # Encode all FAQ questions ONCE, up front (fast lookups afterwards)
        self.question_embeddings = self.model.encode(
            self.df["question"].tolist(), convert_to_tensor=True
        )

    def retrieve(self, query: str, threshold: float = 0.35):
        """
        Returns the top_k most semantically similar FAQ entries to the query,
        each with its similarity score. Filters out anything below threshold.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.question_embeddings)[0]

        k = min(self.top_k, len(self.df))
        top_results = scores.topk(k=k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            score = float(score)
            if score < threshold:
                continue
            row = self.df.iloc[int(idx)]
            results.append({
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "score": round(score, 3),
            })
        return results


if __name__ == "__main__":
    retriever = RAGRetriever("data/faqs.csv", top_k=3)

    test_queries = [
        "vpn is not connecting",
        "internet is dead at my desk",   # shares almost no words with any FAQ
        "how to fix slow wifi",
        "what is the capital of France",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q)
        if not results:
            print("  No confident match found.")
        for r in results:
            print(f"  [{r['score']}] {r['question']}  ->  {r['answer'][:60]}...")
