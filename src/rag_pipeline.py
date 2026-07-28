"""
rag_pipeline.py
Combines rag_retriever.py (retrieval) + rag_generator.py (generation)
into a single easy-to-use RAGChatbot class.

This is the complete RAG system - the "upgrade" version of Task 2.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from rag_retriever import RAGRetriever
from rag_generator import RAGGenerator


class RAGChatbot:
    def __init__(self, csv_path: str, top_k: int = 3, similarity_threshold: float = 0.35):
        self.retriever = RAGRetriever(csv_path, top_k=top_k)
        self.generator = RAGGenerator()
        self.similarity_threshold = similarity_threshold

    def answer(self, query: str) -> dict:
        retrieved = self.retriever.retrieve(query, threshold=self.similarity_threshold)
        reply = self.generator.generate(query, retrieved)
        return {
            "answer": reply,
            "matched": len(retrieved) > 0,
            "sources": retrieved,  # useful to show "matched FAQs" in the UI
        }


if __name__ == "__main__":
    bot = RAGChatbot("data/faqs.csv")

    test_queries = [
        "vpn is not connecting",
        "internet is dead at my desk",
        "what is the capital of France",
    ]

    for q in test_queries:
        result = bot.answer(q)
        print(f"\nQuery : {q}")
        print(f"Answer: {result['answer']}")
        print(f"Matched sources: {[s['question'] for s in result['sources']]}")
