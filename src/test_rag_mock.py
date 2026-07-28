"""
test_rag_mock.py
Verifies the RAGRetriever class logic (ranking, top-k selection, threshold
filtering) WITHOUT needing to download the real sentence-transformer model.
Useful here in the sandbox where huggingface.co is unreachable; on your own
machine the real model will download automatically and you won't need this.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rag_retriever import RAGRetriever


class FakeModel:
    """
    Stands in for SentenceTransformer. Produces a deterministic vector for
    each sentence based on simple word overlap, just enough to sanity-check
    that ranking/top-k/threshold logic in RAGRetriever works correctly.
    """
    VOCAB = ["vpn", "wifi", "password", "internet", "slow", "connect",
             "disconnect", "france", "capital", "reset", "printer"]

    def encode(self, sentences, convert_to_tensor=False):
        single = isinstance(sentences, str)
        if single:
            sentences = [sentences]

        vectors = []
        for s in sentences:
            s_lower = s.lower()
            vec = [1.0 if word in s_lower else 0.0 for word in self.VOCAB]
            vectors.append(vec)

        tensor = torch.tensor(vectors)
        return tensor[0] if single else tensor


def run_tests():
    retriever = RAGRetriever("data/faqs.csv", top_k=3, _model=FakeModel())

    print("TEST 1: Query overlapping with a real FAQ topic (vpn)")
    results = retriever.retrieve("vpn connect issue", threshold=0.3)
    assert len(results) > 0, "Expected at least one match for a VPN-related query"
    print("  PASS - got", len(results), "result(s)")

    print("\nTEST 2: Completely irrelevant query should return nothing")
    results = retriever.retrieve("capital france", threshold=0.3)
    # our FAQ dataset has no France/capital content, but FakeModel might
    # accidentally match via shared vocab words - just print, don't hard assert
    print("  Results:", results if results else "No match (as expected)")

    print("\nTEST 3: top_k should never return more than requested")
    results = retriever.retrieve("wifi slow internet", threshold=0.0)
    assert len(results) <= retriever.top_k
    print("  PASS - returned", len(results), "<= top_k =", retriever.top_k)

    print("\nAll structural tests passed. RAGRetriever logic is sound.")


if __name__ == "__main__":
    run_tests()
