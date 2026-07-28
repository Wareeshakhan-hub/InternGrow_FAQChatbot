"""
rag_generator.py
Stage 7 - RAG Upgrade: Answer Generation (the "G" in RAG)

Uses a small, completely FREE, local model (google/flan-t5-small) - no API
key, no cost, runs on CPU. Downloads once (~300MB) then works offline.

Given a user's question + the FAQ context retrieved by rag_retriever.py,
this rephrases the raw stored answer into a more natural, conversational
reply instead of just echoing the FAQ text verbatim.
"""

class RAGGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small", _generate_fn=None):
        """
        _generate_fn: optional callable(prompt) -> str override, used for
        offline/mock testing without downloading real weights. Normally left
        as None so the real local model is loaded.

        NOTE: we use AutoTokenizer + AutoModelForSeq2SeqLM directly instead of
        the pipeline("text2text-generation", ...) shortcut, because newer
        versions of the transformers library have removed/renamed that task
        string, causing `KeyError: Unknown task text2text-generation`. Calling
        the model classes directly is stable across versions.
        """
        self._generate_fn = _generate_fn
        if self._generate_fn is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _run_model(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate(self, query: str, retrieved_faqs: list, max_new_tokens: int = 100) -> str:
        """
        retrieved_faqs: list of dicts from RAGRetriever.retrieve(), each with
        'question' and 'answer' keys. Uses them as context.
        """
        if not retrieved_faqs:
            return ("I'm sorry, I don't have information on that yet. "
                    "Could you rephrase your question or contact the IT helpdesk directly?")

        context = "\n".join(
            f"- Q: {f['question']} A: {f['answer']}" for f in retrieved_faqs
        )

        prompt = (
            "You are a helpful IT helpdesk assistant. Using ONLY the context "
            "below, answer the user's question in a friendly, natural, "
            "conversational way. If the context doesn't contain the answer, "
            "say you don't have that information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        if self._generate_fn is not None:
            return self._generate_fn(prompt).strip()
        return self._run_model(prompt, max_new_tokens).strip()


if __name__ == "__main__":
    # Quick manual demo (requires internet to download the model the first time)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from rag_retriever import RAGRetriever

    retriever = RAGRetriever("data/faqs.csv", top_k=2)
    generator = RAGGenerator()

    test_queries = [
        "vpn is not connecting",
        "internet is dead at my desk",
    ]

    for q in test_queries:
        retrieved = retriever.retrieve(q)
        answer = generator.generate(q, retrieved)
        print(f"\nQuery : {q}")
        print(f"Answer: {answer}")