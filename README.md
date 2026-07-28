# InternGrow_FAQChatbot

Context-Aware Intelligent FAQ Chatbot — InternGrow AI Track, Task 2.

## Stages
1. Dataset (data/faqs.csv) — done
2. Preprocessing (NLTK/spaCy) — coming next
3. TF-IDF + Cosine Similarity matching engine
4. Streamlit chat UI (base version)
5. Testing & GitHub push
6. RAG upgrade: sentence-transformer embeddings
7. RAG upgrade: LLM-based answer generation
8. Merge base + RAG into one UI with a mode toggle
9. Demo video + LinkedIn post + submission

## Setup
```bash
pip install -r requirements.txt
```

## Run (once app.py exists)
```bash
streamlit run src/app.py
```

## Quick-Win Features (added on top of the base + RAG chatbot)
- **👍 / 👎 feedback buttons** — appear under every matched answer. Taps are
  logged to `data/feedback_log.csv` (query, mode, matched question, score,
  feedback).
- **Unmatched query logging** — whenever the bot can't confidently answer,
  the query is auto-logged to `data/unmatched_log.csv` with a timestamp and
  the best score it found. This is how you find real gaps in the FAQ set.
- **Color-coded confidence** — every answer shows a 🟢/🟡/🔴 confidence badge
  plus a progress bar, in both Basic and RAG mode.
- **Dataset-expansion helper** — run `python3 src/review_unmatched.py` any
  time to get a ranked list of the most frequent unmatched questions, so you
  know exactly which new rows to add to `data/faqs.csv`.

Both log CSVs are created automatically the first time something is logged
— no setup needed, and they're safe to `.gitignore` if you don't want raw
user queries in your repo history.