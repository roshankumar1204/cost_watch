"""Step 3b: a small local vector index over the notes (the "R" in RAG).

Groq has no embedding endpoint, and the corpus is 10 short notes, so we use
TF-IDF vectors (word + bigram) with cosine similarity. Free, offline and fully
deterministic. Search supports a metadata filter, like a vector DB would:
first keep only notes that could apply (route + time), then rank by similarity.
Swapping in dense embeddings only means changing `_embed`.
"""
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.notes import NoteFacts


class NoteIndex:
    def __init__(self, facts: list[NoteFacts]):
        self.facts = facts
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        docs = [f"{f.applies_to}. {f.text}" for f in facts]
        self.matrix = self.vectorizer.fit_transform(docs)

    def _embed(self, text: str):
        return self.vectorizer.transform([text])

    def search(self, query: str, keep=lambda f: True, k: int = config.RETRIEVAL_TOP_K):
        """Return [(NoteFacts, score)] for the top-k notes passing `keep`.
        Ties are broken by note_id so the order never changes between runs."""
        scores = (self.matrix @ self._embed(query).T).toarray().ravel()
        ranked = sorted(
            ((f, float(s)) for f, s in zip(self.facts, scores) if keep(f)),
            key=lambda fs: (-round(fs[1], 9), fs[0].note_id),
        )
        return ranked[:k]


def build_query(route: str, week_of: date) -> str:
    """What we 'ask' the index for a flagged week."""
    origin, _, dest = route.partition("-")
    return (f"{route} {origin} {dest} corridor route freight transport cost increase higher costs "
            f"{week_of:%B %Y} fuel diesel toll weather flooding disruption festival surcharge detour")
