"""
The retriever — turns text into vectors and finds the closest matches.

This is what "RAG" actually is under the hood, in three steps:

  1. EMBED   every corpus entry into a vector (a list of numbers that captures
             its meaning). Similar meanings -> nearby vectors.
  2. INDEX   all those vectors in FAISS, a library built for fast
             nearest-neighbour search over millions of vectors.
  3. SEARCH  embed the user's question the same way, and ask FAISS for the k
             entries whose vectors are closest -> those are the most relevant.

We normalize vectors so that "closeness" = cosine similarity (angle between
vectors), the standard similarity measure for text embeddings.

The model + index are built ONCE the first time this module is used, then
reused for every query (building them per-question would be wasteful).

Run:  python -m app.rag.retriever      # demo: prints what gets retrieved
"""

import faiss
from sentence_transformers import SentenceTransformer

from app.rag.corpus import build_documents

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, 384-dim; great for short text


class Retriever:
    def __init__(self):
        self.docs = build_documents()
        self.model = SentenceTransformer(MODEL_NAME)
        # embed every doc's `text`; normalize so inner product == cosine similarity
        vecs = self.model.encode(
            [d["text"] for d in self.docs],
            normalize_embeddings=True,
        )
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # IP = inner product (cosine, since normed)
        self.index.add(vecs)

    def retrieve(self, query: str, k: int = 4):
        """Return the k documents most similar in meaning to `query`."""
        q = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:  # FAISS returns -1 to pad if fewer than k exist
                continue
            doc = dict(self.docs[i])
            doc["score"] = float(score)
            results.append(doc)
        return results


# module-level singleton: built once, reused everywhere
_retriever = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve_context(query: str, k: int = 4) -> str:
    """Retrieve the top-k entries and format them as a prompt block."""
    hits = get_retriever().retrieve(query, k)
    return "\n".join(h["content"] for h in hits)


if __name__ == "__main__":
    r = get_retriever()
    for q in [
        "which items might run out of stock?",
        "revenue split across regions",
        "who are our worst suppliers",
    ]:
        print(f"\nQ: {q}")
        for h in r.retrieve(q, k=3):
            print(f"  [{h['score']:.2f}] ({h['kind']}) {h['text'][:70]}")
