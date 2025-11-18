from typing import Any
import numpy as np
import logging
from dataclasses import dataclass
from typing import List
import numpy as np

from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import torch
import joblib
from datasets import load_dataset

logger = logging.getLogger("backend.models.text_model")


def load_text_model(path: str) -> Any:
    logger.info(f"Attempting to load text model from {path}")

    class SimpleTextModel:
        def embed_text(self, text: str) -> np.ndarray:
            seed = abs(hash(text)) % (2**32 - 1)
            return np.random.RandomState(seed).randn(512).astype("float32")

    return SimpleTextModel()

@dataclass
class Recommendation:
    item_id: int
    score: float
    row_index: int


class TransformerTextFashionRecommender:
    """
    Text-based recommender that:
      - Uses transformer embeddings for semantic similarity
      - ALSO uses baseColour + articleType to make queries like 'blue shirt'
        match actual blue shirts first.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        n_neighbors_default: int = 50,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.n_neighbors_default = n_neighbors_default

        self.used_columns = [
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
            "productDisplayName",
        ]

        self._texts: List[str] = []
        self._item_ids: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._base_colours: List[str] = []
        self._article_types: List[str] = []
        self.nn: NearestNeighbors | None = None

    # -------- text building ----------

    def _textify(self, row) -> str:
        parts = []
        for col in self.used_columns:
            if col in row and row[col] is not None:
                parts.append(str(row[col]).lower())
        return " ".join(parts)

    # -------- training / fitting ----------

    def fit(self, ds):
        """
        Build item texts, encode with transformer, store attributes and build NN index.
        """
        print("Preparing text & attributes...")
        self._texts = [self._textify(row) for row in ds]
        self._item_ids = np.array([int(row["id"]) for row in ds])

        # store baseColour and articleType for each item for rule-based filtering
        self._base_colours = [
            (row["baseColour"].lower() if row.get("baseColour") is not None else "")
            for row in ds
        ]
        self._article_types = [
            (row["articleType"].lower() if row.get("articleType") is not None else "")
            for row in ds
        ]

        print("Encoding with transformer...")
        self._embeddings = self.encoder.encode(
            self._texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print("Building nearest-neighbor index (for item-id queries)...")
        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(self.n_neighbors_default, len(self._embeddings)),
        )
        self.nn.fit(self._embeddings)

        print("Model ready 🎉")

    # -------- main recommend method ----------

    def recommend(self, query: str, top_k: int = 5) -> List[Recommendation]:
        """
        If `query` is a numeric id present in the dataset → recommend by item id.
        Else → treat as fashion text ('blue shirt') and:
                1) filter catalog by baseColour/articleType
                2) rank by semantic similarity.
        """
        if self._embeddings is None or self._item_ids is None:
            raise RuntimeError("Model not fitted yet. Call fit(ds) first.")

        query = query.strip()
        # -------------------------------------------------
        # Case 1: query is a valid product ID
        # -------------------------------------------------
        if query.isdigit() and int(query) in self._item_ids:
            item_id = int(query)
            query_idx = int(np.where(self._item_ids == item_id)[0][0])
            vec = self._embeddings[query_idx]

            distances, indices = self.nn.kneighbors(
                [vec],
                n_neighbors=min(top_k + 1, len(self._embeddings)),
            )

            recs: List[Recommendation] = []
            for d, i in zip(distances[0], indices[0]):
                i_int = int(i)
                if i_int == query_idx:  # skip the same product
                    continue
                sim = 1.0 - float(d)
                recs.append(
                    Recommendation(
                        item_id=int(self._item_ids[i_int]),
                        score=sim,
                        row_index=i_int,
                    )
                )
                if len(recs) >= top_k:
                    break
            return recs

        # -------------------------------------------------
        # Case 2: text query like "blue shirt"
        # -------------------------------------------------
        q_lower = query.lower()
        words = set(q_lower.split())

        # 1) pick candidate items whose baseColour or articleType
        #    contains a word from the query
        candidate_indices: List[int] = []
        for i, (colour, art) in enumerate(zip(self._base_colours, self._article_types)):
            for w in words:
                if w and (w in colour or w in art):
                    candidate_indices.append(i)
                    break  # already matched; no need to check other words

        # if nothing matched, fall back to using the whole catalog
        if not candidate_indices:
            candidate_indices = list(range(len(self._embeddings)))

        # 2) semantic similarity within candidate set
        vec = self.encoder.encode([q_lower], normalize_embeddings=True, convert_to_numpy=True)[0]
        cand_embs = self._embeddings[candidate_indices]  # shape [C, d], already normalized
        sims = cand_embs @ vec  # cosine similarity because both normalized

        # top-k indices in candidate_indices based on sims
        top_idx = np.argsort(-sims)[:top_k]

        recs: List[Recommendation] = []
        for j in top_idx:
            global_i = int(candidate_indices[int(j)])
            sim = float(sims[int(j)])
            recs.append(
                Recommendation(
                    item_id=int(self._item_ids[global_i]),
                    score=sim,
                    row_index=global_i,
                )
            )

        return recs

print("Loading dataset (this may take a moment)...")
ds = load_dataset("ashraq/fashion-product-images-small", split="train")

model = TransformerTextFashionRecommender()
model.fit(ds)

user_input = input("\nType product ID or search text (e.g. 'blue shirt'):\n→ ")

recs = model.recommend(user_input, top_k=5)

print("\nRecommended items:")
for r in recs:
    row = ds[int(r.row_index)]
    print(f"⭐ {r.score:.3f} | {row['productDisplayName']} (ID: {r.item_id})")

print("Loading dataset (this may take a moment)...")
ds = load_dataset("ashraq/fashion-product-images-small", split="train")

model = TransformerTextFashionRecommender()
model.fit(ds)

# Optional: product names, useful for quick testing later
product_names = [row["productDisplayName"] for row in ds]

state = {
    "item_ids": model._item_ids,
    "embeddings": model._embeddings,
    "base_colours": model._base_colours,
    "article_types": model._article_types,
    "product_names": product_names,  # optional but nice to have
}

joblib.dump(state, "fashion_recommender_state.pkl")

print("Saved as fashion_recommender_state.pkl ✅")
files.download("fashion_recommender_state.pkl")
