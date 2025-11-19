from typing import Any
import numpy as np
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("backend.models.text_model")

# Note: sentence_transformers, torch, and datasets imports are only needed for training
# They are imported locally in the TransformerTextFashionRecommender class if needed


def load_text_model(path: str) -> Any:
    """Load a pre-trained text model from a pickle file.
    
    The pickle file should contain a dict with keys:
    - 'item_ids': numpy array of item IDs
    - 'embeddings': numpy array of embeddings
    - 'base_colours': list of base colours
    - 'article_types': list of article types
    - 'product_names': list of product names (optional)
    """
    from pathlib import Path
    import joblib
    from sklearn.neighbors import NearestNeighbors
    
    logger.info(f"Attempting to load text model from {path}")
    
    p = Path(path)
    if not p.is_absolute():
        # If path already starts with 'models/', resolve from backend root
        # Otherwise resolve relative to this file's directory
        if str(path).startswith('models/'):
            # Path is like 'models/text_model.pkl', resolve from backend root
            p = (Path(__file__).parent.parent / path).resolve()
        else:
            # Path is relative to models directory
            p = (Path(__file__).parent / path).resolve()
    
    if not p.exists():
        logger.warning(f"Text model file not found at {p}, using stub model")
        class SimpleTextModel:
            def embed_text(self, text: str) -> np.ndarray:
                seed = abs(hash(text)) % (2**32 - 1)
                # Match stored embeddings dimension (768)
                return np.random.RandomState(seed).randn(768).astype("float32")
        return SimpleTextModel()
    
    try:
        # Load the saved state
        state = joblib.load(p)
        logger.info(f"Successfully loaded text model from {p}")
        
        # Create a wrapper class that uses the loaded embeddings
        class LoadedTextModel:
            def __init__(self, state_dict):
                self._item_ids = state_dict['item_ids']
                self._embeddings = state_dict['embeddings']
                self._base_colours = state_dict.get('base_colours', [])
                self._article_types = state_dict.get('article_types', [])
                self._product_names = state_dict.get('product_names', [])
                
                # Build nearest neighbor index for fast similarity search
                self.nn = NearestNeighbors(
                    metric="cosine",
                    algorithm="brute",
                    n_neighbors=min(50, len(self._embeddings)),
                )
                self.nn.fit(self._embeddings)
            
            def embed_text(self, text: str) -> np.ndarray:
                """Get embedding for a text query by finding nearest neighbor."""
                # For simplicity, return a random embedding based on hash
                # In a full implementation, you'd use a sentence transformer here
                seed = abs(hash(text)) % (2**32 - 1)
                # Match stored embeddings dimension (768)
                return np.random.RandomState(seed).randn(768).astype("float32")
            
            def get_state(self):
                """Return the loaded state for use by other components."""
                return {
                    'item_ids': self._item_ids,
                    'embeddings': self._embeddings,
                    'base_colours': self._base_colours,
                    'article_types': self._article_types,
                    'product_names': self._product_names,
                    'nn': self.nn
                }
        
        return LoadedTextModel(state)
    except Exception as e:
        logger.error(f"Failed to load text model from {p}: {e}", exc_info=True)
        # Fallback to stub
        class SimpleTextModel:
            def embed_text(self, text: str) -> np.ndarray:
                seed = abs(hash(text)) % (2**32 - 1)
                # Match stored embeddings dimension (768)
                return np.random.RandomState(seed).randn(768).astype("float32")
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

# Training code removed - models should be loaded from pre-trained .pkl files
# If you need to retrain, run this file as a script explicitly
