import numpy as np
import logging
from config import Settings

logger = logging.getLogger("backend.indexer")
settings = Settings()

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


def load_index(index_path: str = None):
    index_path = index_path or settings.FAISS_INDEX_PATH
    if _HAS_FAISS:
        try:
            idx = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
            return idx
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
            return None
    else:
        logger.info("FAISS not available; falling back to brute-force")
        return None


def save_index(index, path: str):
    if _HAS_FAISS and index is not None:
        faiss.write_index(index, path)
        logger.info(f"Saved FAISS index to {path}")
    else:
        logger.warning("FAISS not available or index is None; cannot save")


def search_index(index, query_vec: np.ndarray, k: int, embeddings: np.ndarray = None, candidate_ids: list = None):
    """
    Search the index or fall back to brute-force cosine over `embeddings`.

    - If FAISS is available and `candidate_ids` is None, use FAISS index.search.
    - Otherwise perform a restricted brute-force search over `candidate_ids` (or all embeddings if None).

    Returns: (scores: List[float], ids: List[int])
    """
    # FAISS fast path (only when no candidate restriction is requested)
    if _HAS_FAISS and index is not None and candidate_ids is None:
        q = query_vec.astype("float32").reshape(1, -1)
        D, I = index.search(q, k)
        return D[0].tolist(), I[0].tolist()

    # Brute-force path (requires embeddings)
    if embeddings is None:
        raise ValueError("Embeddings array required for brute-force search")

    q = query_vec.astype("float32")
    q = q / (np.linalg.norm(q) + 1e-9)
    em = embeddings.astype("float32")
    norms = np.linalg.norm(em, axis=1, keepdims=True) + 1e-9
    em_n = em / norms

    if candidate_ids is None:
        sims = em_n.dot(q)
        idx = np.argsort(-sims)[:k]
        scores = sims[idx].tolist()
        ids = idx.tolist()
        return scores, ids

    # restricted candidate set
    cand_idx = np.asarray(candidate_ids, dtype=int)
    if cand_idx.size == 0:
        return [], []
    cand_embs = em_n[cand_idx]
    sims = cand_embs.dot(q)
    order = np.argsort(-sims)[:k]
    top_idxs = cand_idx[order].tolist()
    top_scores = sims[order].tolist()
    return top_scores, top_idxs
