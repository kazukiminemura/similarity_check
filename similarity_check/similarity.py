from typing import List, Tuple
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


def rank_similar(target_vec: np.ndarray, candidate_vecs: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for path, vec in candidate_vecs:
        if vec is None or vec.size == 0:
            score = -1.0
        else:
            score = cosine_similarity(target_vec, vec)
        scores.append((path, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

