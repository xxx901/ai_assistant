"""
长期记忆检索：多路召回(语义/实体/最近) + 综合排序。

设计：
- 三路召回各自独立，合并到一个 by_id 字典去重(同一个记忆可能被多路命中)。
- 语义路先填，实体/最近路用 setdefault，避免覆盖语义路带回来的真实相似度。
- 排序综合三要素：相似度 × 重要性 × 时效，权重在config里可调。
"""

import time
import logging

import numpy as np

from config import MEMORY_RETRIEVAL_TOP_K, MEMORY_RECENT_K, MEMORY_RANKING_WEIGHTS
from rag.indexer import get_embedding_model

logger = logging.getLogger(__name__)


def _embed_query(query: str) -> list:
    model = get_embedding_model()
    out = model.encode([query], return_dense=True)
    vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def retrieve(store, query: str) -> list:
    """三路召回后合并去重、综合排序，返回按score降序的记忆列表。"""
    qv = _embed_query(query)
    by_id = {}

    # 语义路：cosine top-k，带真实相似度
    for item in store.semantic_search(qv, MEMORY_RETRIEVAL_TOP_K):
        by_id[item["id"]] = item

    # 实体路：实体名精确命中
    for item in store.entity_search(query):
        by_id.setdefault(item["id"], item)

    # 最近路：最近访问过的
    for item in store.recent(MEMORY_RECENT_K):
        by_id.setdefault(item["id"], item)

    now = time.time()
    w = MEMORY_RANKING_WEIGHTS
    ranked = []
    for item in by_id.values():
        age_days = (now - item["last_accessed"]) / 86400.0
        recency = 1.0 / (1.0 + age_days)
        importance_norm = (item["importance"] - 1) / 4.0  # 1~5 -> 0~1
        score = (
            w["similarity"] * item["similarity"]
            + w["importance"] * importance_norm
            + w["recency"] * recency
        )
        ranked.append({**item, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
