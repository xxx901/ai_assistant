"""
本地RAG检索：混合检索(粗排) -> RRF融合排序 -> 去重映射到父块 -> reranker精排 -> 按阈值过滤

流程：
1. 粗排：query同时编码成dense向量和sparse(词法)权重。
   - dense：在LanceDB里做ANN搜索，拿到按向量相似度排序的候选。
   - sparse：10万字量级的语料子块数量很小(几百个)，直接在内存里对全部子块
     算一次词法匹配分数(点积)，不需要额外接BM25引擎，简单且够快。
   两路排名各自取前COARSE_TOP_K，用RRF(Reciprocal Rank Fusion)融合成一个排序——
   RRF只看"排第几"不看具体分数数值，天然绕开"dense的cosine相似度"和
   "sparse的词法匹配分数"两者量纲不同、没法直接相加的问题。
2. 精排：融合后的候选子块(短、语义聚焦)直接和query一起送进cross-encoder
   reranker重新打分。这里刻意对"子块"精排而不是对"父块"——父块800字太长，
   关键信息会被作者邮箱、脚注等噪声稀释，cross-encoder打分会被拉低。
3. 把精排命中的子块按parent_id去重、映射回父块全文，作为最终喂给LLM的上下文，
   再按阈值过滤。这样"检索精度"(短子块)和"生成时的上下文完整性"(长父块)两不误。
4. 按RERANK_SCORE_THRESHOLD过滤：分数太低说明知识库里其实没有相关内容，
   诚实地告诉调用方"没找到"，而不是硬塞不相关的内容进去噪音答案。

⚠️ 依赖的LanceDB表里必须有sparse_json字段、parent_store里必须有char_start/char_end字段
   (由indexer.py新版本写入)，如果是旧数据请先重新索引，见indexer.py顶部的说明。
"""

import json
import time
import logging

import numpy as np
import lancedb

from config import (
    LANCEDB_PATH,
    PARENT_STORE_PATH,
    COARSE_TOP_K,
    RRF_K,
    RERANK_TOP_K,
    RERANK_SCORE_THRESHOLD,
)
from rag.indexer import get_embedding_model
from rag.reranker import rerank

logger = logging.getLogger(__name__)


def retrieve(query: str) -> dict:
    """
    返回：
        {"found": True,  "contexts": [{"text":..., "source_file":..., "position":...,
                                        "char_start":..., "char_end":..., "score":...}, ...]}
      或
        {"found": False, "contexts": []}   # 相关度太低，认为库里没有相关资料
    """
    db = lancedb.connect(LANCEDB_PATH)
    if "chunks" not in db.table_names():
        return {"found": False, "contexts": []}
    table = db.open_table("chunks")

    t0 = time.time()
    query_output = get_embedding_model().encode([query], return_dense=True, return_sparse=True)
    query_vector = np.asarray(query_output["dense_vecs"][0], dtype=np.float32)
    query_vector = (query_vector / np.linalg.norm(query_vector)).tolist()
    query_sparse = {str(k): float(v) for k, v in query_output["lexical_weights"][0].items()}
    logger.debug(f"query编码耗时: {time.time() - t0:.3f}秒")

    t0 = time.time()
    # all_rows只加载一次，dense召回和sparse召回都从这份内存数据里算，
    # 避免重复查表——10万字量级下这份数据本来就不大，全量加载到内存完全没问题，
    # 语料规模大到需要担心这一步之前，通常reranker早就是更大的瓶颈了。
    all_rows = table.to_arrow().to_pylist()
    id_to_row = {row["id"]: row for row in all_rows}

    dense_hits = table.search(query_vector).distance_type("cosine").limit(COARSE_TOP_K).to_list()
    dense_ranked_ids = [h["id"] for h in dense_hits]

    sparse_scored = [
        (row["id"], _lexical_matching_score(query_sparse, json.loads(row["sparse_json"])))
        for row in all_rows
    ]
    sparse_scored.sort(key=lambda x: x[1], reverse=True)
    sparse_ranked_ids = [row_id for row_id, _ in sparse_scored[:COARSE_TOP_K]]

    fused_ids = _reciprocal_rank_fusion([dense_ranked_ids, sparse_ranked_ids])[:COARSE_TOP_K]
    logger.debug(f"混合粗排耗时: {time.time() - t0:.3f}秒，融合候选数={len(fused_ids)}")

    child_candidates = []
    for child_id in fused_ids:
        row = id_to_row.get(child_id)
        if row is None:
            continue
        child_candidates.append({"text": row["text"], "parent_id": row["parent_id"]})
    if not child_candidates:
        return {"found": False, "contexts": []}

    t0 = time.time()
    reranked = rerank(query, child_candidates, top_k=RERANK_TOP_K)
    logger.debug(f"精排耗时: {time.time() - t0:.3f}秒")

    parent_store = _load_parent_store()
    contexts = _map_children_to_parents(reranked, parent_store, RERANK_SCORE_THRESHOLD)
    return {"found": len(contexts) > 0, "contexts": contexts}


def _lexical_matching_score(weights_a: dict, weights_b: dict) -> float:
    """稀疏词法匹配分数：两组token权重的点积，只看两边都出现过的token。
    没有自己调用FlagEmbedding内置的compute_lexical_matching_score，
    是为了避免不同库版本对token key类型(str/int)处理不一致导致匹配不上，
    自己实现的点积逻辑简单、可控、跟BGE-M3论文描述的sparse匹配方式一致。"""
    score = 0.0
    for token, weight in weights_a.items():
        if token in weights_b:
            score += weight * weights_b[token]
    return score


def _reciprocal_rank_fusion(ranked_id_lists: list, k: int = RRF_K) -> list:
    """
    RRF: score(id) = sum(1 / (k + rank))，rank从1开始计。
    只看排名不看具体分数值，天然解决"dense的cosine相似度"和
    "sparse的词法匹配分数"量纲不同、没法直接相加的问题。
    """
    fused_scores = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)


def _map_children_to_parents(reranked_children: list, parent_store: dict, threshold: float) -> list[dict]:
    """精排后的子块按parent_id去重，映射回父块全文作为最终上下文。

    子块分数就当作父块的相关性分数用——子块是父块的一部分，子块相关
    说明父块里大概率有答案。多个子块命中同一个父块时只保留分数最高那次。
    """
    seen_parent_ids = set()
    contexts = []
    for child in reranked_children:
        if child["score"] < threshold:
            continue
        parent_id = child["parent_id"]
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        parent_info = parent_store.get(parent_id)
        if parent_info:
            contexts.append({
                "text": parent_info["text"],
                "source_file": parent_info["source_file"],
                "position": parent_info["position"],
                "char_start": parent_info["char_start"],
                "char_end": parent_info["char_end"],
                "score": child["score"],
            })
    return contexts


def _load_parent_store() -> dict:
    with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)