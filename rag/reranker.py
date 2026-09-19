"""
Cross-encoder重排序(精排)。

粗排阶段(retriever.py的混合检索)召回的候选数量不多(COARSE_TOP_K个)，
可以承受"每个候选单独和query组成一对、跑一次前向推理"的开销，
换来比"两个向量算cosine相似度"精确得多的相关性排序——
这正是"粗排负责快速圈候选、精排负责精确排序"两阶段分工的意义所在。

模型先用体积较小的BAAI/bge-reranker-base跑通整条链路，如果50题评测集
测出来精度不够，再考虑换成参数量更大的bge-reranker-v2-m3或者Qwen3-Reranker系列——
只需要改config.py里的RERANKER_MODEL_NAME，不用动这个模块或调用方(retriever.py)的代码。
"""

import logging
from FlagEmbedding import FlagReranker

from config import RERANKER_MODEL_NAME, RERANKER_DEVICE

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker() -> FlagReranker:
    """延迟加载——只有第一次真正用到reranker时才加载模型，避免程序一启动就等很久。"""
    global _reranker
    if _reranker is None:
        kwargs = {"use_fp16": True}
        if RERANKER_DEVICE:
            kwargs["device"] = RERANKER_DEVICE
        _reranker = FlagReranker(RERANKER_MODEL_NAME, **kwargs)
        logger.info(f"reranker加载完成: {RERANKER_MODEL_NAME}")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    candidates: retriever.py传来的父块候选列表，每个元素至少要有text字段。
    返回：按reranker分数从高到低排序、只保留前top_k个，
          每个元素多一个score字段(0~1，来自normalize=True的sigmoid归一化)，
          调用方(retriever.py)会拿这个score去跟RERANK_SCORE_THRESHOLD比较。
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = get_reranker().compute_score(pairs, normalize=True)
    # 只有一个候选时compute_score可能直接返回单个float而不是list，统一成list处理
    if isinstance(scores, float):
        scores = [scores]

    for candidate, score in zip(candidates, scores):
        candidate["score"] = score

    return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]