"""
本地RAG检索：问题转向量 -> 在子块向量库里找最相似的几个 ->
映射回它们所属的父块(去重) -> 按相关度阈值过滤，决定要不要真的采用检索结果。

分数阈值的存在，正是为了呼应模块2留下的伏笔：
"路由层没法准确判断chat和rag的模糊边界，应该把置信度判断交给下游更有信息量的层"——
这里的分数就是那个更可靠的、客观算出来的判断依据。
"""

import json
from config import LANCEDB_PATH, PARENT_STORE_PATH, RAG_SCORE_THRESHOLD
from rag.indexer import get_embedding_model

import lancedb


def retrieve(query: str, top_k: int = 5) -> dict:
    """
    返回：
        {"found": True,  "contexts": [{"text":..., "source_file":..., "position":..., "score":...}, ...]}
      或
        {"found": False, "contexts": []}   # 相关度太低，认为库里没有相关资料
    """
    model = get_embedding_model()
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

    db = lancedb.connect(LANCEDB_PATH)
    if "chunks" not in db.table_names():
        return {"found": False, "contexts": []}

    table = db.open_table("chunks")
    # 显式指定cosine距离（LanceDB默认是l2距离），这样"1 - 距离 = 相似度"这个换算才成立
    results = table.search(query_vector).distance_type("cosine").limit(top_k).to_list()

    parent_store = _load_parent_store()
    seen_parent_ids = set()
    contexts = []

    for r in results:
        score = 1 - r["_distance"]  # cosine距离转换成"越大越相关"的相似度
        if score < RAG_SCORE_THRESHOLD:
            continue

        parent_id = r["parent_id"]
        if parent_id in seen_parent_ids:
            # 同一个父块下的多个子块都命中了，只需要返回一次父块全文
            continue
        seen_parent_ids.add(parent_id)

        parent_info = parent_store.get(parent_id)
        if parent_info:
            contexts.append({
                "text": parent_info["text"],
                "source_file": parent_info["source_file"],
                "position": parent_info["position"],
                "score": round(score, 4),
            })

    return {"found": len(contexts) > 0, "contexts": contexts}


def _load_parent_store() -> dict:
    with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)