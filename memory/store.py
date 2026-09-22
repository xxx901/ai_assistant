"""
长期记忆存储：LanceDB 表 + 语义检索 + 相似去重。

设计：
- 独立DB路径(./data/memory_lancedb)，和RAG的 ./data/lancedb 隔离，
  避免表名/数据互相污染。
- 每条记忆一行，字段：id/text/vector/type/importance/created_at/
  last_accessed/source。
- 写入前先向量搜top-1做相似去重：余弦相似度 >= MEMORY_DEDUP_THRESHOLD 就
  更新旧记忆(保留id、刷新last_accessed、importance取较大值)，否则新增。
- LanceDB 的 distance_type("cosine") 返回的是余弦距离(=1-余弦相似度)，
  所以判断"相似"要用 1 - _distance 和阈值比。
"""

import time
import uuid
import logging

import numpy as np
import lancedb

from config import (
    MEMORY_DB_PATH,
    MEMORY_TABLE_NAME,
    MEMORY_DEDUP_THRESHOLD,
)
from rag.indexer import get_embedding_model

logger = logging.getLogger(__name__)

VALID_TYPES = {"preference", "fact", "entity", "event"}


class MemoryStore:
    def __init__(self):
        self.db = lancedb.connect(MEMORY_DB_PATH)
        # 表在首次写入时才建(用第一条数据推断schema)，读之前不存在就返回空。
        self.table = None
        if MEMORY_TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(MEMORY_TABLE_NAME)

    def _embed(self, text: str) -> list:
        """bge-m3 dense向量 + L2归一化。复用RAG的嵌入模型(同一个实例)。"""
        model = get_embedding_model()
        out = model.encode([text], return_dense=True)
        vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def upsert(self, text: str, type_: str, importance: int, source: str):
        """写入一条记忆：相似则更新旧记忆，否则新增。返回 (action, id)。

        action 为 "updated" 或 "created"，方便上层记日志/统计去重效果。
        """
        if type_ not in VALID_TYPES:
            type_ = "fact"
        importance = max(1, min(5, int(importance)))
        vec = self._embed(text)
        now = time.time()

        if self.table is not None:
            hits = self.table.search(vec).distance_type("cosine").limit(1).to_list()
            if hits and (1.0 - hits[0]["_distance"]) >= MEMORY_DEDUP_THRESHOLD:
                hit = hits[0]
                hit_id = hit["id"]
                # 只更新标量字段，vector保留旧值：既然判定为"相似"，
                # 旧向量已经是这条记忆簇的好代表，无需每次重算。
                self.table.update(
                    where=f"id = '{hit_id}'",
                    values={
                        "text": text,
                        "type": type_,
                        "importance": max(hit["importance"], importance),
                        "last_accessed": now,
                    },
                )
                logger.debug(f"记忆去重命中，更新: {hit_id} -> {text!r}")
                return "updated", hit_id

        mem_id = str(uuid.uuid4())
        row = {
            "id": mem_id,
            "text": text,
            "vector": vec,
            "type": type_,
            "importance": importance,
            "created_at": now,
            "last_accessed": now,
            "source": source,
        }
        if self.table is None:
            self.table = self.db.create_table(MEMORY_TABLE_NAME, data=[row])
        else:
            self.table.add([row])
        logger.debug(f"记忆新增: {mem_id} -> {text!r}")
        return "created", mem_id

    def semantic_search(self, query_vec: list, k: int) -> list:
        """语义召回：按余弦相似度取top-k，每条带 similarity 字段。"""
        if self.table is None:
            return []
        hits = self.table.search(query_vec).distance_type("cosine").limit(k).to_list()
        return [self._row_to_item(h, similarity=1.0 - h["_distance"]) for h in hits]

    def entity_search(self, query: str) -> list:
        """实体召回：type=entity 的记忆名和query互相包含(忽略大小写)即召回。"""
        if self.table is None:
            return []
        q = query.lower()
        results = []
        for row in self._all_rows():
            if row["type"] != "entity":
                continue
            text = row["text"].lower()
            if q in text or text in q:
                # 实体名精确命中，视为高相似信号
                results.append(self._row_to_item(row, similarity=1.0))
        return results

    def recent(self, k: int) -> list:
        """最近路召回：按 last_accessed 取最近 k 条。"""
        if self.table is None:
            return []
        rows = sorted(self._all_rows(), key=lambda r: r["last_accessed"], reverse=True)[:k]
        return [self._row_to_item(r, similarity=0.0) for r in rows]

    def touch(self, mem_id: str):
        """命中后刷新 last_accessed，供时效排序和"最近"召回用。"""
        if self.table is None:
            return
        self.table.update(where=f"id = '{mem_id}'", values={"last_accessed": time.time()})

    def count(self) -> int:
        return len(self._all_rows())

    def _all_rows(self) -> list:
        if self.table is None:
            return []
        return self.table.to_arrow().to_pylist()

    @staticmethod
    def _row_to_item(row: dict, similarity: float) -> dict:
        return {
            "id": row["id"],
            "text": row["text"],
            "type": row["type"],
            "importance": row["importance"],
            "last_accessed": row["last_accessed"],
            "similarity": similarity,
        }
