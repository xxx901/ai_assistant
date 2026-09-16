"""
本地RAG索引构建：文档 -> 父子分块 -> 向量化(子块) -> 存入LanceDB + 本地父块存储

设计：
- 父块(parent chunk)：较大的文本块，保留完整上下文，是最终要喂给LLM的内容。
- 子块(child chunk)：从父块里再切出更小的块，专门用于向量检索——块越小语义越聚焦，
  检索命中越精确；但小块单独给LLM看又不够完整，所以命中子块后，
  实际返回的是它所属的父块全文，兼顾"检索精度"和"生成时的上下文完整性"。
"""

import os
import json
import uuid
import lancedb
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    LANCEDB_PATH,
    PARENT_STORE_PATH,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

_embedding_model = None


def get_embedding_model():
    """延迟加载嵌入模型——只有第一次真正用到时才加载，避免程序一启动就等很久。"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def split_into_parent_chunks(text: str, chunk_size: int = PARENT_CHUNK_SIZE) -> list:
    """不重叠，从头到尾按字符数切。"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def split_into_child_chunks(text: str, chunk_size: int = CHILD_CHUNK_SIZE, overlap: int = CHILD_CHUNK_OVERLAP) -> list:
    """步长是chunk_size - overlap，让相邻子块之间有重叠。"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def index_document(file_path: str):
    """读取一个文档，完成父子分块 -> 嵌入 -> 存储的完整流程。"""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    parents = split_into_parent_chunks(full_text)
    parent_store = _load_parent_store()

    child_records = []
    for position, parent_text in enumerate(parents):
        parent_id = str(uuid.uuid4())
        parent_store[parent_id] = {
            "text": parent_text,
            "source_file": file_path,
            "position": position,  # 这是第几个父块，供以后"跳转到原文"使用
        }
        for child_text in split_into_child_chunks(parent_text):
            child_records.append({
                "id": str(uuid.uuid4()),
                "parent_id": parent_id,
                "text": child_text,
                "source_file": file_path,
            })

    _save_parent_store(parent_store)
    _embed_and_store_children(child_records)


def _embed_and_store_children(child_records: list):
    if not child_records:
        return

    model = get_embedding_model()
    texts = [r["text"] for r in child_records]
    # normalize_embeddings=True：把每个向量都归一化成单位长度，
    # 这样后面检索时可以直接用cosine距离，数值好解释（下面retriever.py会用到）
    vectors = model.encode(texts, normalize_embeddings=True)

    for record, vector in zip(child_records, vectors):
        record["vector"] = vector.tolist()

    db = lancedb.connect(LANCEDB_PATH)
    if "chunks" in db.table_names():
        table = db.open_table("chunks")
        table.add(child_records)
    else:
        db.create_table("chunks", data=child_records)


def _load_parent_store() -> dict:
    if os.path.exists(PARENT_STORE_PATH):
        with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_parent_store(store: dict):
    os.makedirs(os.path.dirname(PARENT_STORE_PATH), exist_ok=True)
    with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)