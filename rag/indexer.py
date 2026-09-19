"""
本地RAG索引构建：文档 -> 父子分块 -> 向量化(子块，dense+sparse) -> 存入LanceDB + 本地父块存储

设计：
- 父块(parent chunk)：较大的文本块，保留完整上下文，是最终要喂给LLM的内容。
  额外记录它在原文档里的char_start/char_end——这是"引用溯源"的关键字段，
  以后不管是CLI里高亮打印原文，还是网页版做"点击跳转到原文段落"，
  都靠这个精确定位，而不是只有"第几块"这种不透明的相对序号。
- 子块(child chunk)：从父块里再切出更小的块，专门用于检索——块越小语义越聚焦，
  检索命中越精确；但小块单独给LLM看又不够完整，所以命中子块后，
  实际返回的是它所属的父块全文，兼顾"检索精度"和"生成时的上下文完整性"。
- 每个子块同时存dense向量和sparse(词法)权重，为retriever.py的混合检索做准备——
  BGE-M3支持"一次前向传播同时输出两种表示"，不需要额外接一套BM25引擎。

⚠️ 这份文件相对旧版本改了存储schema(新增sparse_json/char_start/char_end字段)，
   如果你之前已经跑过index_document()产出过旧的data/lancedb和data/parents.json，
   升级到这份代码之前请先删掉这两处旧数据，重新全量索引一遍，否则读取旧记录时
   会因为缺字段直接报KeyError。
"""

import os
import json
import uuid
import numpy as np
import lancedb
from FlagEmbedding import BGEM3FlagModel

from config import (
    EMBEDDING_MODEL_NAME,
    LANCEDB_PATH,
    PARENT_STORE_PATH,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

_embedding_model = None


def get_embedding_model() -> BGEM3FlagModel:
    """延迟加载嵌入模型——只有第一次真正用到时才加载，避免程序一启动就等很久。

    用BGEM3FlagModel而不是sentence-transformers的SentenceTransformer，
    是因为只有这个接口能一次性同时拿到dense向量和sparse(词法)权重，
    后者是retriever.py做混合检索要用的，省得再单独接一套BM25。
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
    return _embedding_model


def split_into_parent_chunks(text: str, chunk_size: int = PARENT_CHUNK_SIZE) -> list[dict]:
    """不重叠，从头到尾按字符数切。每块附带它在原文里的char_start/char_end，
    供以后"引用溯源/跳转到原文"使用。"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        chunks.append({
            "text": chunk_text,
            "char_start": i,
            "char_end": i + len(chunk_text),
        })
    return chunks


def split_into_child_chunks(text: str, chunk_size: int = CHILD_CHUNK_SIZE, overlap: int = CHILD_CHUNK_OVERLAP) -> list:
    """步长是chunk_size - overlap，让相邻子块之间有重叠。"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def index_document(file_path: str):
    """读取一个文档，完成父子分块 -> 嵌入(dense+sparse) -> 存储的完整流程。"""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    parents = split_into_parent_chunks(full_text)
    parent_store = _load_parent_store()

    child_records = []
    for position, parent in enumerate(parents):
        parent_id = str(uuid.uuid4())
        parent_store[parent_id] = {
            "text": parent["text"],
            "source_file": file_path,
            "position": position,               # 第几个父块，人类可读的相对序号
            "char_start": parent["char_start"],  # 在原文档里的精确起始位置，用于溯源
            "char_end": parent["char_end"],
        }
        for child_text in split_into_child_chunks(parent["text"]):
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
    # return_dense: 稠密向量，用于语义检索
    # return_sparse: 词法权重(token id -> weight，类似BM25)，用于关键词/专有名词检索，
    #                两者一次前向传播同时拿到，不需要额外跑一遍BM25索引。
    output = model.encode(texts, return_dense=True, return_sparse=True)

    for record, vector, lexical_weights in zip(child_records, output["dense_vecs"], output["lexical_weights"]):
        # 不同版本的FlagEmbedding对dense_vecs是否预归一化处理不完全一致，
        # 这里显式做一次L2归一化，保证后面LanceDB用cosine距离检索时数值是对的。
        vector = np.asarray(vector, dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        record["vector"] = vector.tolist()
        # lexical_weights的key是token id，序列化前统一转成字符串key存进LanceDB，
        # 检索时(retriever.py)要用同样的字符串key格式来做匹配，两边必须一致。
        record["sparse_json"] = json.dumps(
            {str(k): float(v) for k, v in lexical_weights.items()}
        )

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