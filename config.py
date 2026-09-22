"""
全局配置。以后想换模型、换Ollama地址，只改这里，不用动业务代码。
"""

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"     # 按你机器配置换成 4b/7b/14b 等量化档位
OLLAMA_TIMEOUT = 120             # 秒，超过这个时间还没响应就算超时（CPU跑7b生成慢，30s不够）
OLLAMA_NUM_GPU = 999            # None=用Ollama默认策略（自动判断怎么分配GPU/CPU）

# --- RAG基础配置 ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LANCEDB_PATH = "./data/lancedb"
PARENT_STORE_PATH = "./data/parents.json"
PARENT_CHUNK_SIZE = 800   # 父块大小（字符数），保留给LLM看的完整上下文
CHILD_CHUNK_SIZE = 200    # 子块大小（字符数），专门用于向量检索，越小语义越聚焦
CHILD_CHUNK_OVERLAP = 40  # 子块之间的重叠字符数，避免语义被硬生生切断

# --- 混合检索（粗排）相关配置 ---
# bge-m3一次前向传播能同时给出dense向量和sparse(词法)权重，
# 粗排阶段dense和sparse各自独立召回COARSE_TOP_K个子块，
# 用RRF(Reciprocal Rank Fusion)把两路排名融合成一个排序，
# 再映射去重到父块，送去给reranker精排。
COARSE_TOP_K = 20
RRF_K = 60  # RRF公式里的平滑常数，60是社区里最常见的经验值，一般不用改

# --- 重排序（精排）相关配置 ---
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # 多语言reranker；base版纯英文，中文打分近零已弃用
RERANKER_DEVICE = None    # None=交给FlagEmbedding自动探测；也可以强制写"cpu"或"cuda"
RERANK_TOP_K = 6          # 精排后最终返回给LLM的父块数量
RERANK_SCORE_THRESHOLD = 0.0  # 精排分数阈值；单文档RAG下文档总含相关答案，阈值设0总是返回top-N让LLM自己判断，多文档时再调高
# 注意：这个阈值现在作用在reranker的sigmoid归一化分数上（0~1），
# 和过去那个"cosine相似度阈值"量纲完全不同，不能直接沿用旧数值，
# 务必换了模型/换了阈值之后，用50题评测集重新跑一遍再定。

# --- Agent控制循环相关配置 ---
MAX_AGENT_STEPS = 5  # 最多循环几轮，超过还没完成就诚实汇报进展，不硬跑下去

# --- 日志配置 ---
LOG_LEVEL = "INFO"                 # 排查问题时可临时改成"DEBUG"看更详细的内部信息
LOG_FILE_PATH = "./data/logs/app.log"

# --- Memory配置（用户画像长期记忆，与RAG文档库正交）---
MEMORY_DB_PATH = "./data/memory_lancedb"  # 独立DB，避免和RAG的 ./data/lancedb 互相污染
MEMORY_TABLE_NAME = "memories"
MEMORY_DEDUP_THRESHOLD = 0.88      # 余弦相似度超过它视为"同一条记忆"，更新而非新增
MEMORY_IMPORTANCE_THRESHOLD = 4    # importance>=它 实时写，否则进会话末缓冲区统一提炼
MEMORY_RETRIEVAL_TOP_K = 8         # 长期记忆语义召回的上限条数
MEMORY_RECENT_K = 5                # "最近"路召回条数
MEMORY_RANKING_WEIGHTS = {"similarity": 0.5, "importance": 0.3, "recency": 0.2}  # 三项权重和为1
SHORT_TERM_MAX_TOKENS = 4000       # 短期记忆的token预算（按token裁剪，不是按条数）