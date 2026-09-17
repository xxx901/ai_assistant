"""
全局配置。以后想换模型、换Ollama地址，只改这里，不用动业务代码。
"""

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"     # 按你机器配置换成 4b/7b/14b 等量化档位
OLLAMA_TIMEOUT = 30              # 秒，超过这个时间还没响应就算超时
OLLAMA_NUM_GPU = None            # None=用Ollama默认策略（自动判断怎么分配GPU/CPU）

# --- RAG相关配置 ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LANCEDB_PATH = "./data/lancedb"
PARENT_STORE_PATH = "./data/parents.json"
PARENT_CHUNK_SIZE = 800   # 父块大小（字符数），保留给LLM看的完整上下文
CHILD_CHUNK_SIZE = 200    # 子块大小（字符数），专门用于向量检索，越小语义越聚焦
CHILD_CHUNK_OVERLAP = 40  # 子块之间的重叠字符数，避免语义被硬生生切断
RAG_SCORE_THRESHOLD = 0.35  # 相关度阈值，初始值，后面要用真实问题测试后再调

# --- Agent控制循环相关配置 ---
MAX_AGENT_STEPS = 5  # 最多循环几轮，超过还没完成就诚实汇报进展，不硬跑下去

# --- 日志配置 ---
LOG_LEVEL = "INFO"                 # 排查问题时可临时改成"DEBUG"看更详细的内部信息
LOG_FILE_PATH = "./data/logs/app.log"