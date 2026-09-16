"""
Memory模块：
- 短期上下文：只保留最近MAX_HISTORY_MESSAGES条，避免无限增长撑爆上下文窗口、拖慢推理速度。
- 长期持久化：对话历史存成本地JSON文件，下次启动能接着上次的对话继续聊。
"""

import json
import os

HISTORY_PATH = "./data/history.json"
MAX_HISTORY_MESSAGES = 20


def load_history() -> list:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """只保留最近max_messages条，超过的部分直接丢弃（v1先不做摘要压缩）。"""
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]