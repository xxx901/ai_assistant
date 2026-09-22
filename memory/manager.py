"""
记忆编排：写时机(实时 vs 会话末) + 检索门控 + 琐碎缓冲区。

这是唯一对外的门面(MemoryManager)，把短期、长期、提炼、检索串起来：
- 读：retrieve() 只对"非闲聊"的输入做长期多路召回，闲聊直接跳过。
- 写：observe() 里 importance 高的记忆实时入库，低的进琐碎缓冲区；
  flush_session() 在会话结束把缓冲区统一提炼一遍再入库。
"""

import logging
import time

from config import (
    MEMORY_IMPORTANCE_THRESHOLD,
    MEMORY_RETRIEVAL_TOP_K,
)
from llm.ollama_client import OllamaClient
from memory.short_term import ShortTermMemory
from memory.store import MemoryStore
from memory.extractor import extract_memories
from memory.retriever import retrieve as retrieve_long_term

logger = logging.getLogger(__name__)

# 简单闲聊标记：命中即视为闲聊，既不检索记忆、也不实时提炼，只进琐碎缓冲区。
CHIT_CHAT_MARKERS = [
    "你好", "您好", "嗨", "哈喽", "在吗", "早上好", "下午好", "晚上好",
    "谢谢", "感谢", "好的", "好呀", "哦", "嗯", "拜拜", "再见", "晚安", "ok",
]


def is_chit_chat(user_input: str) -> bool:
    text = user_input.strip().lower()
    # 纯表情/超短输入也算闲聊
    if len(text) <= 2:
        return True
    return any(text == m or text.startswith(m) for m in CHIT_CHAT_MARKERS)


class MemoryManager:
    def __init__(self, client: OllamaClient, source: str = None):
        self.client = client
        self.source = source or f"session-{int(time.time())}"
        self.short_term = ShortTermMemory()
        self.store = MemoryStore()
        self.pending: list[str] = []  # 琐碎对话片段缓冲区，会话末统一提炼

    # ---- 读 ----
    def retrieve(self, user_input: str) -> str | None:
        """返回要注入给LLM的"用户画像"文本；闲聊或没召回时返回None。"""
        if is_chit_chat(user_input):
            return None
        memories = retrieve_long_term(self.store, user_input)
        if not memories:
            return None

        top = memories[:MEMORY_RETRIEVAL_TOP_K]
        for m in top:
            self.store.touch(m["id"])

        lines = [
            f"- [{m['type']}] {m['text']}（重要度{m['importance']}）" for m in top
        ]
        return "关于用户，你已知的长期记忆：\n" + "\n".join(lines)

    def messages_with(self, user_input: str) -> list:
        """返回"短期记忆 + 当前用户消息"，给LLM当对话上下文。"""
        return self.short_term.as_messages() + [{"role": "user", "content": user_input}]

    # ---- 写 ----
    def observe(self, user_input: str, assistant_reply: str):
        """一轮对话结束后的写入口：进短期记忆 + 提炼写长期记忆。"""
        self.short_term.append("user", user_input)
        self.short_term.append("assistant", assistant_reply)

        if is_chit_chat(user_input):
            return

        dialogue = f"用户说：{user_input}\n助手回复：{assistant_reply}"
        memories = extract_memories(
            dialogue, self.client, context=self.short_term.to_recall_text()
        )

        created = updated = 0
        for m in memories:
            if m["importance"] >= MEMORY_IMPORTANCE_THRESHOLD:
                action, _ = self.store.upsert(m["text"], m["type"], m["importance"], self.source)
                if action == "created":
                    created += 1
                else:
                    updated += 1
            else:
                # 琐碎内容，暂存原文，会话末统一提炼
                self.pending.append(dialogue)

        if created or updated:
            logger.info(f"记忆实时写入: 新增{created}条 更新{updated}条")

    def flush_session(self) -> int:
        """会话结束：把琐碎缓冲区统一提炼一遍，去重写入。返回写入了多少条。"""
        if not self.pending:
            return 0
        dialogue = "\n\n".join(self.pending)
        memories = extract_memories(dialogue, self.client)
        written = 0
        for m in memories:
            self.store.upsert(m["text"], m["type"], m["importance"], self.source)
            written += 1
        self.pending.clear()
        logger.info(f"会话末统一提炼写入 {written} 条记忆")
        return written
