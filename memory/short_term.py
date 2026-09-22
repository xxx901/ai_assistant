"""
短期记忆：只保留最近一段对话，但按 token 数裁剪（不是按条数）。

设计：
- 内部仍是 [{role, content}, ...] 的消息列表，和LLM的messages格式对齐，
  方便直接喂给 OllamaClient.chat()。
- 裁剪阈值是 token 预算而不是条数：一条长文本可能顶十条短回复，按条数裁
  会误伤；所以从最新一条往前累加估算的 token 数，超预算就丢弃更旧的消息。
- token 数是粗略估算（不引 tiktoken 这个重依赖）：中文≈1 token/字、
  其他字符≈1 token/4字符，对"裁剪"这个用途精度足够。
"""

import re

from config import SHORT_TERM_MAX_TOKENS


def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数。中文按1字1 token、其余按4字符1 token算。"""
    cjk = len(re.findall(r"[一-鿿]", text))
    other = len(text) - cjk
    return cjk + other // 4


class ShortTermMemory:
    def __init__(self, max_tokens: int = SHORT_TERM_MAX_TOKENS):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def append(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self):
        # 从最新一条往前累加，超预算就停；至少保留最新一条（哪怕它自己就超预算）。
        total = 0
        keep: list[dict] = []
        for msg in reversed(self.messages):
            total += estimate_tokens(msg["content"])
            if total > self.max_tokens and keep:
                break
            keep.append(msg)
        self.messages = list(reversed(keep))

    def as_messages(self) -> list:
        return list(self.messages)

    def to_recall_text(self) -> str:
        """把短期记忆拼成一段文本，给记忆提炼器当原料（帮助判断指代）。"""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

    def __len__(self) -> int:
        return len(self.messages)
