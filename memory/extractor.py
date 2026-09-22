"""
记忆提炼器：把一段对话提炼成结构化的记忆条目（不是摘要）。

设计：
- 用LLM结构化输出(JSON)从对话片段里提取关于"用户"的、值得长期记住的信息，
  每条带 type 和 importance。
- 输出格式在system prompt里写死，解析用 json.loads + 逐字段校验，
  失败返回[]并log，绝不让提炼异常打断主对话。
"""

import json
import logging

from llm.ollama_client import OllamaClient, OllamaClientError
from memory.store import VALID_TYPES

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM = """你是一个记忆提炼器。从下面的对话里提取关于"用户"的、值得长期记住的信息，输出结构化的记忆条目。

可提取的四种类型（type）：
- preference: 用户的偏好、习惯、喜好（例如"喜欢用中文交流"）
- fact: 关于用户的客观事实（例如"用户名叫小明"、"用户在做某个项目"）
- entity: 用户提到的专有名词/实体（例如"LanceDB"、"Transformer"），text只写实体名本身
- event: 用户做过或正在做的事（例如"用户今天开始重构记忆系统"）

importance 是1~5的整数：5=极其重要（身份、长期偏好），1=琐碎细节。宁可少提取，也不要提取无关紧要或重复的内容。

只输出一个JSON，格式严格如下，不要输出任何其他文字、不要加解释、不要用markdown代码块包裹：
{"memories": [{"text": "记忆内容", "type": "preference", "importance": 4}]}
如果这段对话没有值得记住的信息，输出 {"memories": []}
"""


def extract_memories(dialogue: str, client: OllamaClient, context: str = None) -> list:
    """从对话片段里提炼记忆，返回 [{"text","type","importance"}, ...]。

    dialogue: 一段对话文本（比如"用户说：... / 助手回复：..."）
    context:  可选的近期上下文，帮助判断指代（比如"用户叫什么"里的"用户"是谁）
    """
    prompt = dialogue
    if context:
        prompt = f"近期上下文（帮助判断指代）：\n{context}\n\n{dialogue}"

    try:
        raw = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            system=EXTRACTOR_SYSTEM,
            format_json=True,
        )
    except OllamaClientError as e:
        logger.warning(f"记忆提炼LLM调用失败: {e}")
        return []

    return _parse(raw)


def _parse(raw: str) -> list:
    """解析提炼器输出，逐字段校验，任何不合法都跳过，绝不抛异常。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"记忆提炼输出不是合法JSON: {raw[:200]!r}")
        return []

    items = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    memories = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        type_ = item.get("type")
        if type_ not in VALID_TYPES:
            type_ = "fact"
        try:
            importance = max(1, min(5, int(item.get("importance", 3))))
        except (TypeError, ValueError):
            importance = 3
        memories.append({"text": text.strip(), "type": type_, "importance": importance})
    return memories
