"""
意图路由模块。

设计原则回顾：
1. 分层路由——这一层只判断粗粒度的四大类(chat/rag/file_tool/skill)，
   具体调用哪个skill、哪个工具，交给下游模块（模块3/模块6）决定。
2. 规则优先，LLM兜底——表达方式固定的类别用关键词判断（快、准、零延迟），
   语义模糊的类别交给LLM（泛化能力强）。
3. 不确定时退到风险最低的分支——分类失败/不合法时统一归为chat，
   而不是随便选一个可能触发危险操作的类别。
"""

import json
import time
import logging
from llm.ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger(__name__)

VALID_INTENTS = {"chat", "rag", "file_tool", "skill"}

# 关键词规则表：命中即直接判定为对应大类，不需要走LLM。
# 注意：这里只到"file_tool"/"skill"这个大类粒度，
# 具体是翻译还是摘要，是模块6要管的事，这一层不关心。
KEYWORD_RULES = {
    "file_tool": ["整理", "分类", "重命名", "去重", "文件夹"],
    "skill": ["翻译", "摘要", "总结", "会议纪要", "解释这段代码", "代码解释"],
}

ROUTER_SYSTEM_PROMPT = """你是一个意图分类器。根据用户输入，从以下四个类别中选择最合适的一个：
- chat: 日常聊天、闲聊、通用问答，不需要查资料或操作文件
- rag: 用户在询问某个具体文档/资料/历史记录里的内容
- file_tool: 用户想要整理、分类、重命名、去重文件
- skill: 用户想要翻译、摘要、生成会议纪要、解释代码等特定技能

只输出一个JSON，格式严格如下，不要输出任何其他文字、不要加解释、不要用markdown代码块包裹：
{"intent": "chat"}
"""


def match_by_keywords(keyword_rules: dict, user_input: str) -> str | None:
    """
    遍历 keyword_rules，检查 user_input 里有没有出现某个类别下的任意关键词，
    命中就返回对应的类别名，一个都没命中就返回 None。
    """
    for name, keywords in keyword_rules.items():
        for keyword in keywords:
            if keyword in user_input:
                return name
    return None


def classify_by_llm(user_input: str, client: OllamaClient) -> tuple[str, str]:
    """
    LLM兜底分类，返回 (intent, source)。
    source 为 "llm" 表示LLM给出了合法分类；
    source 为 "fallback" 表示解析失败/类别不合法，安全兜底为chat。
    """
    try:
        start = time.time()
        raw = client.chat(
            messages=[{"role": "user", "content": user_input}],
            stream=False,
            system=ROUTER_SYSTEM_PROMPT,
        )
        logger.info(f"路由分类耗时: {time.time() - start:.2f}秒")
        data = json.loads(raw)
        intent = data.get("intent")
        if intent in VALID_INTENTS:
            return intent, "llm"
        logger.warning(f"路由分类返回了不合法的intent: {intent!r}，兜底为chat")
        return "chat", "fallback"
    except (OllamaClientError, json.JSONDecodeError, AttributeError) as e:
        # OllamaClientError: 模型调用本身失败（服务挂了/超时）
        # JSONDecodeError: 模型没按要求输出JSON
        # AttributeError: data不是字典（比如模型返回了一个列表或纯文本）
        logger.warning(f"路由分类失败，兜底为chat: {e}")
        return "chat", "fallback"


def route(user_input: str, client: OllamaClient, history: list = None) -> dict:
    """
    history 目前暂不使用，为以后处理上下文依赖（指代、省略句）预留接口。
    """
    rule_result = match_by_keywords(KEYWORD_RULES, user_input)
    if rule_result:
        return {"intent": rule_result, "source": "rule"}

    intent, source = classify_by_llm(user_input, client)
    return {"intent": intent, "source": source}