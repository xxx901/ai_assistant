"""
Skills执行：先用LLM判断该用哪个技能、抽取需要的信息（比如目标语言），
再把用户内容套进对应模板，第二次调用LLM生成最终结果。
"""

import json
from skills.registry import SKILL_REGISTRY
from llm.ollama_client import OllamaClient, OllamaClientError

DECIDE_SKILL_PROMPT = """你是一个技能分发器。根据用户输入，判断该使用以下哪个技能，
并提取出要处理的正文内容(content)，如果是翻译还要提取目标语言(target_lang，默认"英文")：

{skills_desc}

只输出一个JSON，不要输出其他文字，格式：
{{"skill": "技能名", "content": "要处理的正文", "target_lang": "目标语言(仅translate需要)"}}
"""


def _build_skills_desc() -> str:
    lines = [f"- {name}: {info['description']}" for name, info in SKILL_REGISTRY.items()]
    return "\n".join(lines)


def decide_skill(user_input: str, client: OllamaClient) -> dict:
    """返回 {"success": True, "skill":..., "content":..., "target_lang":...} 或 {"success": False, "error":...}"""
    system_prompt = DECIDE_SKILL_PROMPT.format(skills_desc=_build_skills_desc())
    try:
        raw = client.chat(
            messages=[{"role": "user", "content": user_input}],
            stream=False,
            system=system_prompt,
        )
        data = json.loads(raw)
        skill = data.get("skill")
        if skill not in SKILL_REGISTRY:
            return {"success": False, "error": f"无法识别的技能: {skill}"}
        return {
            "success": True,
            "skill": skill,
            "content": data.get("content", user_input),
            "target_lang": data.get("target_lang", "英文"),
        }
    except (OllamaClientError, json.JSONDecodeError, AttributeError) as e:
        return {"success": False, "error": f"技能识别失败: {e}"}


def run_skill(user_input: str, client: OllamaClient) -> str:
    """完整流程：识别技能 -> 套用模板 -> 生成结果，返回一段可以直接展示给用户的文字。"""
    decision = decide_skill(user_input, client)
    if not decision["success"]:
        return f"抱歉，没能理解你想用哪个技能：{decision['error']}"

    skill_info = SKILL_REGISTRY[decision["skill"]]
    prompt = skill_info["prompt_template"].format(
        content=decision["content"], target_lang=decision.get("target_lang", "英文")
    )

    try:
        return client.chat(messages=[{"role": "user", "content": prompt}], stream=False)
    except OllamaClientError as e:
        return f"抱歉，处理这个技能请求时出错了：{e}"