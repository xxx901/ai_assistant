"""
Skills注册表：每个技能是一个"提示词模板"，把用户内容套进去、丢给LLM即可。
跟tools/registry.py的思路一致（字典存储、可扩展），但技能不操作外部世界，
不需要has_side_effect、不需要真实Python函数，只需要一段提示词模板。
"""

SKILL_REGISTRY = {
    "translate": {
        "description": "将用户提供的内容翻译成目标语言（默认英文）",
        "prompt_template": "请将以下内容翻译成{target_lang}，只输出翻译结果，不要解释：\n\n{content}",
    },
    "summarize": {
        "description": "对用户提供的内容进行简明扼要的总结",
        "prompt_template": "请对以下内容进行简明扼要的总结：\n\n{content}",
    },
    "meeting_notes": {
        "description": "将会议记录整理成结构化的会议纪要",
        "prompt_template": (
            "请把以下会议记录整理成结构化的会议纪要"
            "（包含：讨论主题、关键决定、待办事项）：\n\n{content}"
        ),
    },
    "code_explain": {
        "description": "解释一段代码的作用和逻辑",
        "prompt_template": "请逐步解释以下代码的作用和逻辑：\n\n{content}",
    },
}