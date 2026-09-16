"""
工具调用的决策与执行。

流程：
1. build_tools_prompt()  把TOOL_REGISTRY里所有工具的说明拼成文字，塞进system prompt
2. decide_tool_call()    把用户输入和工具列表一起丢给LLM，要求输出 {"tool":..., "arguments":{...}}
3. execute_tool_call()   校验工具名和参数是否合法，合法才真正执行对应函数

设计选择：这里的失败（工具不存在/缺参数/执行报错）用返回值 {"success": False, "error": ...}
表达，不用异常抛出去。因为LLM选错工具、漏传参数是模型能力限制下"预期内会发生"的情况，
调用方（未来的agent_loop）大概率要拿着这个错误信息回头重新问一次LLM，属于正常业务流程，
不是意外崩溃——跟ollama_client.py里"服务连不上"这种真正意外的情况不是一回事。
"""

import json
from tools.registry import TOOL_REGISTRY
from llm.ollama_client import OllamaClient, OllamaClientError


def build_tools_prompt() -> str:
    """
    只负责生成"有哪些工具可用"这段说明文字，不包含输出格式要求——
    格式要求由具体调用场景各自决定并附加（decide_tool_call和agent_loop
    要求的JSON格式不一样，混在一起会让模型同时看到两条互相矛盾的格式指令）。
    """
    lines = ["你可以调用以下工具："]
    for name, info in TOOL_REGISTRY.items():
        # 用json.dumps而不是直接把字典塞进f-string，
        # 保证展示给模型看的格式是标准JSON（双引号、小写true/false），
        # 避免模型学着Python字典的写法（单引号、大写True）来输出，导致json.loads解析失败。
        params_json = json.dumps(info["parameters"], ensure_ascii=False)
        lines.append(f"- {name}: {info['description']}，参数说明: {params_json}")
    return "\n".join(lines)


def decide_tool_call(user_input: str, client: OllamaClient) -> dict:
    """向LLM请求一次工具调用决策，返回结构化结果，不抛异常。"""
    system_prompt = build_tools_prompt() + (
        "\n只输出一个JSON，格式严格如下，不要输出任何其他文字、不要加解释：\n"
        '{"tool": "工具名", "arguments": {"参数名": "参数值"}}'
    )
    try:
        raw = client.chat(
            messages=[{"role": "user", "content": user_input}],
            stream=False,
            system=system_prompt,
        )
        data = json.loads(raw)
        return {
            "success": True,
            "tool": data.get("tool"),
            "arguments": data.get("arguments", {}),
        }
    except (OllamaClientError, json.JSONDecodeError, AttributeError) as e:
        return {"success": False, "error": f"无法从模型输出解析出工具调用: {e}"}


def validate_arguments(parameters: dict, arguments: dict) -> str | None:
    """
    检查arguments里是否包含parameters中所有required=True的参数。
    缺任意一个必填参数，返回说明缺了哪个的错误字符串；全部齐全返回None。
    """
    for name, info in parameters.items():
        if info.get("required"):
            if name not in arguments:
                return f"缺少必填参数：{name}"
    return None


def execute_tool_call(tool: str, arguments: dict) -> dict:
    """校验+执行，返回 {"success": True, "output": ...} 或 {"success": False, "error": ...}。"""
    if tool not in TOOL_REGISTRY:
        return {"success": False, "error": f"工具不存在: {tool}"}

    tool_info = TOOL_REGISTRY[tool]
    validation_error = validate_arguments(tool_info["parameters"], arguments)
    if validation_error:
        return {"success": False, "error": validation_error}

    try:
        output = tool_info["func"](**arguments)
        return {"success": True, "output": output}
    except Exception as e:
        # 注意：这里罕见地用了宽泛的except Exception，是有意为之——
        # 这是整个调用链的"最外层边界"，任何一个具体工具函数内部出什么错，
        # 都不能让整个agent崩掉，必须在这里兜住并转换成统一的错误格式。
        # 这跟"平时不要用except Exception"的建议不矛盾：
        # 边界层兜底 vs 内部代码偷懒式地吞掉异常，是两回事。
        return {"success": False, "error": f"工具执行失败: {e}"}