"""
ReAct循环：思考(Thought) -> 行动(Action) -> 观察(Observation) -> 再思考...
直到LLM判断任务完成，或者到达最大步数上限。

设计要点：
1. agent_state记录整个过程（用户请求 + 每一轮的thought/action/observation），
   每一轮"思考"都把这份完整记录喂给LLM看，让它知道"之前做过什么、看到了什么结果"。
2. 每一轮思考的输出用status字段区分两种情况：
   "continue"（还要继续，带tool+arguments）或"done"（完成，带final_answer）。
3. 执行有副作用的工具之前，先查has_side_effect，需要用户确认才真正执行。
4. 循环有MAX_AGENT_STEPS上限，到点了还没完成，要把已有进展诚实地告诉用户。
5. 不能完全指望模型自己说"done"或者自己意识到"这个操作已经试过了"：
   - 参数一到手就先做规整化（_normalize_arguments），统一反斜杠等格式问题，
     不然同一个文件模型每次吐出的反斜杠数量不一样，字符串比较会误判成"不同的操作"。
   - 用确定性的历史比对(_already_executed)兜底"重复执行"——不管是重复一个
     已经成功的操作，还是固执地重试一个已经失败的操作，都直接拦下来，
     不再问用户、也不再重复执行。用户主动拒绝不算"失败"，不会被拉黑。
"""

import json
import os
import logging
from config import MAX_AGENT_STEPS
from llm.ollama_client import OllamaClient, OllamaClientError
from tools.registry import TOOL_REGISTRY
from tools.tool_caller import execute_tool_call, build_tools_prompt

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT_TEMPLATE = """你是一个任务执行agent，需要通过多轮"思考-行动-观察"来完成用户的请求。

{tools_prompt}

到目前为止发生的事情：
{steps_summary}

重要规则：
1. 如果上面某一步的"观察"里success字段是true，说明那个操作已经成功执行过了，不要再重复执行同一个操作。
2. 每一轮都要对照用户最初的请求，判断"用户想要的结果是否已经达成"——如果已经达成
   （比如观察结果显示目标文件已经存在、目标状态已经满足），应该直接输出status=done，
   不要在没有用户明确要求的情况下，继续做任何额外的、目标之外的操作。

请思考下一步该做什么，只输出一个JSON，不要输出任何其他文字：
- 如果还需要继续调用工具：{{"status": "continue", "thought": "你的思考", "tool": "工具名", "arguments": {{"参数名": "参数值"}}}}
- 如果任务已经完成，可以给用户答案了：{{"status": "done", "thought": "你的思考", "final_answer": "给用户的最终回复"}}
"""


def _format_steps_summary(steps: list) -> str:
    if not steps:
        return "（还没有执行过任何操作）"
    lines = []
    for i, step in enumerate(steps, 1):
        action_json = json.dumps(step["action"], ensure_ascii=False)
        observation_json = json.dumps(step["observation"], ensure_ascii=False)
        lines.append(f"第{i}步思考: {step['thought']}")
        lines.append(f"第{i}步行动: {action_json}")
        lines.append(f"第{i}步观察: {observation_json}")
    return "\n".join(lines)


def _think(agent_state: dict, client: OllamaClient) -> dict:
    """让LLM根据目前的进展，决定下一步该做什么。解析失败时安全兜底为done，不让循环崩掉。"""
    tools_prompt = build_tools_prompt()
    steps_summary = _format_steps_summary(agent_state["steps"])
    system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        tools_prompt=tools_prompt, steps_summary=steps_summary
    )

    try:
        raw = client.chat(
            messages=[{"role": "user", "content": agent_state["user_request"]}],
            stream=False,
            system=system_prompt,
        )
    except OllamaClientError as e:
        logger.warning(f"LLM调用失败，agent循环兜底为done: {e}")
        return {
            "status": "done",
            "thought": f"思考过程出错: {e}",
            "final_answer": "抱歉，处理这个请求时遇到了问题，可以换个方式再说一遍吗？",
        }

    try:
        data = json.loads(raw)
        if data.get("status") not in ("continue", "done"):
            raise ValueError("status字段不合法")
        return data
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning(f"模型输出解析失败，agent循环兜底为done: {e}")
        logger.debug(f"模型原始输出: {raw!r}")
        return {
            "status": "done",
            "thought": f"思考过程出错: {e}",
            "final_answer": "抱歉，处理这个请求时遇到了问题，可以换个方式再说一遍吗？",
        }


def _normalize_arguments(arguments: dict) -> dict:
    """只做无损规范化：合并连续的反斜杠/斜杠，不改变路径结构。"""
    normalized = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            # 多个连续反斜杠 -> 单个；多个连续正斜杠 -> 单个
            while "\\\\" in value:
                value = value.replace("\\\\", "\\")
            while "//" in value:
                value = value.replace("//", "/")
            normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def _already_executed(agent_state: dict, tool: str, arguments: dict):
    """
    检查历史步骤里有没有跟这次完全一样(工具名、参数都一样)的操作，且当时成功或失败过。

    - 防"成功后重复执行"：LLM在操作已成功后还提同样的操作
    - 防"失败后重复执行"：LLM在工具报错后固执地重试同一个操作

    注意：用户主动拒绝(confirm_callback返回False)不算"失败"，因为那是用户的选择，
    不是操作本身有问题——否则用户拒绝一次后，这个操作就被永久拉黑，
    连改主意的机会都没有了。

    返回 (outcome, detail)：
    - ("succeeded", 那次的output)  如果历史上成功过（优先）
    - ("failed",    那次的error)   如果历史上失败过（且不是用户拒绝）
    - (None, None)                  如果没执行过，或只有用户拒绝的记录
    """
    succeeded = None
    failed = None

    for step in agent_state["steps"]:
        obs = step["observation"]
        if not isinstance(obs, dict):
            continue
        if step["action"]["tool"] != tool or step["action"]["arguments"] != arguments:
            continue

        if obs.get("success"):
            succeeded = obs.get("output")
        elif obs.get("success") is False and obs.get("error") != "用户拒绝执行这个操作":
            failed = obs.get("error")

    if succeeded is not None:
        return "succeeded", succeeded
    if failed is not None:
        return "failed", failed
    return None, None


def _success_summary(tool: str, output) -> str:
    """把一次已成功的工具执行结果，转成一句给用户的最终回复。"""
    result_text = json.dumps(output, ensure_ascii=False)
    return f"已完成「{tool}」操作，结果：{result_text}"


def _failure_summary(tool: str, error) -> str:
    """把一次重复出现的失败结果，转成一句诚实告知用户的话，而不是再问一次确认。"""
    error_text = json.dumps(error, ensure_ascii=False) if not isinstance(error, str) else error
    return (
        f"「{tool}」操作之前已经执行过一次，但没有成功，错误信息是：{error_text}。\n"
        f"我没有再重复尝试同一个操作。请检查一下输入的信息（比如文件名、路径是否正确），"
        f"或者换一种方式告诉我你想怎么做？"
    )


def run_agent_loop(user_request: str, client: OllamaClient, confirm_callback) -> str:
    """
    confirm_callback: 一个函数，签名是 (tool: str, arguments: dict) -> bool，
                       返回True表示用户同意执行这个有副作用的操作，False表示拒绝。
    """
    agent_state = {"user_request": user_request, "steps": []}

    for _ in range(MAX_AGENT_STEPS):
        decision = _think(agent_state, client)

        if decision["status"] == "done":
            return decision["final_answer"]

        tool = decision.get("tool")
        arguments = _normalize_arguments(decision.get("arguments", {}))

        # 确定性兜底：这次要做的操作，历史上是不是已经做过（成功或失败）？
        outcome, detail = _already_executed(agent_state, tool, arguments)
        if outcome == "succeeded":
            return _success_summary(tool, detail)
        if outcome == "failed":
            return _failure_summary(tool, detail)

        tool_info = TOOL_REGISTRY.get(tool)
        if tool_info and tool_info.get("has_side_effect"):
            approved = confirm_callback(tool, arguments)
            if not approved:
                agent_state["steps"].append({
                    "thought": decision.get("thought", ""),
                    "action": {"tool": tool, "arguments": arguments},
                    "observation": {"success": False, "error": "用户拒绝执行这个操作"},
                })
                continue

        result = execute_tool_call(tool, arguments)
        logger.info(f"工具执行完成: tool={tool}, arguments={arguments}, result={result}")
        agent_state["steps"].append({
            "thought": decision.get("thought", ""),
            "action": {"tool": tool, "arguments": arguments},
            "observation": result,
        })

    return _summarize_unfinished(agent_state)


def _summarize_unfinished(agent_state: dict) -> str:
    lines = [f"这个任务比较复杂，我已经尝试了{MAX_AGENT_STEPS}步，还没完全搞定，目前的进展是："]
    for i, step in enumerate(agent_state["steps"], 1):
        lines.append(f"{i}. {step['thought']}")
    lines.append("要不要告诉我接下来怎么继续，或者换个更具体的说法？")
    return "\n".join(lines)