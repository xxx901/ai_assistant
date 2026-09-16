"""
ReAct循环：思考(Thought) -> 行动(Action) -> 观察(Observation) -> 再思考...
直到LLM判断任务完成，或者到达最大步数上限。

设计要点：
1. agent_state记录整个过程（用户请求 + 每一轮的thought/action/observation），
   每一轮"思考"都把这份完整记录喂给LLM看，让它知道"之前做过什么、看到了什么结果"。
2. 每一轮思考的输出用status字段区分两种情况：
   "continue"（还要继续，带tool+arguments）或"done"（完成，带final_answer）。
3. 执行有副作用的工具之前，先查has_side_effect，需要用户确认才真正执行——
   这个判断加在"要不要调用execute_tool_call"这一层，execute_tool_call本身完全不用改。
4. 循环有MAX_AGENT_STEPS上限，到点了还没完成，要把已有进展诚实地告诉用户，
   不能装作完成、也不能沉默失败。
5. 不能完全指望模型自己说"done"：qwen2.5:7b实测中会在操作已经成功之后，
   继续提出一模一样的操作。这里用确定性的历史比对兜底——如果这次要做的
   (tool, arguments)跟历史上某一步完全一样、且那一步已经成功过，直接把
   当时的结果当最终答案返回，不再重复执行、也不再重复问用户确认。
   注意：这个兜底只拦"完全重复的操作"，不会阻止agent连续执行几次
   不同参数的有副作用操作（比如连续重命名两个不同的文件）。
"""

import json
from config import MAX_AGENT_STEPS
from llm.ollama_client import OllamaClient, OllamaClientError
from tools.registry import TOOL_REGISTRY
from tools.tool_caller import execute_tool_call, build_tools_prompt

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
        print(f"[调试] LLM调用失败: {e}")
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
        print(f"[调试] 解析失败: {e}\n[调试] 模型原始输出: {raw!r}")
        return {
            "status": "done",
            "thought": f"思考过程出错: {e}",
            "final_answer": "抱歉，处理这个请求时遇到了问题，可以换个方式再说一遍吗？",
        }


def _already_succeeded(agent_state: dict, tool: str, arguments: dict):
    """ attention：导致系统变慢的主要原因
    检查历史步骤里有没有跟这次完全一样(工具名、参数都一样)、且当时已经成功过的操作。
    跟历史上'任何一步'比对，不只是紧邻的上一步——因为中间可能隔了几次
    只读操作（比如list_files），重复的操作不一定紧挨着出现。
    返回 (True, 那次的output) 或 (False, None)。
    """
    for step in agent_state["steps"]:
        obs = step["observation"]
        if (
            isinstance(obs, dict)
            and obs.get("success") is True
            and step["action"]["tool"] == tool
            and step["action"]["arguments"] == arguments
        ):
            return True, obs.get("output")
    return False, None


def _success_summary(tool: str, output) -> str:
    """把一次已成功的工具执行结果，转成一句给用户的最终回复。"""
    result_text = json.dumps(output, ensure_ascii=False)
    return f"已完成「{tool}」操作，结果：{result_text}"


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
        arguments = decision.get("arguments", {})

        # 确定性兜底：这次要做的操作，历史上是不是已经做过且成功了？
        # 不依赖模型自己"意识到"重复，直接用记录比对。
        already_done, prior_output = _already_succeeded(agent_state, tool, arguments)
        if already_done:
            return _success_summary(tool, prior_output)

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

        print(f"[调试] 执行结果: {result}")
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