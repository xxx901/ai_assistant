"""
入口：Ollama健康检查 -> 路由 -> 四大功能分发（chat/rag/file_tool/skill）-> Memory持久化。

日志说明：程序内部状态一律走logging（控制台+落盘到 data/logs/app.log），
print只保留给真正的用户界面文字（聊天内容、确认提示、启动/退出提示）。
"""

import time
import logging

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_NUM_GPU
from core.logging_config import setup_logging
from llm.ollama_client import OllamaClient, OllamaClientError, is_ollama_running
from core.router import route
from core.agent_loop import run_agent_loop
from memory import MemoryManager
from skills.skill_runner import run_skill
from rag.retriever import retrieve
from tools.file_ops import set_llm_client

logger = logging.getLogger(__name__)


def confirm_callback(tool: str, arguments: dict) -> bool:
    """向用户确认是否执行有副作用的操作——这是终端界面文字，不是日志，所以用print/input。"""
    print(f"\n[需要确认] 即将执行操作：{tool}，参数：{arguments}")
    answer = input("是否执行？(y/n): ").strip().lower()
    return answer == "y"


def confirm_plan_callback(summary: str, operations: list) -> bool:
    """向用户展示批量操作计划并确认——终端界面文字，用print/input。"""
    print(f"\n[计划预览] {summary}")
    for i, op in enumerate(operations, 1):
        print(f"  {i}. {op['tool']}({op['arguments']})")
    answer = input("是否执行以上操作？(y/n): ").strip().lower()
    return answer == "y"


def answer_with_rag(user_input: str, client: OllamaClient) -> str:
    result = retrieve(user_input)
    if not result["found"]:
        logger.info("RAG未检索到相关内容，退化为通用回答")
        fallback = client.chat(messages=[{"role": "user", "content": user_input}], stream=False)
        return "（未在知识库中找到相关内容，以下是基于通用知识的回答）\n" + fallback

    context_text = "\n\n".join(
        f"[来源: {c['source_file']}]\n{c['text']}" for c in result["contexts"]
    )
    system_prompt = "请结合以下参考资料回答用户问题，如果资料不足以回答，如实说明：\n\n" + context_text
    answer = client.chat(
        messages=[{"role": "user", "content": user_input}], stream=False, system=system_prompt
    )
    sources = "、".join(sorted({c["source_file"] for c in result["contexts"]}))
    logger.info(f"RAG命中并返回结果，来源: {sources}")
    return f"{answer}\n\n（参考来源：{sources}）"


def main():
    setup_logging()
    logger.info("助手启动中...")

    if not is_ollama_running(OLLAMA_BASE_URL):
        logger.error(f"连不上Ollama服务（{OLLAMA_BASE_URL}）")
        print(f"[启动失败] 连不上Ollama服务，请先执行 `ollama serve`，并确认已拉取模型 {OLLAMA_MODEL}")
        return

    client = OllamaClient(
        base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT, num_gpu=OLLAMA_NUM_GPU
    )
    set_llm_client(client)
    memory = MemoryManager(client)
    print(f"本地AI助手已启动（模型：{OLLAMA_MODEL}），输入 exit 退出\n")
    logger.info(f"启动完成，模型={OLLAMA_MODEL}，长期记忆条数={memory.store.count()}")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "exit":
            written = memory.flush_session()
            logger.info(f"用户退出，会话末提炼写入 {written} 条记忆")
            print(f"对话结束，本次额外整理了 {written} 条记忆，下次见。")
            break
        if not user_input:
            continue

        result = route(user_input, client)
        intent, source = result["intent"], result["source"]
        logger.info(f"路由结果: intent={intent}, source={source}")

        if intent == "file_tool":
            reply = run_agent_loop(user_input, client, confirm_callback, confirm_plan_callback)
            print(f"助手: {reply}\n")
            continue

        if intent == "skill":
            reply = run_skill(user_input, client)
            print(f"助手: {reply}\n")
            continue

        if intent == "rag":
            reply = answer_with_rag(user_input, client)
            print(f"助手: {reply}\n")
            continue

        mem_context = memory.retrieve(user_input)
        messages = memory.messages_with(user_input)
        try:
            print("助手: ", end="", flush=True)
            full_reply = ""
            chat_start = time.time()
            first_chunk_logged = False
            for chunk in client.chat(messages, stream=True, system=mem_context):
                if not first_chunk_logged:
                    logger.info(f"对话首Token延迟: {time.time() - chat_start:.2f}秒")
                    first_chunk_logged = True
                print(chunk, end="", flush=True)
                full_reply += chunk
            print()
            memory.observe(user_input, full_reply)
        except OllamaClientError as e:
            logger.error(f"对话调用失败: {e}")
            print(f"\n[出错了] {e}")


if __name__ == "__main__":
    main()