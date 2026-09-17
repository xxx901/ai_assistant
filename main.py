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
from core.memory import load_history, save_history, trim_history
from skills.skill_runner import run_skill
from rag.retriever import retrieve

logger = logging.getLogger(__name__)


def confirm_callback(tool: str, arguments: dict) -> bool:
    """向用户确认是否执行有副作用的操作——这是终端界面文字，不是日志，所以用print/input。"""
    print(f"\n[需要确认] 即将执行操作：{tool}，参数：{arguments}")
    answer = input("是否执行？(y/n): ").strip().lower()
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
    history = load_history()
    print(f"本地AI助手已启动（模型：{OLLAMA_MODEL}），输入 exit 退出\n")
    if history:
        print(f"（已加载上次的对话记录，共{len(history)}条）\n")
    logger.info(f"启动完成，模型={OLLAMA_MODEL}，历史记录条数={len(history)}")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "exit":
            save_history(history)
            logger.info("用户退出，历史记录已保存")
            print("对话已保存，下次见。")
            break
        if not user_input:
            continue

        result = route(user_input, client, history=history)
        intent, source = result["intent"], result["source"]
        logger.info(f"路由结果: intent={intent}, source={source}")

        if intent == "file_tool":
            reply = run_agent_loop(user_input, client, confirm_callback)
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

        history.append({"role": "user", "content": user_input})
        try:
            print("助手: ", end="", flush=True)
            full_reply = ""
            chat_start = time.time()
            first_chunk_logged = False
            for chunk in client.chat(history, stream=True):
                if not first_chunk_logged:
                    logger.info(f"对话首Token延迟: {time.time() - chat_start:.2f}秒")
                    first_chunk_logged = True
                print(chunk, end="", flush=True)
                full_reply += chunk
            print()
            history.append({"role": "assistant", "content": full_reply})
            history = trim_history(history)
            save_history(history)
        except OllamaClientError as e:
            logger.error(f"对话调用失败: {e}")
            print(f"\n[出错了] {e}")


if __name__ == "__main__":
    main()