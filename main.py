"""
模块7+8联调：接入Memory（对话历史持久化+裁剪），启动时做Ollama健康检查，
四个核心模块（路由/工具/RAG/技能）+ Memory，正式整合成一个完整可用的程序。
"""

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_NUM_GPU
from llm.ollama_client import OllamaClient, OllamaClientError, is_ollama_running
from core.router import route
from core.agent_loop import run_agent_loop
from core.memory import load_history, save_history, trim_history
import time
from skills.skill_runner import run_skill
from rag.retriever import retrieve


def confirm_callback(tool: str, arguments: dict) -> bool:
    print(f"\n[需要确认] 即将执行操作：{tool}，参数：{arguments}")
    answer = input("是否执行？(y/n): ").strip().lower()
    return answer == "y"


def answer_with_rag(user_input: str, client: OllamaClient) -> str:
    result = retrieve(user_input)
    if not result["found"]:
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
    return f"{answer}\n\n（参考来源：{sources}）"


def main():
    if not is_ollama_running(OLLAMA_BASE_URL):
        print(f"[启动失败] 连不上Ollama服务，请先执行 `ollama serve`，并确认已拉取模型 {OLLAMA_MODEL}")
        return

    client = OllamaClient(
        base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT, num_gpu=OLLAMA_NUM_GPU
    )
    history = load_history()
    print(f"本地AI助手已启动（模型：{OLLAMA_MODEL}），输入 exit 退出\n")
    if history:
        print(f"（已加载上次的对话记录，共{len(history)}条）\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "exit":
            save_history(history)
            print("对话已保存，下次见。")
            break
        if not user_input:
            continue

        result = route(user_input, client, history=history)
        intent, source = result["intent"], result["source"]
        print(f"[路由结果] intent={intent}, source={source}")

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
                    print(f"\n[调试] 对话首Token延迟: {time.time() - chat_start:.2f}秒")
                    print("助手: ", end="", flush=True)
                    first_chunk_logged = True
                print(chunk, end="", flush=True)
                full_reply += chunk
            print()
            history.append({"role": "assistant", "content": full_reply})
            history = trim_history(history)
            save_history(history)
        except OllamaClientError as e:
            print(f"\n[出错了] {e}")


if __name__ == "__main__":
    main()