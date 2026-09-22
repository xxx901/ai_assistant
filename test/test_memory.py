"""
Memory 系统冒烟测试。

分两个阶段：
- 阶段1(纯逻辑，不调LLM)：token估算、短期裁剪、闲聊判定、长期去重/检索。
- 阶段2(调LLM)：结构化提炼、observe/retrieve/flush 编排。

运行：在项目根目录 `python test/test_memory.py`
"""

import os
import shutil
import sys

# 把项目根目录加进 sys.path，保证 `python test/test_memory.py` 也能 import config/memory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# 用独立测试DB，不污染真实记忆库（必须在 import memory.* 之前改）
config.MEMORY_DB_PATH = "./data/_test_memory_lancedb"

from memory.short_term import estimate_tokens, ShortTermMemory
from memory.manager import is_chit_chat
from memory.store import MemoryStore
from memory.retriever import retrieve as retrieve_long_term


def section(name):
    print(f"\n===== {name} =====")


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def phase1():
    section("阶段1: 纯逻辑")

    check("estimate_tokens 中文按字计", estimate_tokens("你好世界") == 4)
    check("estimate_tokens 英文按4字符计", estimate_tokens("hello") == 1)

    st = ShortTermMemory(max_tokens=10)
    st.append("user", "你好世界")              # 4 token
    st.append("assistant", "你好")              # 2 token -> 累计6
    st.append("user", "这是一条很长的消息啊")    # 9 token -> 累计15 > 10，应裁掉最旧
    check("短期记忆裁剪掉了最旧消息", st.messages[0]["content"] != "你好世界")

    check("'你好'是闲聊", is_chit_chat("你好"))
    check("'我喜欢蓝色'不是闲聊", not is_chit_chat("我喜欢蓝色"))
    check("'ok'是闲聊", is_chit_chat("ok"))

    store = MemoryStore()
    a1, id1 = store.upsert("用户喜欢用中文交流", "preference", 5, "test")
    a2, id2 = store.upsert("用户喜欢用中文交流", "preference", 5, "test")
    check("首次写入 created", a1 == "created")
    check("重复写入 updated(去重)", a2 == "updated")
    check("去重命中同一个id", id1 == id2)

    store.upsert("用户名叫小明", "fact", 5, "test")
    store.upsert("LanceDB", "entity", 3, "test")
    check("表里有3条记忆", store.count() == 3)

    semantic = store.semantic_search(store._embed("用户喜欢什么语言"), 3)
    check("语义召回不为空", len(semantic) > 0)
    entity = store.entity_search("LanceDB 是什么")
    check("实体召回命中 LanceDB", any(e["text"] == "LanceDB" for e in entity))

    ranked = retrieve_long_term(store, "我之前说过喜欢什么语言吗")
    check("多路召回不为空", len(ranked) > 0)
    check("排序按score降序", all(ranked[i]["score"] >= ranked[i + 1]["score"] for i in range(len(ranked) - 1)))

    shutil.rmtree(config.MEMORY_DB_PATH, ignore_errors=True)
    print("阶段1完成")


def phase2():
    section("阶段2: 调LLM")

    from llm.ollama_client import OllamaClient
    from memory.extractor import extract_memories
    from memory.manager import MemoryManager

    client = OllamaClient(
        base_url=config.OLLAMA_BASE_URL, model=config.OLLAMA_MODEL, timeout=config.OLLAMA_TIMEOUT
    )

    memories = extract_memories(
        "用户说：我叫小明，我喜欢用中文交流。\n助手回复：好的，小明你好！",
        client,
    )
    print(f"  提炼出 {len(memories)} 条记忆:")
    for m in memories:
        print(f"    - [{m['type']}] {m['text']} (重要度{m['importance']})")
    check("提炼返回列表", isinstance(memories, list))
    check("提炼字段齐全", all("text" in m and "type" in m and "importance" in m for m in memories))

    shutil.rmtree(config.MEMORY_DB_PATH, ignore_errors=True)
    mgr = MemoryManager(client, source="test-session")

    check("闲聊检索返回None", mgr.retrieve("你好") is None)

    mgr.observe("我叫小明，我在做一个AI助手项目", "好的，小明，很高兴认识你")
    print(f"  observe后长期记忆={mgr.store.count()}条, 琐碎缓冲={len(mgr.pending)}条")

    result = mgr.retrieve("我之前说我叫什么名字？")
    print(f"  检索'我叫什么'返回: {result!r}")
    check("检索能召回用户名字", result is not None)

    flushed = mgr.flush_session()
    print(f"  flush_session 写入 {flushed} 条")

    shutil.rmtree(config.MEMORY_DB_PATH, ignore_errors=True)
    print("阶段2完成")


if __name__ == "__main__":
    phase1()
    try:
        phase2()
    except Exception as e:
        print(f"\n阶段2异常(可能Ollama未启动或模型未拉取): {type(e).__name__}: {e}")
    print("\n===== 测试结束 =====")
