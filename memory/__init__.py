"""
Memory 包：生产级用户画像记忆系统。

对外只暴露两个东西，上层(main.py)只依赖这个门面，不碰内部模块：
- MemoryManager: 读写记忆的编排器
- estimate_tokens: 短期记忆裁剪用的token估算

内部模块分工：
- short_term: 短期记忆(按token裁剪的对话列表)
- store:      长期记忆(LanceDB表 + 相似去重)
- extractor:  LLM结构化提炼(输出type/importance/text)
- retriever:  长期多路召回(语义/实体/最近) + 排序
- manager:    编排(写时机、去重、琐碎缓冲区)
"""

from memory.manager import MemoryManager, is_chit_chat
from memory.short_term import estimate_tokens

__all__ = ["MemoryManager", "is_chit_chat", "estimate_tokens"]
