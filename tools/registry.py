"""
工具注册表：把每个工具包装成"给LLM看的说明书" + "真正会被执行的函数"。

设计原则：
1. 用字典存储（工具名 -> 工具信息），支持O(1)按名字查找，
   跟意图路由里"用set不用list判断成员"是同一个道理。
2. has_side_effect标记有没有副作用，这一层本身不做任何拦截，
   只是为模块5判断"要不要走预览确认"提供依据。
3. registry里必须存真正的函数引用（func字段），
   工具调用的本质就是把"字符串形式的工具名"翻译回"可执行的代码"。
"""

import os


def list_files(path: str) -> list:
    """列出指定文件夹下的所有文件名（不含子文件夹递归）。只读操作，无副作用。"""
    path = os.path.normpath(path)  # 规整化路径，兜住模型可能输出的重复反斜杠等问题
    if not os.path.isdir(path):
        raise FileNotFoundError(f"路径不存在或不是文件夹：{path}")
    return os.listdir(path)


def rename_file(old_path: str, new_name: str) -> str:
    """把old_path这个文件重命名为new_name（只改文件名，不改所在文件夹）。有副作用。"""
    old_path = os.path.normpath(old_path)
    folder = os.path.dirname(old_path)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    return new_path


TOOL_REGISTRY = {
    "list_files": {
        "description": "列出指定文件夹下的所有文件名，用于查看某个目录里有哪些文件。",
        "parameters": {
            "path": {"type": "string", "required": True, "description": "要查看的文件夹路径"},
        },
        "has_side_effect": False,
        "func": list_files,
    },
    "rename_file": {
        "description": "重命名指定路径下的文件，用于重新命名文件。",
        "parameters": {
            "old_path": {"type": "string", "required": True, "description": "要重命名的文件的完整路径"},
            "new_name": {"type": "string", "required": True, "description": "新的文件名（不含文件夹路径）"},
        },
        "has_side_effect": True,
        "func": rename_file,
    },
    # 以后还可以继续加 dedupe_files 等工具
}