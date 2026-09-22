"""
工具注册表：把每个工具包装成"给LLM看的说明书" + "真正会被执行的函数"。

设计原则：
1. 用字典存储（工具名 -> 工具信息），支持O(1)按名字查找。
2. has_side_effect标记有没有副作用，这一层不做拦截，只供agent_loop决定要不要确认。
3. registry里存真正的函数引用(func字段)，执行时由tool_caller按名字调起。
4. 文件操作统一走 tools/file_ops.py（安全：不覆盖、不永久删除、写日志可撤销）。
"""

from tools import file_ops


TOOL_REGISTRY = {
    "list_files": {
        "description": "列出指定文件夹下的文件（含大小、修改日期、类别）。用于先扫描看目录里有什么。",
        "parameters": {
            "path": {"type": "string", "required": True, "description": "要查看的文件夹路径"},
            "recursive": {"type": "boolean", "required": False, "description": "是否递归子目录，默认否"},
            "extensions": {"type": "array", "required": False, "description": "只保留这些扩展名，如 [\"pdf\", \"txt\"]"},
        },
        "has_side_effect": False,
        "func": file_ops.scan_directory,
    },
    "classify_files": {
        "description": "按文件类型（扩展名）把文件夹里的文件分成 文档/图片/视频/音频/压缩包/代码/其他 几类。",
        "parameters": {
            "path": {"type": "string", "required": True, "description": "要分类的文件夹路径"},
        },
        "has_side_effect": False,
        "func": file_ops.classify_by_type,
    },
    "classify_by_theme": {
        "description": "用AI按语义主题把文件夹里的文件分组（如'机器学习论文''财务报表'）。用于'按主题分类'。",
        "parameters": {
            "path": {"type": "string", "required": True, "description": "要按主题分类的文件夹路径"},
        },
        "has_side_effect": False,
        "func": file_ops.classify_by_theme,
    },
    "find_duplicates": {
        "description": "找出文件夹里内容完全相同的重复文件，返回重复文件组。用于去重前的排查。",
        "parameters": {
            "path": {"type": "string", "required": True, "description": "要查重的文件夹路径"},
            "recursive": {"type": "boolean", "required": False, "description": "是否递归子目录，默认否"},
        },
        "has_side_effect": False,
        "func": file_ops.find_duplicates,
    },
    "move_file": {
        "description": "把某个文件移动到目标文件夹（目标重名会自动加后缀，不会覆盖）。有副作用。",
        "parameters": {
            "src": {"type": "string", "required": True, "description": "要移动的文件的完整路径"},
            "dst_dir": {"type": "string", "required": True, "description": "目标文件夹路径"},
        },
        "has_side_effect": True,
        "func": file_ops.move_file,
    },
    "rename_file": {
        "description": "重命名某个文件（只改文件名，不改所在文件夹；重名自动加后缀）。有副作用。",
        "parameters": {
            "src": {"type": "string", "required": True, "description": "要重命名的文件的完整路径"},
            "new_name": {"type": "string", "required": True, "description": "新的文件名（不含文件夹路径）"},
        },
        "has_side_effect": True,
        "func": file_ops.rename_file,
    },
    "trash_file": {
        "description": "把某个文件移入回收站（不是永久删除，可撤销）。用于删除重复文件等。有副作用。",
        "parameters": {
            "path": {"type": "string", "required": True, "description": "要移入回收站的文件完整路径"},
        },
        "has_side_effect": True,
        "func": file_ops.trash_file,
    },
    "undo": {
        "description": "撤销最近 n 次文件操作（移动/重命名/删除），把文件恢复到操作前的位置。",
        "parameters": {
            "n": {"type": "integer", "required": False, "description": "要撤销的操作次数，默认1"},
        },
        "has_side_effect": True,
        "func": file_ops.undo,
    },
}
