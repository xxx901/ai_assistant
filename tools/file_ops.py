"""
文件操作引擎：扫描/分类/去重/安全移动/重命名/回收站/日志/撤销。

这是"引擎"，不含LLM（按主题分类是唯一例外，通过 set_llm_client 注入客户端）。

安全设计（生产级重点）：
- 绝不覆盖：目标已存在时自动加后缀 " (1)" " (2)" ...，而不是静默覆盖。
- 绝不永久删除："删除"一律移到回收站(TRASH_DIR)，可撤销。
- 每个副作用操作都写日志(file_journal.json)，undo() 能逆序回退。

复用：无外部依赖，只用到 os/shutil/hashlib/json/uuid/time。
"""

import hashlib
import json
import os
import shutil
import time
import uuid

TRASH_DIR = "./data/trash"
JOURNAL_PATH = "./data/file_journal.json"

# 扩展名 -> 类别（按类型分类，确定性）
EXT_CATEGORIES = {
    "文档": {".doc", ".docx", ".pdf", ".txt", ".md", ".rtf", ".xls", ".xlsx", ".ppt", ".pptx", ".csv"},
    "图片": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico"},
    "视频": {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"},
    "音频": {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"},
    "压缩包": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "代码": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".rs", ".go", ".html", ".css", ".json", ".yaml", ".yml", ".sh"},
}
DEFAULT_CATEGORY = "其他"

# 按主题分类要用的LLM客户端，由 set_llm_client 注入（唯一非纯函数能力）
_client = None


def set_llm_client(client):
    """给 classify_by_theme 注入 LLM 客户端。应用启动时调用一次。"""
    global _client
    _client = client


# ---- 扫描 ----

def _category_of(ext: str) -> str:
    for category, exts in EXT_CATEGORIES.items():
        if ext in exts:
            return category
    return DEFAULT_CATEGORY


def _make_entry(path: str) -> dict:
    stat = os.stat(path)
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    return {
        "name": name,
        "path": path,
        "ext": ext,
        "category": _category_of(ext),
        "size": stat.st_size,
        "mtime": time.strftime("%Y-%m-%d", time.localtime(stat.st_mtime)),
    }


def scan_directory(path: str, recursive: bool = False, extensions=None) -> list:
    """列出目录下的文件（含元数据）。extensions 如 ["pdf", ".txt"] 只保留这些。"""
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"路径不存在或不是文件夹：{path}")

    if recursive:
        files = []
        for root, _dirs, filenames in os.walk(path):
            for f in filenames:
                files.append(os.path.join(root, f))
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))]

    entries = [_make_entry(f) for f in files]

    if extensions:
        ext_set = {e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}
        entries = [e for e in entries if e["ext"] in ext_set]
    return entries


# ---- 分类（按类型） ----

def classify_by_type(path: str) -> dict:
    """按扩展名分类，返回 {类别: [文件名, ...]}。"""
    entries = scan_directory(path)
    result = {}
    for e in entries:
        result.setdefault(e["category"], []).append(e["name"])
    return result


# ---- 分类（按主题，LLM） ----

_THEME_SYSTEM = """你是一个文件分类器。根据文件名（和文档内容预览），把这些文件分成几个语义主题组。

要求：
1. 每个文件只归入一个主题；主题名要简短具体（如"机器学习论文"、"财务报表"、"旅游照片"）。
2. 只输出一个JSON，格式严格如下，不要输出其他文字、不要加解释、不要用markdown代码块包裹：
{"themes": [{"theme": "主题名", "files": ["文件名1", "文件名2"]}, {"theme": "主题名2", "files": ["文件名3"]}]}
"""


def _content_preview(path: str, max_chars: int = 120) -> str:
    """读文档开头做预览；读不了(二进制/加密)返回空串。"""
    try:
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "".join((p.extract_text() or "") for p in reader.pages[:1])
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(max_chars * 2)
        return text[:max_chars].replace("\n", " ").strip()
    except Exception:
        return ""


def classify_by_theme(path: str) -> dict:
    """用LLM按语义主题分类，返回 {主题: [文件名, ...]}。文档带内容预览，其余只看文件名。"""
    if _client is None:
        raise RuntimeError("classify_by_theme 需要先 set_llm_client 注入LLM客户端")

    entries = scan_directory(path)
    if not entries:
        return {}

    lines = []
    for e in entries:
        preview = _content_preview(e["path"]) if e["category"] == "文档" else ""
        lines.append(f"- {e['name']}" + (f" | 预览: {preview}" if preview else ""))
    file_list = "\n".join(lines)

    raw = _client.chat(
        messages=[{"role": "user", "content": f"文件列表：\n{file_list}\n\n请把这些文件按主题分组。"}],
        stream=False,
        system=_THEME_SYSTEM,
        format_json=True,
    )

    try:
        data = json.loads(raw)
        themes = data.get("themes", [])
    except json.JSONDecodeError:
        return {"未分类": [e["name"] for e in entries]}

    name_set = {e["name"] for e in entries}
    result = {}
    assigned = set()
    for t in themes:
        if not isinstance(t, dict):
            continue
        theme = str(t.get("theme", "未分类")).strip() or "未分类"
        for fname in t.get("files", []):
            fname = str(fname).strip()
            if fname in name_set and fname not in assigned:
                result.setdefault(theme, []).append(fname)
                assigned.add(fname)

    unassigned = [e["name"] for e in entries if e["name"] not in assigned]
    if unassigned:
        result["未分类"] = unassigned
    return result


# ---- 去重（按内容哈希） ----

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(path: str, recursive: bool = False) -> list:
    """找出内容完全相同的重复文件，返回 [[路径, ...], ...]。先按大小粗筛，再对同大小组算哈希。"""
    entries = scan_directory(path, recursive=recursive)
    by_size = {}
    for e in entries:
        by_size.setdefault(e["size"], []).append(e["path"])

    groups = []
    for _size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_hash = {}
        for p in paths:
            by_hash.setdefault(_sha256(p), []).append(p)
        for dup_paths in by_hash.values():
            if len(dup_paths) >= 2:
                groups.append(sorted(dup_paths))
    return groups


# ---- 安全移动 / 重命名 / 回收站 ----

def _unique_path(dst: str) -> str:
    """目标已存在时，自动加后缀 " (1)" " (2)" ...，保证绝不覆盖。"""
    if not os.path.exists(dst):
        return dst
    base, ext = os.path.splitext(dst)
    i = 1
    while True:
        candidate = f"{base} ({i}){ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def move_file(src: str, dst_dir: str) -> str:
    """把 src 移到 dst_dir 下，目标重名则加后缀。写日志，返回最终路径。"""
    src = os.path.normpath(src)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"文件不存在：{src}")
    os.makedirs(os.path.normpath(dst_dir), exist_ok=True)
    dst = _unique_path(os.path.join(os.path.normpath(dst_dir), os.path.basename(src)))
    shutil.move(src, dst)
    _journal("move", src, dst)
    return dst


def rename_file(src: str, new_name: str) -> str:
    """把 src 重命名为 new_name（同一文件夹内），重名则加后缀。写日志，返回最终路径。"""
    src = os.path.normpath(src)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"文件不存在：{src}")
    dst = _unique_path(os.path.join(os.path.dirname(src), new_name))
    shutil.move(src, dst)
    _journal("rename", src, dst)
    return dst


def trash_file(path: str) -> str:
    """把文件移入回收站(不永久删除)。写日志，返回回收站里的路径。"""
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    os.makedirs(TRASH_DIR, exist_ok=True)
    dst = _unique_path(os.path.join(TRASH_DIR, os.path.basename(path)))
    shutil.move(path, dst)
    _journal("trash", path, dst)
    return dst


# ---- 日志 + 撤销 ----

def _load_journal() -> list:
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_journal(journal: list):
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)


def _journal(action: str, src: str, dst: str):
    journal = _load_journal()
    journal.append({
        "id": str(uuid.uuid4()),
        "action": action,   # move / rename / trash
        "src": src,
        "dst": dst,
        "ts": time.time(),
        "undone": False,
    })
    _save_journal(journal)


def undo(n: int = 1) -> list:
    """逆序回退最近 n 条未撤销的操作，返回每条的回退结果。绝不覆盖已有文件。"""
    journal = _load_journal()
    pending = [e for e in journal if not e["undone"]][-n:]

    results = []
    for entry in reversed(pending):
        entry["undone"] = True
        src, dst = entry["src"], entry["dst"]
        if os.path.exists(dst) and not os.path.exists(src):
            shutil.move(dst, src)
            results.append({"action": entry["action"], "restored": src, "ok": True})
        elif os.path.exists(dst) and os.path.exists(src):
            results.append({"action": entry["action"], "restored": src,
                            "ok": False, "reason": "原位置已被占用，未覆盖"})
        else:
            results.append({"action": entry["action"], "restored": src,
                            "ok": False, "reason": "目标文件已不在回收站/新位置"})

    _save_journal(journal)
    return results
