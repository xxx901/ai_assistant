"""
file_ops 引擎单测（不开 Ollama）：扫描/分类/去重/安全移动/重命名/回收站/撤销。

运行：在项目根目录 `python test/test_file_ops.py`
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.file_ops as fo


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    print("===== file_ops 引擎测试 =====")

    # 隔离：临时工作目录 + 独立 journal/trash，不污染真实数据
    workdir = tempfile.mkdtemp(prefix="fileops_test_")
    fo.JOURNAL_PATH = os.path.join(workdir, "journal.json")
    fo.TRASH_DIR = os.path.join(workdir, "trash")

    src_dir = os.path.join(workdir, "src")
    os.makedirs(src_dir)
    open(os.path.join(src_dir, "a.txt"), "w", encoding="utf-8").write("hello world")
    open(os.path.join(src_dir, "b.pdf"), "wb").write(b"%PDF fake")
    open(os.path.join(src_dir, "dup1.txt"), "w", encoding="utf-8").write("same content")
    open(os.path.join(src_dir, "dup2.txt"), "w", encoding="utf-8").write("same content")
    open(os.path.join(src_dir, "c.py"), "w", encoding="utf-8").write("print(1)")

    # 扫描
    entries = fo.scan_directory(src_dir)
    check("scan 返回5个文件", len(entries) == 5)
    check("scan 带 category/mtime 元数据", all("category" in e and "mtime" in e for e in entries))

    # 按类型分类
    cls = fo.classify_by_type(src_dir)
    check("txt 归入文档", "a.txt" in cls.get("文档", []))
    check("py 归入代码", "c.py" in cls.get("代码", []))

    # 去重（内容哈希）
    dups = fo.find_duplicates(src_dir)
    check("找到1组重复", len(dups) == 1)
    check("重复组含2个文件", dups and len(dups[0]) == 2)

    # 安全移动：目标已存在 → 加后缀不覆盖
    dst_dir = os.path.join(workdir, "dst")
    os.makedirs(dst_dir)
    open(os.path.join(dst_dir, "a.txt"), "w").write("existing")
    moved = fo.move_file(os.path.join(src_dir, "a.txt"), dst_dir)
    check("移动后加后缀避免覆盖", os.path.basename(moved) == "a (1).txt")
    check("原目标文件未被覆盖", open(os.path.join(dst_dir, "a.txt")).read() == "existing")

    # 重命名
    renamed = fo.rename_file(os.path.join(src_dir, "c.py"), "c_renamed.py")
    check("重命名成功", os.path.basename(renamed) == "c_renamed.py")

    # 回收站（不永久删除）
    trashed = fo.trash_file(os.path.join(src_dir, "b.pdf"))
    check("trash 后原文件消失", not os.path.exists(os.path.join(src_dir, "b.pdf")))
    check("trash 文件进了回收站", os.path.exists(trashed))

    # 撤销最近3次（trash b.pdf / rename c.py / move a.txt，逆序回退）
    results = fo.undo(3)
    check("undo 回退3条", len(results) == 3)
    check("a.txt 回到 src", os.path.exists(os.path.join(src_dir, "a.txt")))
    check("c.py 名字恢复", os.path.exists(os.path.join(src_dir, "c.py")))
    check("b.pdf 回到 src", os.path.exists(os.path.join(src_dir, "b.pdf")))

    shutil.rmtree(workdir, ignore_errors=True)
    print("测试结束")


if __name__ == "__main__":
    main()
