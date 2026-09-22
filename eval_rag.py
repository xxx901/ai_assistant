"""
RAG 问答正确率评测脚本。

流程：
1. 解析 eval/qa.txt 里的全部题目（每道题 = 问题 + 参考答案）。
2. 把 eval/ 下的 PDF 文档索引进本地向量库（indexer 已支持 PDF）。
   --pdf 指定具体文件；--translate 决定是否先做"中文问题翻英文"的跨语言检索。
3. 逐题：检索 -> 用检索到的上下文让 LLM 生成中文答案 -> 用 LLM 当裁判，
   对照参考答案判断"正确 / 错误"。
4. 汇总输出正确答案率，逐题结果另存到 eval/eval_result.json 方便人工复核。

运行方式（在项目根目录）：
    python eval_rag.py --pdf "Attention Is All You Need 中文翻译.pdf"        # 中文文档
    python eval_rag.py --pdf "attention is all you need.pdf" --translate     # 英文文档(跨语言)
    python eval_rag.py --pdf "..." --limit 3    # 只跑前 3 题，先做冒烟测试

注意：
- 首次运行要下载 bge-m3（嵌入）和 bge-reranker-base（精排）两个模型，
  共约 3GB，请确保网络通畅、磁盘空间足够。
- 需要本地 Ollama 服务已启动，且已拉取 config.OLLAMA_MODEL。
- 判定对错用"LLM 当裁判"，qwen2.5:7b 当裁判并非绝对准确，
  最终正确率是参考值，建议对个别题人工复核。
"""

import argparse
import json
import os
import re
import shutil

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    LANCEDB_PATH,
    PARENT_STORE_PATH,
)
from llm.ollama_client import OllamaClient, OllamaClientError, is_ollama_running
from rag.indexer import index_document
from rag.retriever import retrieve

EVAL_DIR = "eval"
QA_PATH = os.path.join(EVAL_DIR, "qa.txt")
RESULT_PATH = os.path.join(EVAL_DIR, "eval_result.json")


def parse_qa(path: str) -> list[dict]:
    """解析 qa.txt。格式约定：`**N. 问题**` 占一行，紧接着的下一行是参考答案。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    qa = []
    i = 0
    while i < len(lines):
        m = re.match(r"^\*\*(\d+)\.\s*(.+?)\*\*\s*$", lines[i])
        if m:
            # 题目行下面的第一个非空行就是参考答案
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            answer = lines[j].strip() if j < len(lines) else ""
            qa.append({
                "num": int(m.group(1)),
                "question": m.group(2).strip(),
                "answer": answer,
            })
            i = j + 1
        else:
            i += 1
    return qa


def fresh_index(eval_dir: str, pdf_name: str | None = None):
    """清掉旧索引，重新索引 eval 目录下的 PDF，保证评测结果是可复现的。

    pdf_name 指定时只索引这一个文件；否则索引目录下全部 PDF。
    清掉的是 data/lancedb 和 data/parents.json 这两份"生成产物"，
    它们随时能从源 PDF 重建，删了不会丢信息。
    """
    if os.path.exists(LANCEDB_PATH):
        shutil.rmtree(LANCEDB_PATH)
    if os.path.exists(PARENT_STORE_PATH):
        os.remove(PARENT_STORE_PATH)

    if pdf_name:
        pdfs = [pdf_name]
    else:
        pdfs = [f for f in os.listdir(eval_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise SystemExit(f"在 {eval_dir} 目录下没找到 PDF 文档")
    for name in pdfs:
        path = os.path.join(eval_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"找不到 PDF 文件：{path}")
        print(f"[索引] {name}")
        index_document(path)


TRANSLATE_SYSTEM = (
    "把用户的中文问题翻译成英文。只输出英文译文本身，不要任何解释、引号或额外文字。"
)


def translate_to_english(question: str, client: OllamaClient) -> str:
    """中文问题 -> 英文，供跨语言检索用。

    知识库是英文论文、问题是中文，直接拿中文去检索时，稀疏检索零词法重叠、
    reranker 跨语言打分近零，两路都白费了。所以先把问题翻成英文再检索。
    """
    return client.chat(
        messages=[{"role": "user", "content": question}],
        stream=False,
        system=TRANSLATE_SYSTEM,
    ).strip()


def generate_answer(question: str, client: OllamaClient, translate: bool = False) -> tuple[str, dict, str]:
    """RAG 生成，返回 (答案, 检索结果, 实际检索query)。

    translate=True 时先把中文问题翻成英文再检索（知识库是英文文档时用）；
    知识库是中文文档时用 translate=False，直接用原问题检索。
    生成答案始终用原中文问题、明确要求用中文回答，好跟参考答案对齐。
    """
    query = translate_to_english(question, client) if translate else question
    result = retrieve(query)
    if not result["found"]:
        answer = client.chat(messages=[{"role": "user", "content": question}], stream=False)
        return answer, result, query

    context_text = "\n\n".join(
        f"[来源: {c['source_file']}]\n{c['text']}" for c in result["contexts"]
    )
    system = "请结合以下参考资料，用中文回答用户问题，如果资料不足以回答，请如实说明：\n\n" + context_text
    answer = client.chat(
        messages=[{"role": "user", "content": question}], stream=False, system=system
    )
    return answer, result, query


JUDGE_SYSTEM = (
    "你是一个评分员。判断\"生成答案\"是否包含了问题的核心答案"
    "（与参考答案语义一致即可，不要求逐字相同；无关的额外内容不影响判定）。"
    "只输出一个词：正确 或 错误。"
)


def judge(question: str, reference: str, generated: str, client: OllamaClient) -> str:
    """LLM 当裁判，返回裁判原始输出（预期是"正确"或"错误"）。"""
    prompt = (
        f"问题：{question}\n"
        f"参考答案：{reference}\n"
        f"生成答案：{generated}\n"
        f"生成答案是否正确？只输出：正确 或 错误"
    )
    return client.chat(
        messages=[{"role": "user", "content": prompt}], stream=False, system=JUDGE_SYSTEM
    )


def parse_verdict(raw: str) -> bool | None:
    """把裁判输出解析成 True/False/None(解析不了)。

    注意"不正确"里也含"正确"两个字，所以必须先判断否定词，否则会被误判为对。
    """
    text = raw.strip().replace("。", "").replace(" ", "")
    if "不正确" in text or "不对" in text:
        return False
    if "错误" in text and "正确" not in text:
        return False
    if "正确" in text:
        return True
    return None


def main():
    parser = argparse.ArgumentParser(description="RAG 问答正确率评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟测试用）")
    parser.add_argument("--pdf", type=str, default=None, help="只索引 eval 目录下指定的那个 PDF（默认索引全部）")
    parser.add_argument("--translate", action="store_true", help="检索前把中文问题翻成英文（知识库是英文文档时用）")
    args = parser.parse_args()

    if not is_ollama_running(OLLAMA_BASE_URL):
        raise SystemExit(f"[失败] 连不上 Ollama 服务，请先 `ollama serve` 并确认已拉取 {OLLAMA_MODEL}")

    client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT)

    qa = parse_qa(QA_PATH)
    print(f"[解析] 共 {len(qa)} 道题\n")

    print("[索引] 清空旧索引并重新索引 PDF...")
    fresh_index(EVAL_DIR, args.pdf)
    print("[索引] 完成\n")

    if args.limit:
        qa = qa[: args.limit]
        print(f"[注意] --limit={args.limit}，只跑前 {len(qa)} 题\n")

    correct = wrong = judge_failed = error = 0
    found_hits = 0
    rows = []

    for idx, item in enumerate(qa, 1):
        q, ref = item["question"], item["answer"]
        print(f"[{idx}/{len(qa)}] {q}")

        try:
            generated, result, query = generate_answer(q, client, translate=args.translate)
            if result["found"]:
                found_hits += 1
            judge_raw = judge(q, ref, generated, client)
            verdict = parse_verdict(judge_raw)
        except OllamaClientError as e:
            error += 1
            rows.append({"num": item["num"], "question": q, "reference": ref,
                         "retrieval_query": None, "generated": None,
                         "verdict": "异常", "judge_raw": str(e)})
            print(f"    [异常] {e}\n")
            continue
        except Exception as e:
            # 兜底：单题意外报错（检索/嵌入/序列化等）不能中断整场评测，
            # 记录后继续下一题，跟上面 OllamaClientError 是同一层"边界兜底"。
            error += 1
            rows.append({"num": item["num"], "question": q, "reference": ref,
                         "retrieval_query": None, "generated": None,
                         "verdict": "异常", "judge_raw": f"意外错误: {e}"})
            print(f"    [意外错误] {e}\n")
            continue

        if verdict is True:
            correct += 1
            tag = "正确"
        elif verdict is False:
            wrong += 1
            tag = "错误"
        else:
            judge_failed += 1
            tag = "判定失败"

        rows.append({"num": item["num"], "question": q, "reference": ref,
                     "retrieval_query": query, "generated": generated,
                     "verdict": tag, "judge_raw": judge_raw})
        print(f"    {tag} | 裁判原文: {judge_raw}\n")

    total = len(qa)
    judged = correct + wrong
    accuracy_judged = correct / judged if judged else 0.0
    accuracy_overall = correct / total if total else 0.0

    print("=" * 56)
    print(f"总题数: {total}")
    print(f"检索命中: {found_hits} / {total}")
    print(f"正确: {correct}   错误: {wrong}   裁判判定失败: {judge_failed}   执行异常: {error}")
    print(f"正确答案率（正确/总题数）: {correct}/{total} = {accuracy_overall:.2%}")
    print(f"正确答案率（仅计有效判定）: {correct}/{judged} = {accuracy_judged:.2%}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy_overall": accuracy_overall,
            "accuracy_judged": accuracy_judged,
            "correct": correct, "wrong": wrong,
            "judge_failed": judge_failed, "error": error,
            "found_hits": found_hits, "total": total,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"逐题结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
