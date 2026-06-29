"""
RAG 检索质量自动化评测。

指标：
- Hit Rate@K: 前 K 个结果中是否命中目标文档
- MRR (Mean Reciprocal Rank): 目标文档首次出现的倒数排名

用法:
    python tests/evaluation/rag_eval.py
    python tests/evaluation/rag_eval.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_store import RAGStore, RAGResult
from config import AppConfig


def load_cases(path: str) -> list[dict[str, Any]]:
    """从 JSONL 文件加载评测用例（仅 rag 类）。"""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if case.get("category") == "rag":
                cases.append(case)
    return cases


def hit_rate(results: list[RAGResult], target_doc: str, top_k: int = 3) -> bool:
    """检查前 top_k 个结果是否命中目标文档。"""
    docs = results[:top_k]
    return any(target_doc in (r.metadata.get("source", "") or "") for r in docs)


def reciprocal_rank(results: list[RAGResult], target_doc: str, top_k: int = 10) -> float:
    """计算目标文档的 Reciprocal Rank（0 表示未命中）。"""
    for i, r in enumerate(results[:top_k], 1):
        if target_doc in (r.metadata.get("source", "") or ""):
            return 1.0 / i
    return 0.0


def evaluate(rag: RAGStore, cases_path: str, top_k: int = 3) -> dict[str, Any]:
    """运行评测并返回指标报告。"""
    cases = load_cases(cases_path)
    if not cases:
        return {"error": "无 RAG 类评测用例", "total": 0}

    hits = 0
    mrr_sum = 0.0
    details: list[dict] = []

    for case in cases:
        case_id = case["id"]
        query = case["input"]
        target_docs = case.get("context", {}).get("rag_docs", [])
        if not target_docs:
            continue

        results = rag.search(query)

        hit = False
        rr = 0.0
        target_doc = target_docs[0]  # 取第一个目标文档

        hit = hit_rate(results, target_doc, top_k)
        rr = reciprocal_rank(results, target_doc, top_k)

        if hit:
            hits += 1
        mrr_sum += rr

        details.append({
            "id": case_id,
            "query": query,
            "target": target_doc,
            "hit": hit,
            "rr": round(rr, 4),
            "top_docs": [
                {
                    "source": (r.metadata.get("source") or "")[-60:],
                    "distance": round(r.distance, 4) if r.distance else None,
                }
                for r in results[:top_k]
            ],
        })

    total = len(cases)
    return {
        "total_cases": total,
        "hit_rate": round(hits / total, 4) if total else 0,
        "mrr": round(mrr_sum / total, 4) if total else 0,
        "hits": hits,
        "top_k": top_k,
        "details": details,
    }


def print_report(report: dict[str, Any]) -> None:
    """打印评测报告。"""
    print("=" * 60)
    print("RAG 检索质量评测报告")
    print("=" * 60)
    print(f"  用例总数: {report['total_cases']}")
    print(f"  Top-K:    {report['top_k']}")
    print(f"  Hit Rate: {report['hit_rate']:.2%} ({report['hits']}/{report['total_cases']})")
    print(f"  MRR:      {report['mrr']:.4f}")
    print("-" * 60)

    for d in report.get("details", []):
        status = "✅" if d["hit"] else "❌"
        print(f"  {status} {d['id']}: \"{d['query'][:50]}\"")
        print(f"     target: {d['target']}  RR: {d['rr']}")
        if not d["hit"]:
            print(f"     top results: {[t['source'] for t in d['top_docs']]}")

    print("-" * 60)
    verdict = "✅ 通过" if report["hit_rate"] >= 0.6 else "⚠️  需改进"
    print(f"  结论: {verdict} (阈值: Hit Rate >= 60%)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--cases", default="tests/evaluation/cases_template.jsonl", help="评测用例文件")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K 参数")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    config = AppConfig.from_env()
    rag = RAGStore(config)

    report = evaluate(rag, args.cases, top_k=args.top_k)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    # 返回码: Hit Rate < 0.6 为失败
    if report.get("hit_rate", 0) < 0.6:
        sys.exit(1)


if __name__ == "__main__":
    main()
