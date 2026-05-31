from __future__ import annotations

import argparse

from config import AppConfig
from rag_store import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 SZ-HK Hub RAG 向量库")
    parser.add_argument("--reset", action="store_true", help="重建索引并清空旧数据")
    args = parser.parse_args()
    config = AppConfig.from_env()
    count = ingest_corpus(config, reset=args.reset)
    print(f"已写入 {count} 段文本到向量库。")


if __name__ == "__main__":
    main()
