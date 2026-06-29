"""
RAG 动态语料爬取模块。

从指定 URL 抓取文本内容，清洗后保存为 Markdown 文件到 rag_corpus/ 目录。
适用于定期更新香港政府政策、银行 FAQ、港铁公告等动态内容。

用法:
    python rag_crawler.py                    # 爬取预设源
    python rag_crawler.py --url <URL>        # 爬取单个 URL
    python rag_crawler.py --output <name>    # 指定输出文件名
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── 预设爬取源 ────────────────────────────────────────────
# 每个条目: (名称, URL, CSS选择器)
PRESET_SOURCES: list[tuple[str, str, Optional[str]]] = [
    # 香港海关 - 旅客清关
    (
        "customs-passenger-clearance",
        "https://www.customs.gov.hk/en/service-enforcement-information/passenger-clearance/index.html",
        "article, .main-content, #middle-wrap",
    ),
    # 香港入境处 - 常见问题
    (
        "immd-faq",
        "https://www.immd.gov.hk/eng/faq/",
        "article, .main-content, #content",
    ),
    # MTR 港铁 - 游客信息
    (
        "mtr-tourist",
        "https://www.mtr.com.hk/en/customer/tourist/index.php",
        "article, .content-area, #content",
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

CORPUS_DIR = Path(__file__).parent / "rag_corpus"


def fetch_page(url: str, timeout: int = 15) -> str:
    """获取页面 HTML。"""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def extract_text(html: str, selector: Optional[str] = None) -> str:
    """从 HTML 提取正文。"""
    soup = BeautifulSoup(html, "html.parser")

    # 移除 script / style
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    if selector:
        container = soup.select_one(selector)
        if container:
            return _clean_text(container.get_text(" ", strip=True))

    # 回退：取 body
    body = soup.find("body")
    if body:
        return _clean_text(body.get_text(" ", strip=True))
    return ""


def _clean_text(text: str) -> str:
    """清洗文本：去多余空白、截断过长行。"""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            # 截断超长行
            if len(stripped) > 500:
                stripped = stripped[:500] + "…"
            lines.append(stripped)
    return "\n\n".join(lines[:200])  # 最多 200 行


def save_markdown(content: str, name: str, source_url: str) -> Path:
    """保存为 Markdown 文件。"""
    domain = urlparse(source_url).netloc
    title = name.replace("-", " ").title()
    md = f"# {title}\n\n> 爬取自: {source_url}\n> 域名: {domain}\n> 更新时间: {time.strftime('%Y-%m-%d %H:%M')}\n\n{content}\n"
    path = CORPUS_DIR / f"{name}.md"
    path.write_text(md, encoding="utf-8")
    return path


def crawl_presets() -> list[Path]:
    """爬取所有预设源。"""
    saved: list[Path] = []
    for name, url, selector in PRESET_SOURCES:
        try:
            print(f"⬇️  爬取: {name} ({url})")
            html = fetch_page(url)
            text = extract_text(html, selector)
            if len(text) < 100:
                print(f"   ⚠️  提取内容过短 ({len(text)} 字符)，跳过")
                continue
            path = save_markdown(text, name, url)
            print(f"   ✅ 保存: {path} ({len(text)} 字符)")
            saved.append(path)
            time.sleep(1)  # 礼貌间隔
        except Exception as exc:
            print(f"   ❌ 失败: {exc}")
    return saved


def crawl_single(url: str, name: Optional[str] = None) -> Optional[Path]:
    """爬取单个 URL。"""
    if name is None:
        name = urlparse(url).netloc.replace(".", "-")
    print(f"⬇️  爬取: {name} ({url})")
    try:
        html = fetch_page(url)
        text = extract_text(html)
        if len(text) < 50:
            print(f"   ⚠️  内容过短，跳过")
            return None
        path = save_markdown(text, name, url)
        print(f"   ✅ 保存: {path} ({len(text)} 字符)")
        return path
    except Exception as exc:
        print(f"   ❌ 失败: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 动态语料爬取")
    parser.add_argument("--url", help="爬取单个 URL")
    parser.add_argument("--output", help="输出文件名（配合 --url）")
    parser.add_argument("--ingest", action="store_true", help="爬取后自动重建向量库")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.url:
        result = crawl_single(args.url, args.output)
        if not result:
            sys.exit(1)
    else:
        saved = crawl_presets()
        if not saved:
            print("无内容被保存")
            sys.exit(1)

    if args.ingest:
        print("\n🔨 重建向量库…")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / "rag_ingest.py"), "--reset"])


if __name__ == "__main__":
    main()
