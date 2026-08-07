"""
代码检测脚本：语法、导入、CLI、核心功能冒烟测试。

用法:
    python scripts/check_code.py           # 快速检测（不加载 embedding 模型）
    python scripts/check_code.py --full    # 含检索/估值（较慢）
"""

from __future__ import annotations

import argparse
import compileall
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
WARN = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")


def bad(msg: str, detail: str = "") -> None:
    global FAIL
    FAIL += 1
    line = f"  [FAIL] {msg}"
    if detail:
        line += f" — {detail}"
    print(line)


def warn(msg: str, detail: str = "") -> None:
    global WARN
    WARN += 1
    line = f"  [WARN] {msg}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_syntax() -> None:
    print("\n== 1. 语法检查 ==")
    ok_path = compileall.compile_dir(ROOT / "src", quiet=1)
    ok_scripts = compileall.compile_dir(ROOT / "scripts", quiet=1)
    ok_cfg = compileall.compile_file(ROOT / "config.py", quiet=1)
    if ok_path and ok_scripts and ok_cfg:
        ok("src/ scripts/ config.py 语法正常")
    else:
        bad("语法检查未通过")


def check_imports() -> None:
    print("\n== 2. 模块导入 ==")
    modules = [
        "config",
        "src.collectors.news_collector",
        "src.collectors.market_collector",
        "src.analysis.market_data",
        "src.analysis.valuation",
        "src.analysis.market_compare",
        "src.agent.financial_agent",
        "src.utils.query_insights",
        "src.vectorstore.unified_retrieval",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            ok(name)
        except Exception as exc:
            bad(name, str(exc))


def check_cli() -> None:
    print("\n== 3. CLI 子命令 ==")
    import subprocess

    for cmd in ("analyze", "compare", "valuate", "query", "sync"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "agent.py"), cmd, "-h"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            ok(f"agent.py {cmd} -h")
        else:
            bad(f"agent.py {cmd}", result.stderr.strip()[:120])


def check_market_data() -> None:
    print("\n== 4. 实时行情（网络） ==")
    try:
        from src.analysis.market_data import fetch_market_snapshot

        snap = fetch_market_snapshot("600519.SH", "贵州茅台")
        if snap.price or snap.pe_ttm or snap.pb:
            ok(f"600519 行情: 价={snap.price} PE={snap.pe_ttm} PB={snap.pb} ({snap.source})")
        else:
            warn("600519 行情为空", "网络或代理问题，财报推算仍可工作")
    except Exception as exc:
        bad("行情拉取", str(exc))


def check_news_crawl() -> None:
    print("\n== 5. 网络新闻爬取 ==")
    try:
        from src.collectors.market_collector import fetch_stock_news

        items = fetch_stock_news("贵州茅台", "600519.SH", limit=2)
        if items:
            ok(f"新闻 {len(items)} 条: {items[0].title[:40]}...")
        else:
            warn("未抓到新闻", "RSS/AkShare 可能不可用")
    except Exception as exc:
        bad("新闻爬取", str(exc))


def check_compare() -> None:
    print("\n== 6. 实时对比分析 ==")
    try:
        from src.analysis.market_compare import analyze_market_comparison

        result = analyze_market_comparison(
            "贵州茅台",
            engine=None,
            include_valuation=False,
            save_report=False,
        )
        ok(f"对比完成: {result.relative_verdict} | 同业={len(result.peers)} 新闻={len(result.news)}")
        if not result.target.pe_ttm and not result.target.price:
            warn("对比结果缺少 PE/现价")
    except Exception as exc:
        bad("实时对比", str(exc))


def check_chroma() -> None:
    print("\n== 7. 向量索引 ==")
    chroma = ROOT / "data" / "chroma"
    if chroma.is_dir():
        ok(f"Chroma 目录存在: {chroma}")
    else:
        warn("Chroma 目录不存在", "需先运行 sync")


def check_full_query(full: bool) -> None:
    if not full:
        print("\n== 8. 检索/估值（跳过，加 --full 启用） ==")
        return
    print("\n== 8. 检索 + 估值（加载模型，较慢） ==")
    try:
        from src.agent.financial_agent import create_financial_agent

        agent = create_financial_agent()
        payload = agent.query("贵州茅台2024年净利润", top_k=3)
        insight = payload.get("results") or []
        ok(f"检索返回 {payload.get('count', 0)} 条")
        if not insight:
            warn("检索结果为空")

        result = agent.valuate("贵州茅台", save_report=False)
        ok(f"估值: {result.verdict} 评分={result.score:+.1f} 置信度={result.confidence}")
        if result.fundamentals.get("net_profit"):
            raw = result.fundamentals["net_profit"].get("display", "")
            if "862" in raw or "861" in raw:
                ok(f"净利润抽取正确: {raw}")
            else:
                warn(f"净利润需核对: {raw}")
    except Exception as exc:
        bad("检索/估值", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial PoC 代码检测")
    parser.add_argument("--full", action="store_true", help="含检索与估值（慢）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Financial PoC 代码检测")
    print("=" * 60)

    check_syntax()
    check_imports()
    check_cli()
    check_market_data()
    check_news_crawl()
    check_compare()
    check_chroma()
    check_full_query(args.full)

    print("\n" + "=" * 60)
    print(f"  结果: PASS={PASS}  WARN={WARN}  FAIL={FAIL}")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
