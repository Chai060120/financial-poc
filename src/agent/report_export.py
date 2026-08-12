"""研究报告导出：Markdown / HTML（便于演示与转发）。"""

from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

from config import DEFAULT_ENCODING, DOCS_DIR, ensure_dirs

EXPORT_DIR = DOCS_DIR / "exports"


def _safe_stem(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", (name or "report").strip())
    return cleaned[:60] or "report"


def report_text_to_markdown(
    report_text: str,
    *,
    title: str = "",
    entity_name: str = "",
) -> str:
    heading = title or (f"{entity_name} 投研分析报告" if entity_name else "Financial Research Agent 报告")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = (report_text or "").strip() or "（空报告）"
    return (
        f"# {heading}\n\n"
        f"- 生成时间: {stamp}\n"
        f"- 工具: Financial Research Agent（PoC）\n"
        f"- 免责声明: 不构成投资建议\n\n"
        f"---\n\n"
        f"```text\n{body}\n```\n"
    )


def report_text_to_html(
    report_text: str,
    *,
    title: str = "",
    entity_name: str = "",
) -> str:
    heading = title or (f"{entity_name} 投研分析报告" if entity_name else "Financial Research Agent 报告")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = html.escape((report_text or "").strip() or "（空报告）")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(heading)}</title>
  <style>
    body {{
      margin: 0; padding: 32px 20px; background: #0f1419; color: #e8eef6;
      font-family: "Segoe UI", "PingFang SC", sans-serif; line-height: 1.55;
    }}
    .wrap {{ max-width: 880px; margin: 0 auto; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 8px; }}
    .meta {{ color: #8b9bb0; font-size: 0.9rem; margin-bottom: 20px; }}
    pre {{
      white-space: pre-wrap; word-break: break-word;
      background: #151b23; border: 1px solid #2a3544; border-radius: 12px;
      padding: 18px; font-family: ui-monospace, Consolas, monospace; font-size: 0.86rem;
    }}
    .foot {{ margin-top: 18px; color: #8b9bb0; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(heading)}</h1>
    <div class="meta">生成时间：{stamp} · Financial Research Agent</div>
    <pre>{body}</pre>
    <div class="foot">免责声明：PoC 自动分析，不构成投资建议。可使用浏览器「打印 → 另存为 PDF」导出 PDF。</div>
  </div>
</body>
</html>
"""


def export_report(
    report_text: str,
    *,
    entity_name: str = "",
    fmt: str = "md",
) -> Path:
    """将最新报告写入 docs/exports/，返回文件路径。"""
    ensure_dirs()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or "md").lower().strip()
    if fmt not in {"md", "html", "markdown"}:
        raise ValueError("仅支持 format=md 或 html")
    if fmt == "markdown":
        fmt = "md"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{_safe_stem(entity_name or 'report')}_{stamp}"
    if fmt == "html":
        content = report_text_to_html(report_text, entity_name=entity_name)
        path = EXPORT_DIR / f"{stem}.html"
    else:
        content = report_text_to_markdown(report_text, entity_name=entity_name)
        path = EXPORT_DIR / f"{stem}.md"
    path.write_text(content, encoding=DEFAULT_ENCODING)
    return path
