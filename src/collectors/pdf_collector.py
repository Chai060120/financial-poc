"""
PDF 文件采集器：扫描 raw/pdf 目录，发现 PDF 文件路径。

仅负责文件发现，不读取、不解析 PDF 内容。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RAW_PDF_DIR, ensure_dirs, setup_logging

logger = setup_logging(__name__)

PDF_SUFFIX = ".pdf"


def is_pdf_file(path: Path) -> bool:
    """判断路径是否为 PDF 文件（按后缀名，大小写不敏感）。"""
    return path.is_file() and path.suffix.lower() == PDF_SUFFIX


def collect_pdf_paths(
    directory: Path | None = None,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    扫描目录下的 PDF 文件，返回路径列表。

    Args:
        directory: 扫描目录，默认使用 config.RAW_PDF_DIR。
        recursive: 是否递归扫描子目录，默认 False。

    Returns:
        按路径字符串排序的 PDF 文件 Path 列表。
    """
    scan_dir = directory or RAW_PDF_DIR

    if not scan_dir.exists():
        logger.warning("PDF 目录不存在: %s", scan_dir)
        return []

    if not scan_dir.is_dir():
        logger.error("PDF 路径不是目录: %s", scan_dir)
        return []

    if recursive:
        candidates = scan_dir.rglob("*")
    else:
        candidates = scan_dir.iterdir()

    pdf_paths = sorted(
        (path.resolve() for path in candidates if is_pdf_file(path)),
        key=lambda p: str(p).lower(),
    )

    logger.info("在 %s 发现 %d 个 PDF 文件", scan_dir, len(pdf_paths))
    return pdf_paths


def main() -> None:
    """命令行调试入口。"""
    ensure_dirs()
    paths = collect_pdf_paths()

    if not paths:
        print(f"未发现 PDF 文件，请将文件放入: {RAW_PDF_DIR}")
        return

    print(f"共发现 {len(paths)} 个 PDF 文件:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
