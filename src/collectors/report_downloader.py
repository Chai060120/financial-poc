"""
从巨潮资讯（cninfo）按公司+年份下载定期报告 PDF。

供网页/对话 Agent：输入「美的集团 2024」时自动抓年报，无需手动上传。
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RAW_PDF_DIR, ensure_dirs, setup_logging

logger = setup_logging(__name__)

CNINFO_SEARCH = "https://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "https://static.cninfo.com.cn/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Origin": "https://www.cninfo.com.cn",
    "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "X-Requested-With": "XMLHttpRequest",
}

_CATEGORY = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
}

_SKIP_TITLE = re.compile(
    r"(摘要|已取消|取消|英文|English|更正公告|补充公告|意见|说明|提示性)"
)


@dataclass
class DownloadedReport:
    path: Path
    entity_name: str
    entity_id: str
    report_year: str
    report_type: str
    title: str
    url: str
    from_cache: bool = False


class ReportDownloadError(Exception):
    """年报下载失败。"""


def _symbol_and_market(entity_id: str) -> tuple[str, str]:
    code = str(entity_id or "").split(".")[0].strip().zfill(6)
    market = "sh" if code.startswith(("5", "6", "9")) else "sz"
    if "." in str(entity_id):
        suffix = str(entity_id).split(".")[-1].upper()
        if suffix == "SH":
            market = "sh"
        elif suffix == "SZ":
            market = "sz"
        elif suffix == "BJ":
            market = "bj"
    return code, market


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def _lookup_org_id(session: requests.Session, code: str) -> str:
    """通过巨潮 topSearch 查 orgId（需 POST，GET 会 500）。"""
    try:
        # 先访问首页建立会话，降低风控拦截概率
        session.get("https://www.cninfo.com.cn/", timeout=15)
        resp = session.post(
            CNINFO_SEARCH,
            data={"keyWord": code},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise ReportDownloadError(f"查询 orgId 失败: {code}") from exc

    rows = data if isinstance(data, list) else data.get("data") or data.get("list") or []
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sec = str(row.get("code") or row.get("secCode") or "").zfill(6)
        if sec == code:
            org = str(row.get("orgId") or row.get("orgid") or "").strip()
            if org:
                return org
    # 宽松回退：取第一条带 orgId 的
    for row in rows:
        if isinstance(row, dict) and row.get("orgId"):
            return str(row["orgId"]).strip()
    raise ReportDownloadError(f"未找到巨潮 orgId: {code}")


def _date_window(year: str, report_type: str) -> str:
    y = int(year)
    if report_type == "年报":
        # 年报通常在次年披露
        return f"{y + 1}-01-01~{y + 1}-12-31"
    if report_type == "半年报":
        return f"{y}-07-01~{y + 1}-03-31"
    if report_type == "一季报":
        return f"{y}-04-01~{y}-09-30"
    if report_type == "三季报":
        return f"{y}-10-01~{y + 1}-03-31"
    return f"{y}-01-01~{y + 1}-12-31"


def _is_preferred_title(title: str, year: str, report_type: str) -> bool:
    text = title or ""
    if _SKIP_TITLE.search(text):
        return False
    if year and year not in text and f"{year}年" not in text:
        # 有些标题写「2024年年度报告」
        return False
    if report_type == "年报":
        return "年度报告" in text or "年报" in text
    if report_type == "半年报":
        return "半年度报告" in text or "中期报告" in text or "半年报" in text
    if report_type == "一季报":
        return "一季度" in text or "第一季度" in text
    if report_type == "三季报":
        return "三季度" in text or "第三季度" in text
    return True


def _query_announcements(
    session: requests.Session,
    *,
    code: str,
    market: str,
    org_id: str,
    year: str,
    report_type: str,
) -> list[dict[str, Any]]:
    category = _CATEGORY.get(report_type, _CATEGORY["年报"])
    column = "szse" if market == "sz" else "sse"
    if market == "bj":
        column = "bj"
    payload = {
        "pageNum": "1",
        "pageSize": "30",
        "column": column,
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},{org_id}",
        "searchkey": "",
        "secid": "",
        "category": category,
        "trade": "",
        "seDate": _date_window(year, report_type),
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    try:
        resp = session.post(CNINFO_QUERY, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise ReportDownloadError(f"查询公告失败: {code} {year}{report_type}") from exc

    rows = data.get("announcements") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _pick_announcement(
    rows: list[dict[str, Any]],
    *,
    year: str,
    report_type: str,
) -> dict[str, Any] | None:
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("announcementTitle") or row.get("title") or "")
        if _SKIP_TITLE.search(title):
            continue
        adjunct = str(row.get("adjunctUrl") or "")
        if not adjunct.lower().endswith(".pdf"):
            continue
        if _is_preferred_title(title, year, report_type):
            preferred.append(row)
        else:
            fallback.append(row)
    if preferred:
        return preferred[0]
    return fallback[0] if fallback else None


def _safe_filename(entity_name: str, year: str, report_type: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", entity_name).strip() or "report"
    return f"{name}{year}年{report_type}.pdf"


def find_local_report(
    entity_name: str,
    year: str,
    *,
    report_type: str = "年报",
    directory: Path | None = None,
) -> Path | None:
    """在本地 raw/pdf 中查找已下载的同名年报。"""
    scan_dir = directory or RAW_PDF_DIR
    if not scan_dir.is_dir():
        return None
    expected = _safe_filename(entity_name, year, report_type)
    exact = scan_dir / expected
    if exact.is_file() and exact.stat().st_size > 50_000:
        return exact
    for path in scan_dir.glob("*.pdf"):
        stem = path.stem
        if entity_name in stem and year in stem and path.stat().st_size > 50_000:
            if report_type == "年报" and ("半年" in stem or "季报" in stem or "季度" in stem):
                continue
            return path
    return None


def download_periodic_report(
    entity_name: str,
    entity_id: str,
    year: str,
    *,
    report_type: str = "年报",
    force: bool = False,
) -> DownloadedReport:
    """
    下载指定公司某年定期报告 PDF 到 data/raw/pdf/。

    优先复用本地同名文件；否则从巨潮资讯抓取。
    """
    year = str(year).strip()
    if not re.fullmatch(r"20\d{2}", year):
        raise ReportDownloadError(f"年份无效: {year}")
    if report_type not in _CATEGORY:
        report_type = "年报"

    ensure_dirs()
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = find_local_report(entity_name, year, report_type=report_type)
        if cached is not None:
            logger.info("复用本地财报: %s", cached.name)
            return DownloadedReport(
                path=cached,
                entity_name=entity_name,
                entity_id=entity_id,
                report_year=year,
                report_type=report_type,
                title=cached.name,
                url="",
                from_cache=True,
            )

    code, market = _symbol_and_market(entity_id)
    session = _session()
    org_id = _lookup_org_id(session, code)
    rows = _query_announcements(
        session,
        code=code,
        market=market,
        org_id=org_id,
        year=year,
        report_type=report_type,
    )
    picked = _pick_announcement(rows, year=year, report_type=report_type)
    if picked is None:
        # 扩大窗口再试一次（有的年报披露较晚）
        payload_year = str(int(year) + 1)
        rows = _query_announcements(
            session,
            code=code,
            market=market,
            org_id=org_id,
            year=payload_year,
            report_type=report_type,
        )
        # 仍按原报告年份筛标题
        picked = _pick_announcement(rows, year=year, report_type=report_type)
    if picked is None:
        raise ReportDownloadError(
            f"巨潮未找到 {entity_name}（{entity_id}）{year}年{report_type}"
        )

    adjunct = str(picked.get("adjunctUrl") or "")
    title = str(picked.get("announcementTitle") or f"{entity_name}{year}年{report_type}")
    url = CNINFO_STATIC + adjunct.lstrip("/")
    dest = RAW_PDF_DIR / _safe_filename(entity_name, year, report_type)

    logger.info("下载财报: %s -> %s", title, dest.name)
    try:
        with session.get(url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".pdf.part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        fh.write(chunk)
            size = tmp.stat().st_size
            if size < 50_000:
                tmp.unlink(missing_ok=True)
                raise ReportDownloadError(f"下载文件过小（{size} bytes），可能不是完整 PDF")
            tmp.replace(dest)
    except ReportDownloadError:
        raise
    except Exception as exc:
        raise ReportDownloadError(f"下载 PDF 失败: {url}") from exc

    time.sleep(0.3)
    return DownloadedReport(
        path=dest,
        entity_name=entity_name,
        entity_id=entity_id,
        report_year=year,
        report_type=report_type,
        title=title,
        url=url,
        from_cache=False,
    )


def ensure_report_pdf(
    entity_name: str,
    entity_id: str,
    year: str | None = None,
    *,
    report_type: str = "年报",
) -> DownloadedReport:
    """确保本地有可用财报；缺少年份时默认最近完整年报年（去年）。"""
    from datetime import date

    report_year = year or str(date.today().year - 1)
    return download_periodic_report(
        entity_name,
        entity_id,
        report_year,
        report_type=report_type,
    )
