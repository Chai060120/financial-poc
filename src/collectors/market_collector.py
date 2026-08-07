"""
网络行情与资讯采集：个股新闻、行业成分股、批量实时报价。

供实时对比分析（market_compare）使用，不依赖 LLM。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import feedparser

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DEFAULT_ENCODING, REFERENCE_DIR, setup_logging
from src.analysis.market_data import MarketSnapshot, _without_proxy, fetch_market_snapshot

logger = setup_logging(__name__)

MARKET_CACHE_PATH = REFERENCE_DIR / "market_cache.json"
CACHE_TTL_SECONDS = 300


@dataclass
class NewsItem:
    title: str
    url: str = ""
    publish_time: str = ""
    source: str = ""


@dataclass
class PeerQuote:
    entity_id: str
    entity_name: str
    price: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    change_pct: float | None = None
    market_cap: float | None = None


def _symbol_from_entity_id(entity_id: str) -> str:
    return str(entity_id or "").split(".")[0].strip().zfill(6)


def _entity_id_from_code(code: str) -> str:
    code = code.zfill(6)
    suffix = "SH" if code.startswith("6") else "SZ"
    return f"{code}.{suffix}"


def _load_cache() -> dict[str, Any]:
    if not MARKET_CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(MARKET_CACHE_PATH.read_text(encoding=DEFAULT_ENCODING))
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding=DEFAULT_ENCODING,
    )


def _cache_get(key: str) -> Any | None:
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    ts = entry.get("ts", 0)
    if time.time() - ts > CACHE_TTL_SECONDS:
        return None
    return entry.get("data")


def _cache_set(key: str, data: Any) -> None:
    cache = _load_cache()
    cache[key] = {"ts": time.time(), "data": data}
    _save_cache(cache)


def fetch_stock_news(
    entity_name: str,
    entity_id: str = "",
    *,
    limit: int = 5,
) -> list[NewsItem]:
    """爬取个股相关新闻（东方财富 + Google News RSS）。"""
    code = _symbol_from_entity_id(entity_id) if entity_id else ""
    items: list[NewsItem] = []
    seen: set[str] = set()

    # 1) 东方财富个股新闻
    if code:
        cached = _cache_get(f"news_em:{code}")
        if isinstance(cached, list):
            for row in cached[:limit]:
                title = str(row.get("title") or "").strip()
                if title and title not in seen:
                    seen.add(title)
                    items.append(
                        NewsItem(
                            title=title,
                            url=str(row.get("url") or ""),
                            publish_time=str(row.get("publish_time") or ""),
                            source="eastmoney",
                        )
                    )
        else:
            try:
                import akshare as ak

                with _without_proxy():
                    df = ak.stock_news_em(symbol=code)
                rows: list[dict[str, str]] = []
                if df is not None and not df.empty:
                    for _, record in df.head(limit * 2).iterrows():
                        title = str(record.get("新闻标题") or record.get("title") or "").strip()
                        if not title:
                            continue
                        row = {
                            "title": title,
                            "url": str(record.get("新闻链接") or record.get("url") or ""),
                            "publish_time": str(
                                record.get("发布时间") or record.get("publish_time") or ""
                            ),
                        }
                        rows.append(row)
                        if title not in seen:
                            seen.add(title)
                            items.append(
                                NewsItem(
                                    title=title,
                                    url=row["url"],
                                    publish_time=row["publish_time"],
                                    source="eastmoney",
                                )
                            )
                        if len(items) >= limit:
                            break
                _cache_set(f"news_em:{code}", rows)
            except Exception as exc:
                logger.debug("东方财富个股新闻失败: %s", exc)

    # 2) Google News RSS（按公司名）
    if len(items) < limit and entity_name:
        query = urllib.parse.quote(f"{entity_name} 股票")
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        )
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[: limit * 2]:
                title = str(getattr(entry, "title", "") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                published = str(getattr(entry, "published", "") or "")
                link = str(getattr(entry, "link", "") or "")
                items.append(
                    NewsItem(title=title, url=link, publish_time=published, source="google_news")
                )
                if len(items) >= limit:
                    break
        except Exception as exc:
            logger.debug("Google News RSS 失败: %s", exc)

    return items[:limit]


def fetch_industry_peers(
    industry: str,
    *,
    exclude_code: str = "",
    limit: int = 8,
) -> list[tuple[str, str]]:
    """
    从网络获取同行业成分股（代码, 名称）。

    优先 AkShare 行业板块；失败时返回空列表。
    """
    if not industry or industry in {"-", "--"}:
        return []

    cache_key = f"peers:{industry}:{exclude_code}:{limit}"
    cached = _cache_get(cache_key)
    if isinstance(cached, list):
        return [(str(a), str(b)) for a, b in cached]

    peers: list[tuple[str, str]] = []
    try:
        import akshare as ak

        with _without_proxy():
            boards = ak.stock_board_industry_name_em()
            if boards is None or boards.empty:
                return []

            name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
            matched_name = ""
            for _, row in boards.iterrows():
                board_name = str(row.get(name_col) or "")
                if industry in board_name or board_name in industry:
                    matched_name = board_name
                    break
            if not matched_name:
                key = industry[:2]
                for _, row in boards.iterrows():
                    board_name = str(row.get(name_col) or "")
                    if key in board_name:
                        matched_name = board_name
                        break

            if not matched_name:
                return []

            cons = ak.stock_board_industry_cons_em(symbol=matched_name)
        if cons is None or cons.empty:
            return []

        code_col = "代码" if "代码" in cons.columns else cons.columns[0]
        name_col2 = "名称" if "名称" in cons.columns else cons.columns[1]
        exclude = exclude_code.zfill(6)
        for _, row in cons.iterrows():
            code = str(row.get(code_col) or "").zfill(6)
            name = str(row.get(name_col2) or "")
            if not code or code == exclude:
                continue
            peers.append((code, name))
            if len(peers) >= limit:
                break
    except Exception as exc:
        logger.warning("行业成分股获取失败 (%s): %s", industry, exc)

    _cache_set(cache_key, peers)
    return peers


def snapshot_to_peer(snapshot: MarketSnapshot) -> PeerQuote:
    return PeerQuote(
        entity_id=snapshot.entity_id,
        entity_name=snapshot.entity_name,
        price=snapshot.price,
        pe_ttm=snapshot.pe_ttm,
        pb=snapshot.pb,
        change_pct=snapshot.change_pct,
        market_cap=snapshot.market_cap,
    )


def fetch_peer_quotes(
    peers: list[tuple[str, str]],
    *,
    max_peers: int = 8,
) -> list[PeerQuote]:
    """批量拉取同行实时报价（逐只调用轻量 Eastmoney 接口）。"""
    quotes: list[PeerQuote] = []
    for code, name in peers[:max_peers]:
        entity_id = _entity_id_from_code(code)
        cache_key = f"quote:{entity_id}"
        cached = _cache_get(cache_key)
        if cached:
            quotes.append(PeerQuote(**cached))
            continue
        snapshot = fetch_market_snapshot(entity_id, name)
        peer = snapshot_to_peer(snapshot)
        _cache_set(cache_key, asdict(peer))
        quotes.append(peer)
    return quotes


def fetch_watchlist_quotes(entity_ids: list[str]) -> list[PeerQuote]:
    """监控列表批量实时报价。"""
    quotes: list[PeerQuote] = []
    for entity_id in entity_ids:
        snapshot = fetch_market_snapshot(entity_id)
        quotes.append(snapshot_to_peer(snapshot))
    return quotes
