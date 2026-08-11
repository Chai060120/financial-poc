"""
从多种数据源获取 A 股估值行情（现价 / PE / PB / 行业）。

优先级：东方财富单股 API → 新浪行情 → AkShare 个股/历史 → AkShare 全市场。
网络不可用时由 valuation 模块用「现价 ÷ 财报 EPS/BVPS」补算 PE/PB。
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging

logger = setup_logging(__name__)

_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.eastmoney.com/",
}


@dataclass
class MarketSnapshot:
    """单只股票行情摘要（实时或推算）。"""

    entity_id: str
    entity_name: str
    price: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    market_cap: float | None = None
    change_pct: float | None = None
    industry: str = ""
    source: str = ""
    price_source: str = ""
    pe_source: str = ""
    pb_source: str = ""


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _symbol_from_entity_id(entity_id: str) -> str:
    return str(entity_id or "").split(".")[0].strip().zfill(6)


def _secid(code: str) -> str:
    market = "1" if code.startswith("6") else "0"
    return f"{market}.{code}"


@contextmanager
def _without_proxy():
    """临时禁用系统代理，避免行情请求被错误代理拦截。"""
    saved = {key: os.environ.pop(key) for key in _PROXY_KEYS if key in os.environ}
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ[key] = value


def _http_get(url: str, *, params: dict | None = None, headers: dict | None = None) -> str:
    import requests

    merged = dict(_DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    response = requests.get(url, params=params, headers=merged, timeout=12)
    response.raise_for_status()
    return response.text


def _normalize_price(raw: float | None) -> float | None:
    if raw is None or raw <= 0:
        return None
    if raw > 1000:
        return round(raw / 100.0, 2)
    return round(raw, 2)


def _normalize_ratio(raw: float | None) -> float | None:
    """PE/PB 字段：东财有时放大了 100 倍，有时是直接值。"""
    if raw is None or raw <= 0:
        return None
    if raw > 500:
        return round(raw / 100.0, 2)
    return round(raw, 2)


def _pick_ratio(*values: object) -> float | None:
    for value in values:
        parsed = _normalize_ratio(_to_float(value))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _fetch_eastmoney_quote(code: str, snapshot: MarketSnapshot) -> bool:
    """东方财富单股接口（轻量，不拉全市场）。"""
    try:
        body = _http_get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "secid": _secid(code),
                "fields": "f43,f58,f127,f9,f23,f162,f167,f116,f170",
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            },
        )
    except Exception as exc:
        logger.debug("东方财富单股接口失败: %s", exc)
        return False

    try:
        import json

        payload = json.loads(body)
        data = (payload.get("data") or {}) if isinstance(payload, dict) else {}
    except Exception:
        return False

    if not data:
        return False

    name = str(data.get("f58") or snapshot.entity_name or "")
    if name:
        snapshot.entity_name = name

    snapshot.price = snapshot.price or _normalize_price(_to_float(data.get("f43")))
    snapshot.pe_ttm = snapshot.pe_ttm or _pick_ratio(
        data.get("f162"), data.get("f9"), data.get("f115")
    )
    snapshot.pb = snapshot.pb or _pick_ratio(data.get("f167"), data.get("f23"))

    cap_raw = _to_float(data.get("f116"))
    if cap_raw is not None:
        snapshot.market_cap = cap_raw

    if snapshot.price is not None:
        snapshot.price_source = snapshot.price_source or "eastmoney:quote"
    if snapshot.pe_ttm is not None:
        snapshot.pe_source = snapshot.pe_source or "eastmoney:quote"
    if snapshot.pb is not None:
        snapshot.pb_source = snapshot.pb_source or "eastmoney:quote"

    industry = str(data.get("f127") or "").strip()
    if industry and industry not in {"-", "--"}:
        snapshot.industry = industry

    change = _to_float(data.get("f170"))
    if change is not None:
        snapshot.change_pct = change / 100.0 if abs(change) > 50 else change

    if snapshot.price or snapshot.pe_ttm or snapshot.pb:
        snapshot.source = snapshot.source or "eastmoney:quote"
        return True
    return False


def _fetch_sina_quote(code: str, snapshot: MarketSnapshot) -> bool:
    """新浪实时行情（仅现价，作为轻量备用）。"""
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        raw = _http_get(
            f"https://hq.sinajs.cn/list={prefix}{code}",
            headers={"Referer": "https://finance.sina.com.cn"},
        )
    except Exception as exc:
        logger.debug("新浪行情失败: %s", exc)
        return False

    match = re.search(r'"([^"]+)"', raw)
    if not match:
        return False
    parts = match.group(1).split(",")
    if len(parts) < 4:
        return False

    name = parts[0].strip()
    price = _to_float(parts[3])
    if name:
        snapshot.entity_name = snapshot.entity_name or name
    if price is not None and price > 0:
        snapshot.price = price
        snapshot.price_source = "sina:quote"
        snapshot.source = snapshot.source or "sina:quote"
        return True
    return False


def _apply_spot_row(snapshot: MarketSnapshot, record) -> None:
    snapshot.entity_name = str(record.get("名称") or snapshot.entity_name or "")
    if snapshot.price is None:
        snapshot.price = _to_float(record.get("最新价"))
        if snapshot.price is not None:
            snapshot.price_source = "akshare:spot_em"
    if snapshot.pe_ttm is None:
        snapshot.pe_ttm = _to_float(record.get("市盈率-动态"))
        if snapshot.pe_ttm is not None:
            snapshot.pe_source = "akshare:spot_em"
    if snapshot.pb is None:
        snapshot.pb = _to_float(record.get("市净率"))
        if snapshot.pb is not None:
            snapshot.pb_source = "akshare:spot_em"
    snapshot.market_cap = snapshot.market_cap or _to_float(record.get("总市值"))
    snapshot.change_pct = snapshot.change_pct or _to_float(record.get("涨跌幅"))
    snapshot.source = snapshot.source or "akshare:spot_em"


def _fetch_from_spot_em(ak, code: str, snapshot: MarketSnapshot) -> bool:
    spot = ak.stock_zh_a_spot_em()
    row = spot.loc[spot["代码"] == code]
    if row.empty:
        return False
    _apply_spot_row(snapshot, row.iloc[0])
    return any(
        value is not None
        for value in (snapshot.price, snapshot.pe_ttm, snapshot.pb)
    )


def _fetch_from_individual_spot(ak, code: str, snapshot: MarketSnapshot) -> bool:
    prefix = "SH" if code.startswith("6") else "SZ"
    try:
        df = ak.stock_individual_spot_xq(symbol=f"{prefix}{code}")
    except Exception:
        return False
    if df is None or df.empty:
        return False
    mapping = dict(zip(df["item"].astype(str), df["value"].astype(str), strict=False))
    if snapshot.price is None:
        snapshot.price = _to_float(mapping.get("现价") or mapping.get("最新"))
        if snapshot.price is not None:
            snapshot.price_source = "akshare:individual_spot_xq"
    if snapshot.pe_ttm is None:
        snapshot.pe_ttm = _to_float(
            mapping.get("市盈率(TTM)") or mapping.get("市盈率")
        )
        if snapshot.pe_ttm is not None:
            snapshot.pe_source = "akshare:individual_spot_xq"
    if snapshot.pb is None:
        snapshot.pb = _to_float(mapping.get("市净率"))
        if snapshot.pb is not None:
            snapshot.pb_source = "akshare:individual_spot_xq"
    if snapshot.price or snapshot.pe_ttm or snapshot.pb:
        snapshot.source = snapshot.source or "akshare:individual_spot_xq"
        return True
    return False


def _fetch_from_hist(ak, code: str, snapshot: MarketSnapshot) -> bool:
    """最近一个交易日收盘价。"""
    if snapshot.price is not None:
        return False
    try:
        hist = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            adjust="qfq",
        )
    except Exception as exc:
        logger.debug("AkShare 历史行情失败: %s", exc)
        return False
    if hist is None or hist.empty:
        return False
    last = hist.iloc[-1]
    price = _to_float(last.get("收盘"))
    if price is not None and price > 0:
        snapshot.price = price
        snapshot.price_source = "akshare:hist_close"
        snapshot.source = snapshot.source or "akshare:hist_close"
        return True
    return False


def _fetch_from_individual_info_em(ak, code: str, snapshot: MarketSnapshot) -> bool:
    """东方财富个股资料页（单股轻量，补 PE/PB/EPS）。"""
    if snapshot.pe_ttm and snapshot.pb:
        return False
    try:
        info = ak.stock_individual_info_em(symbol=code)
    except Exception as exc:
        logger.debug("个股资料接口失败: %s", exc)
        return False
    if info is None or info.empty:
        return False

    mapping = dict(
        zip(
            info["item"].astype(str).str.strip(),
            info["value"].astype(str).str.strip(),
            strict=False,
        )
    )
    updated = False
    if snapshot.pe_ttm is None:
        pe = _pick_ratio(
            mapping.get("市盈率-动态"),
            mapping.get("市盈率"),
            mapping.get("PE(TTM)"),
        )
        if pe is not None:
            snapshot.pe_ttm = pe
            snapshot.pe_source = "akshare:individual_info_em"
            updated = True
    if snapshot.pb is None:
        pb = _pick_ratio(mapping.get("市净率"), mapping.get("PB"))
        if pb is not None:
            snapshot.pb = pb
            snapshot.pb_source = "akshare:individual_info_em"
            updated = True
    if updated:
        snapshot.source = snapshot.source or "akshare:individual_info_em"
    return updated


def _network_eps_from_info_em(code: str) -> float | None:
    """从东财资料页读取每股收益，用于无财报时的 PE 推算。"""
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        with _without_proxy():
            info = ak.stock_individual_info_em(symbol=code)
    except Exception:
        return None
    if info is None or info.empty:
        return None
    mapping = dict(
        zip(
            info["item"].astype(str).str.strip(),
            info["value"].astype(str).str.strip(),
            strict=False,
        )
    )
    for key in ("每股收益", "基本每股收益", "EPS"):
        eps = _to_float(mapping.get(key))
        if eps is not None and eps > 0:
            return eps
    return None


def _network_bvps_from_info_em(code: str) -> float | None:
    """从东财资料页读取每股净资产，用于无财报时的 PB 推算。"""
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        with _without_proxy():
            info = ak.stock_individual_info_em(symbol=code)
    except Exception:
        return None
    if info is None or info.empty:
        return None
    mapping = dict(
        zip(
            info["item"].astype(str).str.strip(),
            info["value"].astype(str).str.strip(),
            strict=False,
        )
    )
    for key in ("每股净资产", "每股净资产(元)", "BVPS"):
        bvps = _to_float(mapping.get(key))
        if bvps is not None and bvps > 0:
            return bvps
    return None


def _fetch_industry(ak, code: str, snapshot: MarketSnapshot) -> None:
    if snapshot.industry:
        return
    try:
        info = ak.stock_individual_info_em(symbol=code)
        if info is not None and not info.empty:
            industry_rows = info.loc[info["item"] == "行业", "value"]
            if not industry_rows.empty:
                snapshot.industry = str(industry_rows.iloc[0])
    except Exception as exc:
        logger.debug("行业信息获取跳过: %s", exc)


def enrich_market_from_fundamentals(
    market: MarketSnapshot,
    fundamentals: dict,
) -> MarketSnapshot:
    """
    用财报 EPS / 每股净资产与现价补算 PE、PB（适用于任意已索引财报）。
    """
    eps = _fundamental_scalar(fundamentals, "eps")
    bvps = _fundamental_scalar(fundamentals, "bvps")

    if market.pe_ttm is None and market.price and eps and eps > 0:
        market.pe_ttm = round(market.price / eps, 2)
        market.pe_source = "computed:price/eps"

    if market.pb is None and market.price and bvps and bvps > 0:
        market.pb = round(market.price / bvps, 2)
        market.pb_source = "computed:price/bvps"

    if not market.source and (market.pe_ttm or market.pb or market.price):
        market.source = "fundamentals_enriched"

    return market


def _fundamental_scalar(fundamentals: dict, key: str) -> float | None:
    item = fundamentals.get(key)
    if not isinstance(item, dict):
        return None
    raw = str(item.get("raw") or item.get("display") or "")
    raw = raw.replace("元/股", "").replace("%", "").replace(",", "").strip()
    return _to_float(raw)


def fetch_market_snapshot(entity_id: str, entity_name: str = "") -> MarketSnapshot:
    """拉取单只股票 PE / PB / 价格；多源合并，单源失败不中断。"""
    code = _symbol_from_entity_id(entity_id)
    snapshot = MarketSnapshot(entity_id=entity_id, entity_name=entity_name)
    errors: list[str] = []

    with _without_proxy():
        if not _fetch_eastmoney_quote(code, snapshot):
            errors.append("eastmoney:quote")
        if snapshot.price is None and not _fetch_sina_quote(code, snapshot):
            errors.append("sina:quote")

        try:
            import akshare as ak
        except ImportError:
            ak = None  # type: ignore[assignment]

        if ak is not None:
            for fetcher in (
                lambda: _fetch_from_individual_spot(ak, code, snapshot),
                lambda: _fetch_from_individual_info_em(ak, code, snapshot),
                lambda: _fetch_from_spot_em(ak, code, snapshot),
                lambda: _fetch_from_hist(ak, code, snapshot),
            ):
                try:
                    fetcher()
                except Exception as exc:
                    errors.append(str(exc))
                    logger.debug("AkShare 行情源失败: %s", exc)
            _fetch_industry(ak, code, snapshot)

    if not any((snapshot.price, snapshot.pe_ttm, snapshot.pb)):
        logger.warning(
            "无法获取 %s 实时行情（尝试过: %s）",
            entity_id,
            ", ".join(errors[:3]) if errors else "全部数据源",
        )

    return snapshot


def fetch_market_snapshot_enriched(
    entity_id: str,
    entity_name: str = "",
    fundamentals: dict | None = None,
) -> MarketSnapshot:
    """拉取行情；缺 PE/PB 时用财报 EPS/BVPS 与现价推算。"""
    snapshot = fetch_market_snapshot(entity_id, entity_name)
    fund = dict(fundamentals or {})
    if _fundamental_scalar(fund, "eps") is None:
        code = _symbol_from_entity_id(entity_id)
        eps = _network_eps_from_info_em(code)
        if eps is not None:
            fund.setdefault(
                "eps",
                {"raw": str(eps), "display": f"{eps:.2f} 元/股", "source": "network"},
            )
    if _fundamental_scalar(fund, "bvps") is None:
        code = _symbol_from_entity_id(entity_id)
        bvps = _network_bvps_from_info_em(code)
        if bvps is not None:
            fund.setdefault(
                "bvps",
                {"raw": str(bvps), "display": f"{bvps:.2f} 元/股", "source": "network"},
            )
    if fund:
        snapshot = enrich_market_from_fundamentals(snapshot, fund)
    return snapshot
