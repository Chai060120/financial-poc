"""
股票映射注册表：股票简称 <-> 股票代码。

加载优先级（合并去重）:
1. 内置种子映射（兼容旧版 ENTITY_REGISTRY）
2. 本地 CSV（data/reference/stock_list.csv）
3. 缓存 JSON（data/reference/stock_registry.json）
4. AkShare（可选，首次同步后写入缓存）
5. Tushare（可选，需 TUSHARE_TOKEN）
"""

from __future__ import annotations

import csv
import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    DEFAULT_ENCODING,
    ENABLE_AKSHARE_STOCK_SYNC,
    ENABLE_TUSHARE_STOCK_SYNC,
    STOCK_LIST_CSV,
    STOCK_REGISTRY_CACHE,
    TUSHARE_TOKEN,
    UNKNOWN_ENTITY_ID,
    UNKNOWN_ENTITY_NAME,
    setup_logging,
)

logger = setup_logging(__name__)

# 内置种子（向后兼容）
DEFAULT_STOCK_SEED: dict[str, dict[str, str | tuple[str, ...]]] = {
    "600519.SH": {
        "entity_id": "600519.SH",
        "entity_name": "贵州茅台",
        "aliases": ("贵州茅台", "茅台"),
    },
    "600036.SH": {
        "entity_id": "600036.SH",
        "entity_name": "招商银行",
        "aliases": ("招商银行", "招行"),
    },
    "601318.SH": {
        "entity_id": "601318.SH",
        "entity_name": "中国平安",
        "aliases": ("中国平安", "平安"),
    },
    "600000.SH": {
        "entity_id": "600000.SH",
        "entity_name": "浦发银行",
        "aliases": ("浦发银行", "浦发"),
    },
    "000001.SZ": {
        "entity_id": "000001.SZ",
        "entity_name": "平安银行",
        "aliases": ("平安银行",),
    },
    "000858.SZ": {
        "entity_id": "000858.SZ",
        "entity_name": "五粮液",
        "aliases": ("五粮液",),
    },
}


@dataclass
class StockRecord:
    entity_id: str
    entity_name: str
    aliases: set[str] = field(default_factory=set)


def _normalize_entity_id(raw: str) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        code, market = text.split(".", 1)
        return f"{code.zfill(6)}.{market.upper()}"
    code = text.zfill(6)
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return f"{code}.SH"


def _split_aliases(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [item.strip() for item in str(raw).replace("；", "|").replace(";", "|").split("|")]
    return [item for item in parts if item]


class StockRegistry:
    """股票简称与代码映射注册表（单例）。"""

    _instance: StockRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> StockRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._by_id: dict[str, StockRecord] = {}
        self._alias_to_id: dict[str, str] = {}
        self._sorted_aliases: tuple[str, ...] = ()
        self._load_sources()
        self._initialized = True

    def _register(
        self,
        entity_id: str,
        entity_name: str,
        aliases: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        normalized_id = _normalize_entity_id(entity_id)
        name = str(entity_name or "").strip()
        if not normalized_id or not name:
            return

        alias_set = {name, *_split_aliases("|".join(aliases or ()))}
        alias_set.update(_split_aliases(name))

        existing = self._by_id.get(normalized_id)
        if existing is None:
            self._by_id[normalized_id] = StockRecord(
                entity_id=normalized_id,
                entity_name=name,
                aliases=alias_set,
            )
        else:
            existing.entity_name = name or existing.entity_name
            existing.aliases.update(alias_set)

        record = self._by_id[normalized_id]
        for alias in record.aliases:
            if alias:
                self._alias_to_id[alias] = normalized_id

    def _rebuild_alias_index(self) -> None:
        self._sorted_aliases = tuple(
            sorted(self._alias_to_id.keys(), key=len, reverse=True)
        )

    def _load_sources(self) -> None:
        logger.info("加载股票映射注册表...")
        for entity_id, info in DEFAULT_STOCK_SEED.items():
            self._register(
                str(info["entity_id"]),
                str(info["entity_name"]),
                tuple(str(item) for item in info.get("aliases", ())),
            )

        self._load_csv(STOCK_LIST_CSV)
        self._load_cache(STOCK_REGISTRY_CACHE)

        if ENABLE_AKSHARE_STOCK_SYNC:
            self._load_from_akshare()

        if ENABLE_TUSHARE_STOCK_SYNC and TUSHARE_TOKEN:
            self._load_from_tushare()

        self._rebuild_alias_index()
        self._save_cache()
        logger.info("股票映射加载完成: %d 只股票, %d 个别名", len(self._by_id), len(self._alias_to_id))

    def _load_csv(self, csv_path: Path) -> None:
        if not csv_path.exists():
            logger.info("本地 CSV 不存在，跳过: %s", csv_path)
            return

        loaded = 0
        try:
            with open(csv_path, encoding=DEFAULT_ENCODING, newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    entity_id = (
                        row.get("entity_id")
                        or row.get("ts_code")
                        or row.get("code")
                        or row.get("股票代码")
                        or ""
                    )
                    entity_name = (
                        row.get("entity_name")
                        or row.get("name")
                        or row.get("股票简称")
                        or row.get("简称")
                        or ""
                    )
                    aliases_raw = row.get("aliases") or row.get("别名") or ""
                    if not entity_id or not entity_name:
                        continue
                    self._register(entity_id, entity_name, _split_aliases(aliases_raw) or [entity_name])
                    loaded += 1
        except OSError as exc:
            logger.error("读取 CSV 失败: %s | %s", csv_path, exc)
            return

        logger.info("已从 CSV 加载 %d 条映射: %s", loaded, csv_path)

    def _load_cache(self, cache_path: Path) -> None:
        if not cache_path.exists():
            return
        try:
            with open(cache_path, encoding=DEFAULT_ENCODING) as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取缓存失败，已跳过: %s | %s", cache_path, exc)
            return

        if not isinstance(payload, list):
            return

        loaded = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id") or "")
            entity_name = str(item.get("entity_name") or "")
            aliases = item.get("aliases") or []
            if entity_id and entity_name:
                self._register(entity_id, entity_name, aliases if isinstance(aliases, list) else [])
                loaded += 1

        logger.info("已从缓存加载 %d 条映射: %s", loaded, cache_path)

    def _save_cache(self) -> None:
        cache_path = STOCK_REGISTRY_CACHE
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "entity_id": record.entity_id,
                "entity_name": record.entity_name,
                "aliases": sorted(record.aliases),
            }
            for record in sorted(self._by_id.values(), key=lambda item: item.entity_id)
        ]
        try:
            with open(cache_path, "w", encoding=DEFAULT_ENCODING) as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            logger.debug("股票映射缓存已保存: %s", cache_path)
        except OSError as exc:
            logger.warning("保存股票映射缓存失败: %s | %s", cache_path, exc)

    def _load_from_akshare(self) -> None:
        try:
            import akshare as ak
        except ImportError:
            logger.warning("未安装 akshare，跳过 AkShare 股票列表同步")
            return

        logger.info("从 AkShare 同步 A 股代码名称映射...")
        try:
            df = ak.stock_info_a_code_name()
        except Exception as exc:
            logger.warning("AkShare 股票列表同步失败: %s", exc)
            return

        if df is None or df.empty:
            logger.warning("AkShare 返回空股票列表")
            return

        loaded = 0
        for row in df.itertuples(index=False):
            values = list(row)
            if len(values) < 2:
                continue
            code, name = str(values[0]).strip(), str(values[1]).strip()
            if code and name:
                self._register(code, name, [name])
                loaded += 1

        logger.info("AkShare 同步完成: %d 条", loaded)

    def _load_from_tushare(self) -> None:
        try:
            import tushare as ts
        except ImportError:
            logger.warning("未安装 tushare，跳过 Tushare 股票列表同步")
            return

        logger.info("从 Tushare 同步股票基础信息...")
        try:
            pro = ts.pro_api(TUSHARE_TOKEN)
            df = pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name",
            )
        except Exception as exc:
            logger.warning("Tushare 股票列表同步失败: %s", exc)
            return

        if df is None or df.empty:
            logger.warning("Tushare 返回空股票列表")
            return

        loaded = 0
        for row in df.itertuples(index=False):
            ts_code = str(getattr(row, "ts_code", "") or "").strip()
            name = str(getattr(row, "name", "") or "").strip()
            if ts_code and name:
                self._register(ts_code, name, [name])
                loaded += 1

        logger.info("Tushare 同步完成: %d 条", loaded)

    def lookup_by_id(self, entity_id: str) -> dict[str, str] | None:
        normalized = _normalize_entity_id(entity_id)
        record = self._by_id.get(normalized)
        if record is None:
            return None
        return {"entity_id": record.entity_id, "entity_name": record.entity_name}

    def lookup_by_alias(self, text: str) -> dict[str, str] | None:
        for alias in self._sorted_aliases:
            if alias and alias in text:
                record = self._by_id.get(self._alias_to_id[alias])
                if record:
                    return {
                        "entity_id": record.entity_id,
                        "entity_name": record.entity_name,
                    }
        return None

    def lookup_by_name(self, name: str) -> dict[str, str] | None:
        text = str(name or "").strip()
        if not text:
            return None

        if text in self._alias_to_id:
            return self.lookup_by_id(self._alias_to_id[text])

        return self.lookup_by_alias(text)

    def resolve(self, *, entity_name: str = "", entity_id: str = "") -> dict[str, str]:
        """解析并补全 entity_name / entity_id，找不到则返回 UNKNOWN。"""
        name = str(entity_name or "").strip()
        eid = _normalize_entity_id(entity_id)

        if eid:
            found = self.lookup_by_id(eid)
            if found:
                return found
            if name:
                return {"entity_id": eid, "entity_name": name}
            return {"entity_id": eid, "entity_name": UNKNOWN_ENTITY_NAME}

        if name:
            found = self.lookup_by_name(name)
            if found:
                return found
            return {"entity_id": UNKNOWN_ENTITY_ID, "entity_name": UNKNOWN_ENTITY_NAME}

        return {"entity_id": UNKNOWN_ENTITY_ID, "entity_name": UNKNOWN_ENTITY_NAME}

    def count(self) -> int:
        return len(self._by_id)


def get_stock_registry() -> StockRegistry:
    """获取 StockRegistry 单例。"""
    return StockRegistry()


def reload_stock_registry() -> StockRegistry:
    """强制重建注册表（主要用于测试）。"""
    with StockRegistry._lock:
        StockRegistry._instance = None
    return get_stock_registry()
