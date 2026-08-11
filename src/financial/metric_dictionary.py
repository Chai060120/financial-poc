"""
财务指标名称映射与章节优先级。

避免「每股净资产」误匹配「净资产收益率」等歧义。
"""

from __future__ import annotations

# metric_key -> (aliases, preferred_sections, exclude_if_contains)
METRIC_DEFINITIONS: dict[str, dict] = {
    "revenue": {
        "aliases": ("营业收入", "营业总收入", "收入合计", "营业入"),
        "sections": ("主要会计数据", "利润表", "管理层讨论"),
        "exclude": ("每股", "增长率", "同比"),
    },
    "operating_profit": {
        "aliases": ("营业利润", "经营利润"),
        "sections": ("主要会计数据", "利润表"),
        "exclude": ("率", "同比", "增长"),
    },
    "net_profit": {
        "aliases": ("净利润", "利润总额"),
        "sections": ("主要会计数据", "利润表"),
        "exclude": ("归属于", "扣非", "每股", "率", "同比"),
    },
    "attributable_profit": {
        "aliases": (
            "归属于母公司股东的净利润",
            "归母净利润",
            "归属于上市公司股东",
            "归属于本行股东的净利润",
            "归属于本行普通股股东的净利润",
        ),
        "sections": ("主要会计数据", "利润表"),
        "exclude": ("扣非", "每股", "率", "同比", "增长"),
    },
    "eps": {
        "aliases": (
            "基本每股收益",
            "稀释每股收益",
            "每股收益",
            "归属于本行普通股股东的基本每股收益",
            "归属于本行普通股股东的稀释每股收益",
        ),
        "sections": ("主要会计数据", "利润表"),
        "exclude": ("扣非", "净资产", "每股净资产", "每股经营", "每股现金"),
    },
    "bvps": {
        "aliases": (
            "归属于母公司股东的每股净资产",
            "每股净资产",
            "归属于本行普通股股东的每股净资产",
        ),
        "sections": ("主要会计数据", "资产负债表"),
        "exclude": ("收益率", "净资产收益率"),
    },
    "roe": {
        "aliases": ("加权平均净资产收益率", "净资产收益率"),
        "sections": ("主要会计数据", "利润表"),
        "exclude": ("扣非", "每股", "总资产"),
    },
    "total_assets": {
        "aliases": ("资产总计", "资产总额", "总资产"),
        "sections": ("主要会计数据", "资产负债表"),
        "exclude": ("每股", "率", "同比", "增长"),
    },
    "total_equity": {
        "aliases": (
            "归属于母公司股东权益合计",
            "归属于母公司所有者权益",
            "归属于母公司股东权益",
            "股东权益合计",
            "归属于本行股东权益",
        ),
        "sections": ("主要会计数据", "资产负债表"),
        "exclude": ("每股", "率", "同比"),
    },
    "cash_flow_operating": {
        "aliases": (
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "经营现金流",
        ),
        "sections": ("主要会计数据", "现金流量表"),
        "exclude": ("每股", "率", "同比"),
    },
}

# 向后兼容 valuation / report_card 的旧 key
LEGACY_KEY_MAP: dict[str, str] = {
    "revenue": "revenue",
    "operating_profit": "operating_profit",
    "net_profit": "net_profit",
    "attributable_profit": "net_profit",
    "eps": "eps",
    "bvps": "bvps",
    "roe": "roe",
    "total_assets": "total_assets",
    "total_equity": "total_equity",
    "cash_flow_operating": "cash_flow_operating",
}

PRIORITY_SECTIONS: tuple[str, ...] = (
    "主要会计数据",
    "利润表",
    "资产负债表",
    "现金流量表",
    "管理层讨论",
)

TABLE_CONTEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("主要会计数据和财务指标", "主要会计数据和财务指标"),
    ("主要会计数据及财务指标", "主要会计数据及财务指标"),
    ("近三年主要会计数据", "近三年主要会计数据"),
    ("主要财务指标", "主要财务指标"),
    ("财务摘要", "财务摘要"),
    ("合并资产负债表", "合并资产负债表"),
    ("合并利润表", "合并利润表"),
    ("合并现金流量表", "合并现金流量表"),
)
