"""财报分析、估值与实时对比模块。"""

from src.analysis.valuation import ValuationResult, analyze_valuation

__all__ = [
    "ValuationResult",
    "analyze_valuation",
    "ComparisonResult",
    "analyze_market_comparison",
]


def __getattr__(name: str):
    if name in {"ComparisonResult", "analyze_market_comparison"}:
        from src.analysis.market_compare import ComparisonResult, analyze_market_comparison

        return ComparisonResult if name == "ComparisonResult" else analyze_market_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
