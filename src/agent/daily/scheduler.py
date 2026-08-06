"""Daily Agent 定时调度。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import schedule

from config import AGENT_SCHEDULE_TIME, setup_logging
from src.agent.daily.runner import run_daily_agent

logger = setup_logging(__name__)


def _scheduled_job() -> None:
    logger.info("定时任务触发: Daily Agent")
    try:
        ctx = run_daily_agent()
        if ctx.success:
            logger.info("定时任务成功: %s", ctx.report_path)
        else:
            logger.error("定时任务完成但有错误: %s", ctx.errors)
    except Exception as exc:
        logger.exception("定时任务失败: %s", exc)


def start_scheduler(*, schedule_time: str = AGENT_SCHEDULE_TIME, run_immediately: bool = False) -> None:
    """
    启动每日定时调度（阻塞运行）。

    Args:
        schedule_time: 每日执行时间 HH:MM
        run_immediately: 启动时是否先执行一次
    """
    logger.info("注册 Daily Agent 定时任务: 每天 %s", schedule_time)
    schedule.every().day.at(schedule_time).do(_scheduled_job)

    if run_immediately:
        _scheduled_job()

    logger.info("调度器已启动，等待下次执行...")
    while True:
        schedule.run_pending()
        time.sleep(30)


def run_once_and_exit(**kwargs) -> None:
    """立即执行一次 Daily Agent。"""
    ctx = run_daily_agent(**kwargs)
    if not ctx.success:
        raise SystemExit(1)
