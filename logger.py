"""
日志系统
========
统一的日志管理，替代 print。

功能：
- 支持不同日志级别（DEBUG/INFO/WARNING/ERROR）
- 同时输出到控制台和文件
- 带时间戳和模块名
- 按日期自动分割日志文件

使用方式：
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("初始化完成")
    logger.error("调用失败", exc_info=True)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


# 日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取一个配置好的 logger。

    Args:
        name: logger 名称，通常用 __name__
        level: 日志级别，默认 INFO

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 日志格式
    # [2026-06-18 20:30:15] [INFO] [core] 初始化完成
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 handler（按大小轮转，最大 10MB，保留 5 个备份）
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, f"app_{today}.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
