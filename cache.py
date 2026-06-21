"""
缓存模块
========
对相同问题缓存结果，减少重复的 LLM 调用。

功能：
- 基于问题内容的缓存
- 可配置过期时间
- 缓存命中率统计
- 自动清理过期缓存

使用方式：
    from cache import ResponseCache
    cache = ResponseCache(ttl=300)  # 5分钟过期

    # 检查缓存
    cached = cache.get("笔记本电池能用多久")
    if cached:
        return cached

    # 缓存结果
    cache.set("笔记本电池能用多久", "续航8-12小时...")
"""

import hashlib
import time
from typing import Optional, Dict
from logger import get_logger

logger = get_logger("cache")


class ResponseCache:
    """
    基于问题内容的响应缓存。

    缓存策略：
    - key = 问题内容的 MD5 哈希
    - value = {"response": 回复, "intent": 意图, "timestamp": 时间戳}
    - 超过 ttl 秒自动过期
    """

    def __init__(self, ttl: int = 300, max_size: int = 1000):
        """
        Args:
            ttl: 缓存过期时间（秒），默认 300 秒（5分钟）
            max_size: 最大缓存条数
        """
        self.cache: Dict[str, dict] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.hits = 0  # 缓存命中次数
        self.misses = 0  # 缓存未命中次数

    def _make_key(self, message: str) -> str:
        """生成缓存 key（MD5 哈希）"""
        # 去掉首尾空格，转小写，避免大小写不同导致缓存未命中
        normalized = message.strip().lower()
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def get(self, message: str) -> Optional[dict]:
        """
        获取缓存。

        Args:
            message: 用户消息

        Returns:
            缓存的响应 {"response": ..., "intent": ...} 或 None
        """
        key = self._make_key(message)

        if key in self.cache:
            entry = self.cache[key]
            # 检查是否过期
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                logger.debug(f"缓存命中: '{message[:30]}...' (命中率: {self.hit_rate})")
                return {
                    "response": entry["response"],
                    "intent": entry["intent"],
                }
            else:
                # 过期，删除
                del self.cache[key]
                logger.debug(f"缓存过期: '{message[:30]}...'")

        self.misses += 1
        return None

    def set(self, message: str, response: str, intent: str):
        """
        设置缓存。

        Args:
            message: 用户消息
            response: 回复内容
            intent: 意图
        """
        # 缓存满了，删除最早的一条
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
            logger.debug("缓存已满，删除最早条目")

        key = self._make_key(message)
        self.cache[key] = {
            "response": response,
            "intent": intent,
            "timestamp": time.time(),
        }
        logger.debug(f"缓存已设置: '{message[:30]}...'")

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("缓存已清空")

    def cleanup_expired(self):
        """清理过期缓存"""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry["timestamp"] >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 条过期缓存")

    @property
    def hit_rate(self) -> str:
        """缓存命中率"""
        total = self.hits + self.misses
        if total == 0:
            return "0%"
        return f"{self.hits / total * 100:.1f}%"

    @property
    def size(self) -> int:
        """当前缓存条数"""
        return len(self.cache)

    @property
    def stats(self) -> dict:
        """缓存统计"""
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "ttl": self.ttl,
        }
