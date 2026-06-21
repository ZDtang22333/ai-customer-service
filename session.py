"""
会话管理器
==========
按 user_id 管理每个用户的对话历史。

功能：
- 每个用户独立的对话历史
- 自动限制历史长度（防止内存溢出）
- 支持 LangChain 消息格式转换
"""

from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class SessionManager:
    """
    多用户会话管理器。

    数据结构：
    {
        "user_001": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "有什么可以帮您？"},
        ],
        "user_002": [...]
    }
    """

    def __init__(self, max_history: int = 20):
        """
        Args:
            max_history: 每个用户最多保留的消息数
        """
        self.sessions: Dict[str, List[dict]] = {}
        self.max_history = max_history

    def add_message(self, user_id: str, role: str, content: str):
        """
        添加一条消息。

        Args:
            user_id: 用户ID
            role: 角色，"user" 或 "assistant"
            content: 消息内容
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = []

        self.sessions[user_id].append({
            "role": role,
            "content": content,
        })

        # 超过上限，删除最早的消息
        if len(self.sessions[user_id]) > self.max_history:
            self.sessions[user_id] = self.sessions[user_id][-self.max_history:]

    def get_history(self, user_id: str) -> List[dict]:
        """获取用户的对话历史"""
        return self.sessions.get(user_id, [])

    def get_langchain_messages(self, user_id: str) -> List[BaseMessage]:
        """
        获取 LangChain 格式的消息列表。
        用于传给 RAG 链的 chat_history 参数。
        """
        history = self.get_history(user_id)
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def clear(self, user_id: str):
        """清空用户的对话历史"""
        if user_id in self.sessions:
            del self.sessions[user_id]

    def get_all_user_ids(self) -> List[str]:
        """获取所有在线用户ID"""
        return list(self.sessions.keys())
