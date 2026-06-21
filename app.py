"""
Gradio 界面
===========
调用 core.py 的客服核心，提供网页聊天界面。

运行方式：python app.py
然后浏览器打开 http://127.0.0.1:7860
"""

import gradio as gr
from core import CustomerService
from session import SessionManager
from config import validate_config
from logger import get_logger

logger = get_logger("app")

# 初始化
validate_config()
cs = CustomerService()
session_manager = SessionManager(max_history=20)


def chat(message, history):
    """
    Gradio 调用的聊天函数。

    Args:
        message: 用户消息
        history: Gradio 格式的对话历史 [[user, ai], ...]

    Returns:
        回复内容
    """
    # 用固定 user_id（Gradio 单用户）
    user_id = "gradio_user"

    # 获取对话历史
    chat_history = session_manager.get_langchain_messages(user_id)

    # 调用核心
    response = cs.chat(message, chat_history)

    # 保存对话记录
    session_manager.add_message(user_id, "user", message)
    session_manager.add_message(user_id, "assistant", response)

    return response


# 创建界面
demo = gr.ChatInterface(
    fn=chat,
    title="数码星球智能客服",
    description="有什么可以帮您的？我是客服小智~",
    examples=[
        "笔记本电池能用多久？",
        "怎么退货？",
        "帮我查订单 12345",
        "耳机降噪效果怎么样？",
        "帮我申请退款",
    ],
)

if __name__ == "__main__":
    demo.launch()
