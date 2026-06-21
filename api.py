"""
FastAPI 客服接口
================
提供 RESTful API + 流式输出（SSE）+ 多用户会话管理。

接口列表：
- POST /chat          同步聊天
- POST /chat/stream   流式聊天（SSE）
- GET  /history/{uid} 获取对话历史
- DELETE /history/{uid} 清空对话历史
- GET  /health        健康检查

启动方式：python api.py
API 文档：http://127.0.0.1:8000/docs
"""

import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core import CustomerService
from session import SessionManager
from config import validate_config


# ============================================
# 数据模型
# ============================================

class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: str = "default"  # 用户ID，默认为 "default"
    message: str               # 用户消息

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "message": "笔记本电池能用多久？"
            }
        }


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str   # 回复内容
    intent: str     # 意图："agent" 或 "rag"


# ============================================
# 初始化
# ============================================

# 创建 FastAPI 应用
app = FastAPI(
    title="智能客服 API",
    description="基于 LangChain + RAG + Agent 的智能电商客服系统",
    version="1.0.0",
)

# CORS 配置（允许前端跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化客服系统和会话管理器
print("正在启动 FastAPI 服务...")
validate_config()
cs = CustomerService()
session_manager = SessionManager(max_history=20)
print("FastAPI 服务已就绪！")


# ============================================
# 接口定义
# ============================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "智能客服 API",
        "online_users": len(session_manager.get_all_user_ids()),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    同步聊天接口。

    流程：
    1. 获取用户的对话历史
    2. 调用客服核心处理
    3. 保存对话记录
    4. 返回回复
    """
    # 获取对话历史
    history = session_manager.get_langchain_messages(request.user_id)

    # 调用客服核心
    intent = cs.classify_intent(request.message)
    response = cs.chat(request.message, history)

    # 保存对话记录
    session_manager.add_message(request.user_id, "user", request.message)
    session_manager.add_message(request.user_id, "assistant", response)

    return ChatResponse(response=response, intent=intent)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口（SSE）。

    返回 Server-Sent Events 格式：
    data: {"token": "你"}
    data: {"token": "好"}
    ...
    data: {"done": true, "response": "你好，有什么可以帮您？"}
    """
    # 获取对话历史
    history = session_manager.get_langchain_messages(request.user_id)

    # 意图判断
    intent = cs.classify_intent(request.message)

    async def generate():
        """生成 SSE 流"""
        full_response = ""

        # 流式生成
        for token in cs.chat_stream(request.message, history):
            full_response += token
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        # 保存对话记录
        session_manager.add_message(request.user_id, "user", request.message)
        session_manager.add_message(request.user_id, "assistant", full_response)

        # 结束标记
        yield f"data: {json.dumps({'done': True, 'response': full_response, 'intent': intent}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """获取用户的对话历史"""
    history = session_manager.get_history(user_id)
    return {
        "user_id": user_id,
        "message_count": len(history),
        "messages": history,
    }


@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """清空用户的对话历史"""
    session_manager.clear(user_id)
    return {
        "status": "ok",
        "message": f"用户 {user_id} 的对话历史已清空",
    }


@app.get("/users")
async def get_users():
    """获取所有在线用户"""
    return {
        "users": session_manager.get_all_user_ids(),
        "count": len(session_manager.get_all_user_ids()),
    }


# ============================================
# 启动
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
