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
- GET  /stats         统计信息

启动方式：python api.py
API 文档：http://127.0.0.1:8000/docs
"""

import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core import CustomerService, CustomerServiceError
from session import SessionManager
from config import validate_config
from logger import get_logger

logger = get_logger("api")


# ============================================
# 数据模型
# ============================================

class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: str = "default"
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "message": "笔记本电池能用多久？"
            }
        }


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    intent: str
    elapsed: float  # 耗时（秒）


# ============================================
# 初始化
# ============================================

app = FastAPI(
    title="智能客服 API",
    description="基于 LangChain + RAG + Agent 的智能电商客服系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("正在启动 FastAPI 服务...")
validate_config()

try:
    cs = CustomerService()
    session_manager = SessionManager(max_history=20)
    logger.info("FastAPI 服务已就绪！")
except CustomerServiceError as e:
    logger.error(f"服务启动失败: {e}")
    raise


# ============================================
# 请求计数（简单统计）
# ============================================

request_count = 0
error_count = 0


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
        "request_count": request_count,
        "error_count": error_count,
    }


@app.get("/stats")
async def get_stats():
    """统计信息"""
    return {
        "total_requests": request_count,
        "total_errors": error_count,
        "error_rate": f"{error_count / max(request_count, 1) * 100:.1f}%",
        "online_users": len(session_manager.get_all_user_ids()),
        "users": session_manager.get_all_user_ids(),
        "cache": cs.cache.stats,
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
    global request_count, error_count
    request_count += 1
    start_time = time.time()

    logger.info(f"[{request.user_id}] 收到消息: {request.message}")

    try:
        # 获取对话历史
        history = session_manager.get_langchain_messages(request.user_id)

        # 调用客服核心
        intent = cs.classify_intent(request.message)
        response = cs.chat(request.message, history)

        # 保存对话记录
        session_manager.add_message(request.user_id, "user", request.message)
        session_manager.add_message(request.user_id, "assistant", response)

        elapsed = time.time() - start_time
        logger.info(f"[{request.user_id}] 回复完成: {elapsed:.2f}s, 意图={intent}")

        return ChatResponse(response=response, intent=intent, elapsed=elapsed)

    except Exception as e:
        error_count += 1
        elapsed = time.time() - start_time
        logger.error(f"[{request.user_id}] 请求失败: {e} ({elapsed:.2f}s)", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


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
    global request_count, error_count
    request_count += 1
    start_time = time.time()

    logger.info(f"[{request.user_id}] 流式请求: {request.message}")

    try:
        history = session_manager.get_langchain_messages(request.user_id)
        intent = cs.classify_intent(request.message)

        async def generate():
            full_response = ""

            for token in cs.chat_stream(request.message, history):
                full_response += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            # 保存对话记录
            session_manager.add_message(request.user_id, "user", request.message)
            session_manager.add_message(request.user_id, "assistant", full_response)

            elapsed = time.time() - start_time
            logger.info(f"[{request.user_id}] 流式完成: {elapsed:.2f}s")

            yield f"data: {json.dumps({'done': True, 'response': full_response, 'intent': intent, 'elapsed': elapsed}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        error_count += 1
        elapsed = time.time() - start_time
        logger.error(f"[{request.user_id}] 流式失败: {e} ({elapsed:.2f}s)", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """获取用户的对话历史"""
    history = session_manager.get_history(user_id)
    logger.debug(f"[{user_id}] 获取历史: {len(history)} 条")
    return {
        "user_id": user_id,
        "message_count": len(history),
        "messages": history,
    }


@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """清空用户的对话历史"""
    session_manager.clear(user_id)
    logger.info(f"[{user_id}] 历史已清空")
    return {"status": "ok", "message": f"用户 {user_id} 的对话历史已清空"}


@app.delete("/cache")
async def clear_cache():
    """清空缓存"""
    cs.cache.clear()
    logger.info("缓存已清空")
    return {"status": "ok", "message": "缓存已清空"}


# ============================================
# 启动
# ============================================

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 uvicorn 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
