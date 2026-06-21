# 智能电商客服系统

基于 LangChain + RAG + Agent 构建的智能客服系统，支持知识库问答、订单业务处理、多轮对话记忆。

## 功能特性

### 知识库问答（RAG）
- 支持 txt 文档自动加载、切分、向量化
- 混合检索：BM25 关键词搜索 + 语义搜索
- 重排序：CrossEncoder 精排，提升准确率

### 业务处理（Agent）
- 订单查询：输入订单号查询状态
- 退款申请：提交退款申请
- 退款检查：自动判断是否符合退款条件
- 物流追踪：查询物流信息
- 转人工：无法处理时转接人工客服

### 对话管理
- 多轮对话记忆
- 意图识别与路由（知识问答 / 业务操作）
- 流式输出（SSE）

### Web 界面
- Gradio 聊天界面
- FastAPI RESTful API + 流式接口

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | 小米 MiMo-v2.5-pro |
| Embedding | BAAI/bge-small-zh-v1.5（本地） |
| 向量数据库 | ChromaDB |
| 关键词检索 | BM25（jieba 分词） |
| 重排序 | BAAI/bge-reranker-base |
| Agent | LangChain create_agent |
| Web | Gradio / FastAPI |

## 项目结构

```
├── app.py                 # Gradio 界面
├── api.py                 # FastAPI 接口
├── core.py                # 核心逻辑（Gradio 和 FastAPI 共用）
├── session.py             # 多用户会话管理
├── config.py              # 配置管理
├── hybrid_retriever.py    # 混合检索模块
├── agent_tools.py         # Agent 工具定义
├── knowledge/             # 知识库文档
│   ├── product_faq.txt
│   ├── return_policy.txt
│   └── company_intro.txt
└── requirements.txt       # 依赖列表
```

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/ZDtang22333/ai-customer-service.git
cd ai-customer-service

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
# 创建 .env 文件，填入以下内容：
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
CHAT_MODEL=mimo-v2.5-pro
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 4. 启动 Gradio 界面
python app.py
# 浏览器打开 http://127.0.0.1:7860

# 或者启动 FastAPI 接口
python api.py
# API 文档打开 http://127.0.0.1:8000/docs
```

## API 接口文档

### 同步聊天

```bash
POST /chat

请求：
{
    "user_id": "user_001",
    "message": "笔记本电池能用多久？"
}

响应：
{
    "response": "根据产品信息，笔记本电池续航8-12小时...",
    "intent": "rag"
}
```

### 流式聊天（SSE）

```bash
POST /chat/stream

请求：
{
    "user_id": "user_001",
    "message": "笔记本电池能用多久？"
}

响应（Server-Sent Events）：
data: {"token": "根"}
data: {"token": "据"}
data: {"token": "产品"}
...
data: {"done": true, "response": "根据产品信息...", "intent": "rag"}
```

### 获取对话历史

```bash
GET /history/{user_id}

响应：
{
    "user_id": "user_001",
    "message_count": 4,
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "有什么可以帮您？"}
    ]
}
```

### 清空对话历史

```bash
DELETE /history/{user_id}

响应：
{
    "status": "ok",
    "message": "用户 user_001 的对话历史已清空"
}
```

### 健康检查

```bash
GET /health

响应：
{
    "status": "ok",
    "service": "智能客服 API",
    "online_users": 3
}
```

## 核心流程

```
用户提问
   ↓
意图识别（关键词匹配）
   ↓
   ├── 业务问题 → Agent → 调用工具 → 返回结果
   │
   └── 知识问题 → 混合检索（BM25 + 语义 + 重排序）
                    ↓
                  相关文档 → LLM 生成回答
                    ↓
                  返回回答
```

## 检索优化对比

| 方案 | Top-3 命中率 | 适用场景 |
|------|-------------|----------|
| 纯语义搜索 | ~65% | 语义相似但关键词不同 |
| BM25 关键词 | ~70% | 精确关键词匹配 |
| 混合检索 | ~80% | 综合场景 |
| 混合 + 重排序 | ~89% | 最佳效果 |

## License

MIT
