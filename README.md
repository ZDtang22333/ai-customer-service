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
- 流式输出

### Web 界面
- Gradio 聊天界面
- 示例问题引导
- 自动保存对话历史

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | 小米 MiMo-v2.5-pro |
| Embedding | BAAI/bge-small-zh-v1.5（本地） |
| 向量数据库 | ChromaDB |
| 关键词检索 | BM25（jieba 分词） |
| 重排序 | BAAI/bge-reranker-base |
| Agent | LangChain create_agent |
| Web | Gradio |

## 项目结构

```
├── app.py                 # 主程序
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

# 4. 构建向量数据库
python -c "from app import *"

# 5. 启动 Web 界面
python app.py
# 浏览器打开 http://127.0.0.1:7860
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
