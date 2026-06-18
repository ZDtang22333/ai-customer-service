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
├── app.py                 # 主程序（整合版）
├── config.py              # 配置管理
├── hybrid_retriever.py    # 混合检索模块
├── agent_tools.py         # Agent 工具定义
├── 01_llm_basics.py       # 学习：LLM 基础
├── 02_prompt_template.py  # 学习：Prompt 模板
├── 03_text_splitting.py   # 学习：文本切分与向量化
├── 04_rag_chain.py        # 学习：RAG 检索链
├── 05_memory.py           # 学习：对话记忆
├── 06_full_chatbot.py     # 学习：完整客服
├── 07_web_ui.py           # 学习：Web 界面
├── cheatsheet.md          # 速查表
├── knowledge/             # 知识库文档
│   ├── product_faq.txt
│   ├── return_policy.txt
│   └── company_intro.txt
├── requirements.txt       # 依赖列表
└── .env                   # 环境变量（不提交）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3. 构建向量数据库
python 03_text_splitting.py

# 4. 启动 Web 界面
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

## 学习路径

```
01_llm_basics.py      → LLM 调用基础
02_prompt_template.py → Prompt 模板与 LCEL
03_text_splitting.py  → 文本切分与向量化
04_rag_chain.py       → RAG 检索链
05_memory.py          → 对话记忆
06_full_chatbot.py    → 完整客服
07_web_ui.py          → Web 界面
hybrid_retriever.py   → 混合检索
agent_tools.py        → Agent 工具调用
app.py                → 最终整合版
```

## License

MIT
