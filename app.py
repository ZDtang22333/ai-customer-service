"""
智能电商客服系统 - 完整版
===========================
整合所有功能：
- RAG 知识库检索
- 混合检索（BM25 + 语义搜索 + 重排序）
- Agent 工具调用（订单查询、退款等）
- 多轮对话记忆
- Web 界面（Gradio）

运行方式：python app.py
然后浏览器打开 http://127.0.0.1:7860
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gradio as gr
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION, RETRIEVAL_K,
    validate_config,
)


# ============================================
# 第一部分：模拟业务数据
# ============================================

ORDERS = {
    "12345": {
        "order_id": "12345",
        "product": "AirPods Pro 2",
        "amount": 1899,
        "status": "已签收",
        "sign_date": "2026-06-10",
        "user": "小明",
    },
    "67890": {
        "order_id": "67890",
        "product": "小米笔记本Pro 16",
        "amount": 5999,
        "status": "运输中",
        "sign_date": None,
        "user": "小明",
    },
}

LOGISTICS = {
    "12345": {"company": "顺丰", "tracking": "SF1234567890", "status": "已签收"},
    "67890": {"company": "京东物流", "tracking": "JD9876543210", "status": "运输中，预计明天到达"},
}


# ============================================
# 第二部分：工具定义
# ============================================

@tool
def query_order(order_id: str) -> str:
    """查询订单状态。输入订单号，返回订单详情。"""
    order = ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}，请检查订单号是否正确。"
    return (
        f"订单 {order['order_id']} 详情:\n"
        f"- 商品: {order['product']}\n"
        f"- 金额: {order['amount']}元\n"
        f"- 状态: {order['status']}\n"
        f"- 签收日期: {order['sign_date'] or '未签收'}"
    )


@tool
def apply_refund(order_id: str, reason: str = "不想要了") -> str:
    """提交退款申请。输入订单号和退款原因。"""
    order = ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}，无法提交退款。"
    if order["status"] != "已签收":
        return f"订单 {order_id} 状态为 {order['status']}，未签收的订单请申请取消订单。"
    return (
        f"退款申请已提交:\n"
        f"- 订单号: {order_id}\n"
        f"- 商品: {order['product']}\n"
        f"- 金额: {order['amount']}元\n"
        f"- 原因: {reason}\n"
        f"- 预计 3-5 个工作日退款到原支付方式"
    )


@tool
def check_refund_policy(order_id: str) -> str:
    """检查订单是否符合退款条件。"""
    from datetime import datetime
    order = ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}。"
    if order["status"] != "已签收":
        return f"订单 {order_id} 尚未签收，可以直接取消订单。"
    if order["sign_date"]:
        sign_date = datetime.strptime(order["sign_date"], "%Y-%m-%d")
        days = (datetime.now() - sign_date).days
        if days <= 7:
            return f"订单 {order_id} 签收 {days} 天，符合7天无理由退款条件。"
        elif days <= 15:
            return f"订单 {order_id} 签收 {days} 天，超过7天无理由退款期限。如有质量问题，可申请售后检测。"
        else:
            return f"订单 {order_id} 签收 {days} 天，已超过退款期限。建议联系人工客服咨询维修方案。"
    return "无法判断退款资格，请联系人工客服。"


@tool
def query_logistics(order_id: str) -> str:
    """查询物流信息。输入订单号，返回物流状态。"""
    logistics = LOGISTICS.get(order_id)
    if not logistics:
        return f"未找到订单 {order_id} 的物流信息。"
    return (
        f"物流信息:\n"
        f"- 快递公司: {logistics['company']}\n"
        f"- 运单号: {logistics['tracking']}\n"
        f"- 状态: {logistics['status']}"
    )


@tool
def transfer_human(reason: str = "用户请求") -> str:
    """转接人工客服。当问题无法自动解决时使用。"""
    return f"已为您转接人工客服，转接原因: {reason}。请稍候，人工客服即将接入。"


TOOLS = [query_order, apply_refund, check_refund_policy, query_logistics, transfer_human]


# ============================================
# 第三部分：混合检索
# ============================================

class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self, documents: List[Document], k: int = 3):
        import jieba
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self.k = k
        self.corpus_tokens = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def invoke(self, query: str) -> List[Document]:
        import jieba
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-self.k:][::-1]
        return [self.documents[i] for i in top_indices]


class HybridRetriever:
    """混合检索器：BM25 + 语义搜索"""

    def __init__(self, vectorstore, documents, k=3):
        self.k = k
        self.bm25 = BM25Retriever(documents, k=k * 2)
        self.semantic = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    def invoke(self, query: str) -> List[Document]:
        bm25_docs = self.bm25.invoke(query)
        semantic_docs = self.semantic.invoke(query)

        seen = set()
        merged = []
        for doc in semantic_docs + bm25_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append(doc)
        return merged[:self.k]


class RerankRetriever:
    """重排序检索器"""

    def __init__(self, base_retriever, k=3):
        from sentence_transformers import CrossEncoder

        self.base = base_retriever
        self.k = k
        print("  正在加载重排序模型...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        print("  重排序模型加载完成")

    def invoke(self, query: str) -> List[Document]:
        candidates = self.base.invoke(query)[:10]
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:self.k]]


# ============================================
# 第四部分：初始化所有组件
# ============================================

print("=" * 50)
print("正在初始化客服系统...")
print("=" * 50)

# 1. LLM
print("[1/5] 加载 LLM...")
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# 2. Embedding
print("[2/5] 加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 3. 向量数据库
print("[3/5] 加载向量数据库...")
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION,
)

# 4. 加载文档（用于 BM25）
print("[4/5] 构建混合检索...")
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    "knowledge", glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

hybrid = HybridRetriever(vectorstore, chunks, k=RETRIEVAL_K)
retriever = RerankRetriever(hybrid, k=RETRIEVAL_K)

# 5. Agent
print("[5/5] 创建 Agent...")
agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=(
        "你是数码星球的客服助手小智。\n"
        "你可以调用工具来帮助用户处理订单、退款、物流等问题。\n"
        "如果用户的问题需要查订单或处理业务，一定要调用工具，不要猜测。\n"
        "如果用户只是闲聊或问产品问题，直接回答即可。"
    ),
)

print("=" * 50)
print("初始化完成！启动 Web 服务...")
print("=" * 50)


# ============================================
# 第五部分：RAG 链（回答产品/政策问题）
# ============================================

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是「数码星球旗舰店」的智能客服助手小智。\n"
     "根据参考资料回答问题，不要编造信息。\n"
     "回答简洁友好，不超过100字。\n\n"
     "参考资料:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

rag_chain = rag_prompt | llm | StrOutputParser()


# ============================================
# 第六部分：意图判断 + 路由
# ============================================

def classify_intent(message: str) -> str:
    """
    判断用户意图，决定走哪条链。

    返回:
        "agent"  → 业务操作（查订单、退款、物流）
        "rag"    → 知识问答（产品、政策）
        "chat"   → 闲聊
    """
    # 关键词匹配（简单但有效）
    agent_keywords = ["订单", "退款", "退货", "物流", "快递", "发货", "签收", "取消"]
    for keyword in agent_keywords:
        if keyword in message:
            return "agent"

    return "rag"  # 默认走 RAG


def format_docs(docs):
    """把文档列表格式化成字符串"""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ============================================
# 第七部分：聊天函数（核心）
# ============================================

def chat(message, history):
    """
    Gradio 调用的聊天函数。

    流程：
    1. 判断意图
    2. 业务问题 → Agent
    3. 知识问题 → RAG
    4. 返回回答
    """
    # 构造对话历史
    chat_history = []
    for user_msg, ai_msg in history:
        chat_history.append(HumanMessage(content=user_msg))
        chat_history.append(AIMessage(content=ai_msg))

    # 意图判断
    intent = classify_intent(message)

    if intent == "agent":
        # 走 Agent（业务操作）
        result = agent.invoke({
            "messages": [HumanMessage(content=message)],
        })
        response = result["messages"][-1].content
    else:
        # 走 RAG（知识问答）
        docs = retriever.invoke(message)
        context = format_docs(docs)
        response = rag_chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": message,
        })

    return response


# ============================================
# 第八部分：Web 界面
# ============================================

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
    validate_config()
    demo.launch()
