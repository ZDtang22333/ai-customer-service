"""
客服核心逻辑
============
从 app.py 提取的核心模块，供 Gradio 和 FastAPI 共用。

包含：
- 组件初始化（LLM、Embedding、向量库、Agent）
- RAG 链
- 意图判断
- 聊天处理
- 错误处理 + 日志
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
from typing import List, Generator
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.agents import create_agent

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION, RETRIEVAL_K,
)
from agent_tools import TOOLS
from hybrid_retriever import HybridRetriever, RerankRetriever
from logger import get_logger
from cache import ResponseCache

logger = get_logger("core")


# ============================================
# 自定义异常
# ============================================

class CustomerServiceError(Exception):
    """客服系统基础异常"""
    pass


class LLMError(CustomerServiceError):
    """LLM 调用异常"""
    pass


class RetrievalError(CustomerServiceError):
    """检索异常"""
    pass


class AgentError(CustomerServiceError):
    """Agent 调用异常"""
    pass


# ============================================
# 初始化组件
# ============================================

class CustomerService:
    """客服系统核心类"""

    def __init__(self):
        logger.info("=" * 50)
        logger.info("正在初始化客服系统...")
        logger.info("=" * 50)

        try:
            # 1. LLM
            logger.info("[1/5] 加载 LLM...")
            self.llm = ChatOpenAI(
                model=CHAT_MODEL,
                temperature=0.0,
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            )
            logger.info(f"  LLM 加载完成: {CHAT_MODEL}")

            # 2. Embedding
            logger.info("[2/5] 加载 Embedding 模型...")
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info(f"  Embedding 加载完成: {EMBEDDING_MODEL}")

            # 3. 向量数据库
            logger.info("[3/5] 加载向量数据库...")
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=CHROMA_COLLECTION,
            )
            logger.info("  向量数据库加载完成")

            # 4. 混合检索
            logger.info("[4/5] 构建混合检索...")
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
            logger.info(f"  文档切分完成: {len(documents)} 个文档 → {len(chunks)} 个文本块")

            hybrid = HybridRetriever(self.vectorstore, chunks, k=RETRIEVAL_K)
            self.retriever = RerankRetriever(hybrid, k=RETRIEVAL_K)
            logger.info("  混合检索构建完成")

            # 5. Agent
            logger.info("[5/5] 创建 Agent...")
            self.agent = create_agent(
                model=self.llm,
                tools=TOOLS,
                system_prompt=(
                    "你是数码星球的客服助手小智。\n"
                    "你可以调用工具来帮助用户处理订单、退款、物流等问题。\n"
                    "如果用户的问题需要查订单或处理业务，一定要调用工具，不要猜测。\n"
                    "如果用户只是闲聊或问产品问题，直接回答即可。"
                ),
            )
            logger.info("  Agent 创建完成")

            # RAG 链
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "你是「数码星球旗舰店」的智能客服助手小智。\n"
                 "根据参考资料回答问题，不要编造信息。\n"
                 "回答简洁友好，不超过100字。\n\n"
                 "参考资料:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])
            self.rag_chain = rag_prompt | self.llm | StrOutputParser()

            # 6. 缓存
            self.cache = ResponseCache(ttl=300, max_size=1000)
            logger.info("  缓存初始化完成 (TTL=300s, Max=1000)")

            logger.info("=" * 50)
            logger.info("初始化完成！")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            raise CustomerServiceError(f"系统初始化失败: {e}")

    def classify_intent(self, message: str) -> str:
        """
        判断用户意图。

        Returns:
            "agent" → 业务操作
            "rag"   → 知识问答
        """
        agent_keywords = ["订单", "退款", "退货", "物流", "快递", "发货", "签收", "取消"]
        for keyword in agent_keywords:
            if keyword in message:
                logger.debug(f"意图识别: '{message}' → agent (匹配关键词: {keyword})")
                return "agent"
        logger.debug(f"意图识别: '{message}' → rag")
        return "rag"

    def format_docs(self, docs: List[Document]) -> str:
        """把文档列表格式化成字符串"""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def chat(self, message: str, chat_history: List = []) -> str:
        """
        同步聊天。

        Args:
            message: 用户消息
            chat_history: LangChain 格式的对话历史

        Returns:
            回复内容

        Raises:
            LLMError: LLM 调用失败
            RetrievalError: 检索失败
            AgentError: Agent 调用失败
        """
        start_time = time.time()

        # 检查缓存
        cached = self.cache.get(message)
        if cached:
            logger.info(f"缓存命中: '{message}'")
            return cached["response"]

        intent = self.classify_intent(message)
        logger.info(f"收到消息: '{message}' → 意图: {intent}")

        try:
            if intent == "agent":
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=message)],
                })
                response = result["messages"][-1].content
            else:
                docs = self.retriever.invoke(message)
                context = self.format_docs(docs)
                logger.debug(f"检索到 {len(docs)} 个文档")
                response = self.rag_chain.invoke({
                    "context": context,
                    "chat_history": chat_history,
                    "question": message,
                })

            elapsed = time.time() - start_time
            logger.info(f"回复完成: 耗时 {elapsed:.2f}s, 长度 {len(response)} 字")

            # 缓存结果
            self.cache.set(message, response, intent)

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"聊天处理失败: {e} (耗时 {elapsed:.2f}s)", exc_info=True)
            # 返回友好提示，而不是抛异常
            return "抱歉，系统暂时出了点问题，请稍后再试。"

    def chat_stream(self, message: str, chat_history: List = []) -> Generator[str, None, None]:
        """
        流式聊天。

        Args:
            message: 用户消息
            chat_history: LangChain 格式的对话历史

        Yields:
            每次生成的一个 token
        """
        start_time = time.time()
        intent = self.classify_intent(message)
        logger.info(f"流式请求: '{message}' → 意图: {intent}")

        try:
            if intent == "agent":
                # Agent 暂不支持流式，直接返回完整结果
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=message)],
                })
                yield result["messages"][-1].content
            else:
                docs = self.retriever.invoke(message)
                context = self.format_docs(docs)
                for chunk in self.rag_chain.stream({
                    "context": context,
                    "chat_history": chat_history,
                    "question": message,
                }):
                    yield chunk

            elapsed = time.time() - start_time
            logger.info(f"流式完成: 耗时 {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"流式处理失败: {e} (耗时 {elapsed:.2f}s)", exc_info=True)
            yield "抱歉，系统暂时出了点问题，请稍后再试。"
