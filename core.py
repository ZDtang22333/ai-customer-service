"""
客服核心逻辑
============
从 app.py 提取的核心模块，供 Gradio 和 FastAPI 共用。

包含：
- 组件初始化（LLM、Embedding、向量库、Agent）
- RAG 链
- 意图判断
- 聊天处理
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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


# ============================================
# 初始化组件
# ============================================

class CustomerService:
    """客服系统核心类"""

    def __init__(self):
        print("=" * 50)
        print("正在初始化客服系统...")
        print("=" * 50)

        # 1. LLM
        print("[1/5] 加载 LLM...")
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        # 2. Embedding
        print("[2/5] 加载 Embedding 模型...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 3. 向量数据库
        print("[3/5] 加载向量数据库...")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name=CHROMA_COLLECTION,
        )

        # 4. 混合检索
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

        hybrid = HybridRetriever(self.vectorstore, chunks, k=RETRIEVAL_K)
        self.retriever = RerankRetriever(hybrid, k=RETRIEVAL_K)

        # 5. Agent
        print("[5/5] 创建 Agent...")
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

        print("=" * 50)
        print("初始化完成！")
        print("=" * 50)

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
                return "agent"
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
        """
        intent = self.classify_intent(message)

        if intent == "agent":
            result = self.agent.invoke({
                "messages": [HumanMessage(content=message)],
            })
            return result["messages"][-1].content
        else:
            docs = self.retriever.invoke(message)
            context = self.format_docs(docs)
            return self.rag_chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": message,
            })

    def chat_stream(self, message: str, chat_history: List = []) -> Generator[str, None, None]:
        """
        流式聊天。

        Args:
            message: 用户消息
            chat_history: LangChain 格式的对话历史

        Yields:
            每次生成的一个 token
        """
        intent = self.classify_intent(message)

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
