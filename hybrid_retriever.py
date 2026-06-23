"""
混合检索模块 = BM25 + 语义搜索 + 重排序 + 父子块 + 阈值截断
==============================================================
包含所有检索策略：
- BM25Retriever: 关键词匹配
- HybridRetriever: 混合检索（BM25 + 语义）
- RerankRetriever: 重排序
- ParentChildRetriever: 父子块检索
- ThresholdRetriever: 阈值截断

运行方式：python hybrid_retriever.py（单独测试）
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION, RETRIEVAL_K,
)


# ============================================
# 第一部分：BM25 检索器（关键词匹配）
# ============================================

class BM25Retriever:
    """
    BM25 关键词检索器。

    原理：
    - 把文档分词，建立倒排索引
    - 用户查询也分词，计算每个文档的相关性分数
    - 返回分数最高的 k 个文档

    优点：精确匹配关键词，不会漏掉包含关键信息的文档
    缺点：无法理解语义相似（"电脑"和"笔记本"对BM25来说是不同的词）
    """

    def __init__(self, documents: List[Document], k: int = 3):
        """
        Args:
            documents: 文档列表
            k: 返回数量
        """
        self.documents = documents
        self.k = k

        # 对文档内容分词
        # jieba 是中文分词库，把句子切成词语
        # 例如："笔记本电池能用多久" → ["笔记本", "电池", "能", "用", "多久"]
        self.corpus_tokens = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]

        # 建立 BM25 索引
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def invoke(self, query: str) -> List[Document]:
        """检索：返回最相关的 k 个文档"""
        # 查询分词
        query_tokens = list(jieba.cut(query))

        # BM25 打分
        scores = self.bm25.get_scores(query_tokens)

        # 取分数最高的 k 个
        top_indices = scores.argsort()[-self.k:][::-1]

        return [self.documents[i] for i in top_indices]


# ============================================
# 第二部分：混合检索器（BM25 + 语义搜索）
# ============================================

class HybridRetriever(BaseRetriever):
    """
    混合检索器：结合 BM25 和语义搜索。

    策略：
    1. BM25 找包含关键词的文档（精确匹配）
    2. 语义搜索找意思相近的文档（模糊匹配）
    3. 合并去重，返回结果

    为什么混合？
    - 问"退货要几天" → BM25 找到包含"退货""几天"的文档
    - 问"不想要了怎么办" → 语义搜索能找到"退货政策"（虽然没有"不想要"这个词）
    """

    # Pydantic 配置，允许任意类型
    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        k: int = 3,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ):
        """
        Args:
            vectorstore: ChromaDB 向量数据库
            documents: 文档列表（用于 BM25）
            k: 返回数量
            bm25_weight: BM25 权重（0-1）
            semantic_weight: 语义搜索权重（0-1）
        """
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
        self._bm25_weight = bm25_weight
        self._semantic_weight = semantic_weight

        # BM25 检索器
        self._bm25_retriever = BM25Retriever(documents, k=k * 2)

        # 语义检索器
        self._semantic_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k * 2}
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """混合检索"""
        # 1. BM25 检索
        bm25_docs = self._bm25_retriever.invoke(query)

        # 2. 语义检索
        semantic_docs = self._semantic_retriever.invoke(query)

        # 3. 合并去重（用文档内容作为唯一标识）
        seen = set()
        merged_docs = []

        # 语义搜索结果优先（通常更相关）
        for doc in semantic_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                merged_docs.append(doc)

        # 补充 BM25 结果
        for doc in bm25_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                merged_docs.append(doc)

        # 返回前 k 个
        return merged_docs[:self._k]


# ============================================
# 第三部分：重排序检索器（精排）
# ============================================

class RerankRetriever(BaseRetriever):
    """
    重排序检索器：用 CrossEncoder 对检索结果重新打分。

    流程：
    1. 先用混合检索获取候选文档（数量多一些，比如10个）
    2. 用 CrossEncoder 对每个 (query, doc) 对打分
    3. 按新分数排序，取前 k 个

    为什么需要重排序？
    - 混合检索返回的顺序不一定最优
    - CrossEncoder 能更精确地判断 query 和 doc 的相关性
    - 代价是更慢（需要对每个候选单独计算）
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        k: int = 3,
        rerank_top_n: int = 10,
    ):
        """
        Args:
            base_retriever: 基础检索器（混合检索）
            k: 最终返回数量
            rerank_top_n: 重排序候选数量
        """
        super().__init__()
        self._base_retriever = base_retriever
        self._k = k
        self._rerank_top_n = rerank_top_n

        # CrossEncoder 模型
        # 这个模型专门用于判断两段文本的相关性
        # 比 Embedding 相似度更准确，但更慢
        from sentence_transformers import CrossEncoder
        print("  正在加载重排序模型...")
        self._reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        print("  重排序模型加载完成")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """重排序检索"""
        # 1. 基础检索获取候选
        candidates = self._base_retriever.invoke(query)

        # 限制候选数量
        candidates = candidates[:self._rerank_top_n]

        if not candidates:
            return []

        # 2. CrossEncoder 打分
        # 构造 (query, doc) 对
        pairs = [(query, doc.page_content) for doc in candidates]

        # 打分
        scores = self._reranker.predict(pairs)

        # 3. 按分数排序
        scored_docs = list(zip(scores, candidates))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 返回前 k 个
        return [doc for _, doc in scored_docs[:self._k]]


# ============================================
# 第四部分：父子块检索器
# ============================================

class ParentChildRetriever(BaseRetriever):
    """
    父子块检索器：用小块检索，返回大块。

    原理：
    - 切分时同时生成父块（大块）和子块（小块）
    - 检索时用子块（更精确）
    - 返回时返回父块（上下文更完整）

    示例：
    父块: "数码星球旗舰店退货政策...7天无理由...退款时效1-3天..."
      ├── 子块1: "7天无理由退货"
      ├── 子块2: "退款时效1-3个工作日"
      └── 子块3: "运费由买家承担"

    用户问: "退货要几天"
      → 子块2 匹配（精确）
      → 返回父块（完整退货政策）
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        k: int = 3,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
    ):
        """
        Args:
            vectorstore: ChromaDB 向量数据库
            documents: 原始文档列表
            k: 返回数量
            parent_chunk_size: 父块大小
            child_chunk_size: 子块大小
        """
        super().__init__()
        self._k = k
        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size

        # 生成父子块
        self._parent_docs, self._child_docs, self._child_to_parent = \
            self._create_parent_child_chunks(documents)

        # 用子块建立向量库（用于精确检索）
        self._child_vectorstore = Chroma.from_documents(
            documents=self._child_docs,
            embedding=vectorstore._embedding_function,
            collection_name="child_chunks",
        )

        print(f"  父子块切分完成: {len(self._parent_docs)} 个父块, {len(self._child_docs)} 个子块")

    def _create_parent_child_chunks(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document], Dict[int, int]]:
        """
        创建父子块。

        Returns:
            parent_docs: 父块列表
            child_docs: 子块列表
            child_to_parent: 子块索引 → 父块索引 的映射
        """
        # 父块切分器
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=100,
        )

        # 子块切分器
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=50,
        )

        parent_docs = []
        child_docs = []
        child_to_parent = {}

        for doc in documents:
            # 先切成父块
            parents = parent_splitter.split_documents([doc])

            for _, parent in enumerate(parents):
                parent_idx_global = len(parent_docs)
                parent_docs.append(parent)

                # 再把父块切成子块
                children = child_splitter.split_documents([parent])

                for child in children:
                    child_idx = len(child_docs)
                    child_docs.append(child)
                    child_to_parent[child_idx] = parent_idx_global

        return parent_docs, child_docs, child_to_parent

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """父子块检索：用子块检索，返回父块"""
        # 1. 用子块检索
        child_results = self._child_vectorstore.similarity_search(
            query, k=self._k * 2
        )

        # 2. 找到对应的父块
        seen_parents = set()
        parent_results = []

        for child in child_results:
            child_idx = self._child_docs.index(child)
            parent_idx = self._child_to_parent.get(child_idx)

            if parent_idx is not None and parent_idx not in seen_parents:
                seen_parents.add(parent_idx)
                parent_results.append(self._parent_docs[parent_idx])

        return parent_results[:self._k]


# ============================================
# 第五部分：阈值截断检索器
# ============================================

class ThresholdRetriever(BaseRetriever):
    """
    阈值截断检索器：过滤掉低分文档。

    原理：
    - 检索后，计算每个文档和 query 的相似度
    - 只保留相似度高于阈值的文档
    - 避免不相关文档污染 LLM 输入

    好处：
    - 提高检索精度
    - 没有找到相关文档时，可以返回空列表（让 LLM 说"我不确定"）
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        embeddings: HuggingFaceEmbeddings,
        threshold: float = 0.5,
        k: int = 3,
    ):
        """
        Args:
            base_retriever: 基础检索器
            embeddings: Embedding 模型（用于计算相似度）
            threshold: 相似度阈值（0-1），低于此值的文档被过滤
            k: 返回数量
        """
        super().__init__()
        self._base_retriever = base_retriever
        self._embeddings = embeddings
        self._threshold = threshold
        self._k = k

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """阈值截断检索"""
        # 1. 基础检索
        candidates = self._base_retriever.invoke(query)

        if not candidates:
            return []

        # 2. 计算 query 和每个文档的相似度
        query_embedding = self._embeddings.embed_query(query)

        filtered_docs = []
        for doc in candidates:
            doc_embedding = self._embeddings.embed_query(doc.page_content[:500])
            similarity = self._cosine_similarity(query_embedding, doc_embedding)

            if similarity >= self._threshold:
                filtered_docs.append(doc)

        return filtered_docs[:self._k]


# ============================================
# 第六部分：构建检索器
# ============================================

def build_retriever(
    use_rerank: bool = True,
    use_parent_child: bool = False,
    use_threshold: bool = False,
    threshold: float = 0.5,
):
    """
    构建检索器。

    Args:
        use_rerank: 是否启用重排序
        use_parent_child: 是否启用父子块检索
        use_threshold: 是否启用阈值截断
        threshold: 相似度阈值

    Returns:
        retriever: 检索器
        embeddings: Embedding 模型
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    # 加载文档
    loader = DirectoryLoader(
        "knowledge",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    # 切分文档（小块用于检索）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 向量数据库
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )

    # 选择检索器
    if use_parent_child:
        # 父子块检索
        print("  使用父子块检索...")
        retriever = ParentChildRetriever(
            vectorstore=vectorstore,
            documents=documents,
            k=RETRIEVAL_K,
        )
    else:
        # 混合检索
        hybrid = HybridRetriever(
            vectorstore=vectorstore,
            documents=chunks,
            k=RETRIEVAL_K,
        )
        retriever = hybrid

    # 可选：加重排序
    if use_rerank:
        retriever = RerankRetriever(
            base_retriever=retriever,
            k=RETRIEVAL_K,
        )

    # 可选：加阈值截断
    if use_threshold:
        retriever = ThresholdRetriever(
            base_retriever=retriever,
            embeddings=embeddings,
            threshold=threshold,
            k=RETRIEVAL_K,
        )

    return retriever, embeddings


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("混合检索测试")
    print("=" * 50)

    retriever, _ = build_retriever(
        use_rerank=True,
        use_parent_child=False,
        use_threshold=True,
        threshold=0.3,
    )

    # 测试问题
    test_queries = [
        "退货要几天",
        "不想要了怎么办",
        "笔记本电池续航",
        "asdfgh",  # 无意义输入，测试阈值截断
    ]

    for query in test_queries:
        print(f"\n问题: {query}")
        results = retriever.invoke(query)
        print(f"  找到 {len(results)} 个文档:")
        for i, doc in enumerate(results):
            source = os.path.basename(doc.metadata.get("source", ""))
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i+1}] {source}: {preview}...")
