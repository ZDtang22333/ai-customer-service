"""
RAGAS 评估脚本
==============
评估 RAG 系统的质量。

评估指标：
- Faithfulness: 回答是否忠于检索到的文档
- Answer Relevance: 回答是否和问题相关
- Context Precision: 检索到的文档是否相关
- Context Recall: 是否检索到了所有相关文档

运行方式：python evaluate.py
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import time
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL,
    EMBEDDING_MODEL,
)
from hybrid_retriever import build_retriever
from logger import get_logger

logger = get_logger("evaluate")


# ============================================
# 评估数据集
# ============================================

def load_test_dataset(path: str = "test_dataset.json") -> List[Dict]:
    """加载测试数据集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================
# RAGAS 指标计算
# ============================================

class RAGASEvaluator:
    """
    RAGAS 评估器。

    简化版实现，用 LLM 评估各项指标。
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def _llm_evaluate(self, prompt: str) -> str:
        """调用 LLM 进行评估"""
        response = self.llm.invoke([
            SystemMessage(content="你是一个评估专家，只回答数字分数（0-1），不要解释。"),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def faithfulness(self, question: str, context: str, answer: str) -> float:
        """
        忠实度：回答是否忠于检索到的文档。

        评估方式：让 LLM 判断回答中有多少信息来自文档。
        """
        prompt = f"""请评估以下回答是否忠于提供的文档内容。

问题: {question}

文档内容:
{context}

回答: {answer}

请判断回答中的信息是否都可以从文档中找到。
只回答一个数字（0-1），1表示完全忠于文档，0表示完全不忠于文档。"""

        try:
            score = float(self._llm_evaluate(prompt))
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def answer_relevance(self, question: str, answer: str) -> float:
        """
        回答相关性：回答是否和问题相关。

        评估方式：计算问题和回答的语义相似度。
        """
        q_embedding = self.embeddings.embed_query(question)
        a_embedding = self.embeddings.embed_query(answer)
        return self._cosine_similarity(q_embedding, a_embedding)

    def context_precision(self, question: str, contexts: List[str]) -> float:
        """
        上下文精确度：检索到的文档是否相关。

        评估方式：让 LLM 判断每个文档是否和问题相关。
        """
        if not contexts:
            return 0.0

        scores = []
        for ctx in contexts:
            prompt = f"""请判断以下文档是否和问题相关。

问题: {question}

文档: {ctx[:500]}

只回答一个数字（0-1），1表示相关，0表示不相关。"""
            try:
                score = float(self._llm_evaluate(prompt))
                scores.append(max(0.0, min(1.0, score)))
            except:
                scores.append(0.5)

        return sum(scores) / len(scores)

    def context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        上下文召回率：是否检索到了所有相关文档。

        评估方式：让 LLM 判断标准答案中的信息是否在检索到的文档中。
        """
        context_text = "\n".join(contexts)

        prompt = f"""请判断标准答案中的信息是否出现在以下文档中。

标准答案: {ground_truth}

文档内容:
{context_text[:1000]}

只回答一个数字（0-1），1表示标准答案的信息都在文档中，0表示都不在。"""

        try:
            score = float(self._llm_evaluate(prompt))
            return max(0.0, min(1.0, score))
        except:
            return 0.5


# ============================================
# 评估流程
# ============================================

def run_evaluation(dataset_path: str = "test_dataset.json"):
    """
    运行完整评估。

    流程：
    1. 加载测试数据集
    2. 初始化检索器和评估器
    3. 对每个测试用例：
       - 检索相关文档
       - 生成回答
       - 计算各项指标
    4. 汇总输出评估结果
    """
    logger.info("=" * 50)
    logger.info("开始 RAGAS 评估")
    logger.info("=" * 50)

    # 1. 加载数据集
    dataset = load_test_dataset(dataset_path)
    logger.info(f"加载测试数据集: {len(dataset)} 条")

    # 2. 初始化
    logger.info("初始化检索器...")
    retriever, _ = build_retriever(
        use_rerank=True,
        use_parent_child=False,
        use_threshold=True,
        threshold=0.3,
    )

    logger.info("初始化评估器...")
    evaluator = RAGASEvaluator()

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # 3. 评估每个用例
    results = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"\n[{i}/{len(dataset)}] 评估: {question}")

        # 检索
        start_time = time.time()
        docs = retriever.invoke(question)
        retrieval_time = time.time() - start_time

        contexts = [doc.page_content for doc in docs]
        context_text = "\n\n---\n\n".join(contexts)

        # 生成回答
        start_time = time.time()
        prompt = f"""根据以下参考资料回答问题。如果资料中没有相关信息，请说"我不确定"。

参考资料:
{context_text}

问题: {question}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
        generation_time = time.time() - start_time

        # 计算指标
        logger.info(f"  计算指标...")
        faith = evaluator.faithfulness(question, context_text, answer)
        relevance = evaluator.answer_relevance(question, answer)
        precision = evaluator.context_precision(question, contexts)
        recall = evaluator.context_recall(ground_truth, contexts)

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "num_docs": len(docs),
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "faithfulness": round(faith, 3),
            "answer_relevance": round(relevance, 3),
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
        }
        results.append(result)

        logger.info(f"  回答: {answer[:50]}...")
        logger.info(f"  Faithfulness={faith:.3f}, Relevance={relevance:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    # 4. 汇总
    logger.info("\n" + "=" * 50)
    logger.info("评估汇总")
    logger.info("=" * 50)

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_retrieval_time = sum(r["retrieval_time"] for r in results) / len(results)
    avg_generation_time = sum(r["generation_time"] for r in results) / len(results)

    logger.info(f"Faithfulness:      {avg_faith:.3f}")
    logger.info(f"Answer Relevance:  {avg_relevance:.3f}")
    logger.info(f"Context Precision: {avg_precision:.3f}")
    logger.info(f"Context Recall:    {avg_recall:.3f}")
    logger.info(f"平均检索耗时:      {avg_retrieval_time:.2f}s")
    logger.info(f"平均生成耗时:      {avg_generation_time:.2f}s")

    # 保存结果
    output_path = "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "faithfulness": round(avg_faith, 3),
                "answer_relevance": round(avg_relevance, 3),
                "context_precision": round(avg_precision, 3),
                "context_recall": round(avg_recall, 3),
                "avg_retrieval_time": round(avg_retrieval_time, 2),
                "avg_generation_time": round(avg_generation_time, 2),
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n详细结果已保存到: {output_path}")

    return {
        "faithfulness": avg_faith,
        "answer_relevance": avg_relevance,
        "context_precision": avg_precision,
        "context_recall": avg_recall,
    }


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    run_evaluation()
