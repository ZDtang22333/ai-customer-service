"""
配置管理模块
============
把所有配置集中在一个地方，方便修改，也避免把密钥写死在代码里。

学习要点：
1. 环境变量：敏感信息（如API Key）不应该写在代码里，而是通过环境变量传入
2. .env文件：本地开发时，可以把环境变量放在 .env 文件里，python-dotenv 会自动加载
3. 默认值：非敏感配置可以给默认值，方便开发
"""

import os
from dotenv import load_dotenv

# HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ============================================
# 第一步：加载 .env 文件
# ============================================
# .env 文件格式（在项目根目录创建）：
#   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
#   OPENAI_BASE_URL=https://api.openai.com/v1  (可选，用于代理)
#
# load_dotenv() 会找到 .env 文件并把里面的键值对加载到环境变量中
# 如果 .env 文件不存在也不会报错，只是静默跳过
load_dotenv()

# ============================================
# 第二步：OpenAI 配置
# ============================================

# API Key: 必须配置，否则无法调用OpenAI
# os.getenv() 从环境变量读取，第二个参数是默认值
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Base URL: API服务地址
#   小米MiMo: https://token-plan-cn.xiaomimimo.com/v1
#   OpenAI官方: https://api.openai.com/v1
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")

# 聊天模型:
#   小米MiMo: mimo-v2.5-pro
#   OpenAI: gpt-3.5-turbo, gpt-4o
CHAT_MODEL = os.getenv("CHAT_MODEL", "mimo-v2.5-pro")

# Embedding模型: 把文本转换成向量（数字数组），用于相似度检索
# 本地模型，不需要API，首次运行会自动下载
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ============================================
# 第三步：RAG 配置
# ============================================

# 文本切分参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
# chunk_overlap: 相邻两个文本块重叠的字符数
# 为什么要重叠？防止一句话被切断，导致语义丢失
# 比如 chunk_size=500, overlap=50
# 第1块: 字符 0-499
# 第2块: 字符 450-949  (重叠了50个字符)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索参数
# 每次检索返回最相似的 k 个文档块
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# ============================================
# 第四步：ChromaDB 配置
# ============================================

# 向量数据库持久化目录
# ChromaDB会把数据存在这个目录下，下次启动还能用
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 集合名称（类似数据库的"表名"）
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "customer_service")

# ============================================
# 第五步：验证配置
# ============================================
def validate_config():
    """检查必要的配置是否已设置"""
    if not OPENAI_API_KEY:
        raise ValueError(
            "[ERROR] 未设置 OPENAI_API_KEY！\n"
            "请在 .env 文件中添加：\n"
            "  OPENAI_API_KEY=sk-你的密钥\n"
            "\n"
            "或者设置环境变量：\n"
            "  export OPENAI_API_KEY=sk-你的密钥"
        )
    print("[OK] 配置验证通过")


# ============================================
# 第六步：打印当前配置（调试用）
# ============================================
def print_config():
    """打印当前配置（隐藏敏感信息）"""
    print("=" * 40)
    print("当前配置:")
    print(f"  API Key:    {'已设置' if OPENAI_API_KEY else '❌ 未设置'}")
    print(f"  Base URL:   {OPENAI_BASE_URL}")
    print(f"  聊天模型:    {CHAT_MODEL}")
    print(f"  Embedding:  {EMBEDDING_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"  检索数量 K:  {RETRIEVAL_K}")
    print(f"  ChromaDB:   {CHROMA_PERSIST_DIR}")
    print("=" * 40)


# 如果直接运行这个文件，打印配置看看
if __name__ == "__main__":
    print_config()
    validate_config()
