"""
Agent 工具模块
==============
定义客服可以调用的工具，让客服能"做事"而不只是"回答问题"。

工具列表：
- query_order: 查询订单状态
- apply_refund: 提交退款申请
- check_refund_policy: 检查退款资格
- query_logistics: 查询物流信息
- transfer_human: 转人工客服

运行方式：python agent_tools.py（单独测试）
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from config import OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL


# ============================================
# 模拟数据
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
    "11111": {
        "order_id": "11111",
        "product": "小米手表S3",
        "amount": 1299,
        "status": "已签收",
        "sign_date": "2026-05-20",
        "user": "小红",
    },
}

LOGISTICS = {
    "12345": {"company": "顺丰", "tracking": "SF1234567890", "status": "已签收"},
    "67890": {"company": "京东物流", "tracking": "JD9876543210", "status": "运输中，预计明天到达"},
}


# ============================================
# 工具定义
# ============================================

@tool
def query_order(order_id: str) -> str:
    """
    查询订单状态。输入订单号，返回订单详情。

    Args:
        order_id: 订单号，如 "12345"
    """
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
    """
    提交退款申请。输入订单号和退款原因。

    Args:
        order_id: 订单号
        reason: 退款原因，如 "不想要了"、"质量问题"、"发错货"
    """
    order = ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}，无法提交退款。"

    if order["status"] != "已签收":
        return f"订单 {order_id} 状态为 {order['status']}，未签收的订单请申请取消订单而非退款。"

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
    """
    检查订单是否符合退款条件。

    Args:
        order_id: 订单号
    """
    from datetime import datetime

    order = ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}。"

    if order["status"] != "已签收":
        return f"订单 {order_id} 尚未签收，可以直接取消订单。"

    # 计算签收到现在的天数
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
    """
    查询物流信息。输入订单号，返回物流状态。

    Args:
        order_id: 订单号
    """
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
    """
    转接人工客服。当问题无法自动解决时使用。

    Args:
        reason: 转接原因
    """
    return f"已为您转接人工客服，转接原因: {reason}。请稍候，人工客服即将接入。"


# ============================================
# 工具列表（供 Agent 使用）
# ============================================

TOOLS = [query_order, apply_refund, check_refund_policy, query_logistics, transfer_human]


# ============================================
# 创建 Agent
# ============================================

def create_customer_agent():
    """
    创建能调用工具的 Agent。

    Agent 的工作流程：
    1. 用户提问
    2. LLM 判断是否需要调用工具
    3. 如果需要，LLM 选择工具并生成参数
    4. 执行工具，获取结果
    5. LLM 根据工具结果组织回答

    这和普通链的区别：
    - 普通链：用户 → LLM → 回答（只能用LLM知识）
    - Agent：用户 → LLM → 工具 → LLM → 回答（能调用外部能力）
    """
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # 创建 Agent（新版 LangChain 用 create_agent）
    agent_executor = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=(
            "你是数码星球的客服助手小智。\n"
            "你可以调用工具来帮助用户处理订单、退款、物流等问题。\n"
            "如果用户的问题需要查订单或处理业务，一定要调用工具，不要猜测。\n"
            "如果用户只是闲聊或问产品问题，直接回答即可。"
        ),
    )

    return agent_executor


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("Agent 工具测试")
    print("=" * 50)

    # 测试工具（需要用 .invoke() 调用）
    print("\n[测试] 查询订单:")
    print(query_order.invoke("12345"))

    print("\n[测试] 检查退款资格:")
    print(check_refund_policy.invoke("12345"))

    print("\n[测试] 查询物流:")
    print(query_logistics.invoke("67890"))

    # 测试 Agent
    print("\n" + "=" * 50)
    print("Agent 对话测试")
    print("=" * 50)

    agent = create_customer_agent()
    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = []

    test_inputs = [
        "帮我查一下订单 12345 的状态",
        "这个订单能退款吗？",
        "帮我申请退款，原因是不想要了",
    ]

    for user_input in test_inputs:
        print(f"\n用户: {user_input}")
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)],
        })
        # 新版返回 messages 列表，取最后一条
        response = result["messages"][-1].content
        print(f"小智: {response}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
