import asyncio
import json
import os

import redis.asyncio as redis
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from langchain_openai import ChatOpenAI
from mem0 import AsyncMemoryClient
from pydantic import BaseModel, Field

load_dotenv()

# 获取环境变量
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
MEMORY_TTL = int(os.getenv("MEMORY_TTL_SECONDS", "1800"))
KEY_PREFIX = os.getenv("MEMORY_KEY_PREFIX", "agent:short_memory")

USER_ID = os.getenv("MEM0_USER_ID", "demo_user_001")
SESSION_ID = os.getenv("MEMORY_SESSION_ID", "session_001")
MEM0_TOP_K = int(os.getenv("MEM0_TOP_K", "5"))

class MemoryDecision(BaseModel):
    write_user: bool = Field(
        description="是否写入跨会话长期用户记忆"
    )
    write_session: bool = Field(
        description="是否写入当前会话的任务记忆"
    )
    reason: str = Field(
        description="分类理由，一句话"
    )

CLASSIFIER_PROMPT = """
    你是记忆分层分类器。判断本轮对话是否有新事实需要写入 Mem0，并分到正确层级。

    user 层（跨会话长期）：
    - 用户身份与画像：姓名、职业、居住地、长期爱好
    - 长期偏好与约束：饮食过敏、回答风格、常用技术栈
    - 持续数周以上的个人背景

    session 层（仅当前会话）：
    - 当前正在做的任务、目标、文档大纲、方案草稿
    - 本会话内的进度、决策、待办、临时约定
    - 用户明确用「这次」「本轮」「当前会话」描述的工作上下文

    均不写入：
    - 寒暄、致谢、纯确认
    - 助手生成的通用内容，用户未明确采纳为新事实
    - 无信息增量的复述

    决策原则：
    1. 「这次我们先写 Q1 总结」「当前在排查 XX」优先 session。
    2. user 与 session 可以同时为 true。
    3. 一次性请求且未产生需要跨轮记住的约定，两者均为 false。
"""

SUMMARY_PROMPT = """
    你是对话摘要助手。用中文简洁总结话题、会话内进度、报错和待办。
    用户级长期偏好由外部记忆维护，摘要勿重复堆砌。不要编造。

    待摘要的对话：
    {messages}

    摘要：
"""

class RedisMessageStore:
    def __init__(self, client, key_prefix, ttl_seconds):
        self.client = client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

    def messages_key(self, session_id):
        return f"{self.key_prefix}:{session_id}:messages"

    async def load_messages(self, session_id):
        raw = await self.client.get(self.messages_key(session_id))
        if not raw:
            return []
        return messages_from_dict(json.loads(raw))

    async def save_messages(self, session_id, messages):
        payload = json.dumps(
            messages_to_dict(messages),
            ensure_ascii=False,
        )
        await self.client.set(
            self.messages_key(session_id),
            payload,
            ex=self.ttl_seconds,
        )

    async def clear(self, session_id):
        await self.client.delete(self.messages_key(session_id))

    async def ttl(self, session_id):
        return await self.client.ttl(self.messages_key(session_id))

class Mem0MemoryStore:
    def __init__(self, client, user_id, session_id, top_k, classifier):
        self.client = client
        self.user_id = user_id
        self.session_id = session_id
        self.top_k = top_k
        self.classifier = classifier

    async def search(self, query):
        user_result, session_result = await asyncio.gather(
            self.client.search(
                query,
                filters={"user_id": self.user_id},
                top_k=self.top_k,
            ),
            self.client.search(
                query,
                filters={
                    "AND": [
                        {"user_id": self.user_id},
                        {"run_id": self.session_id},
                    ]
                },
                top_k=self.top_k,
                rerank=True,
            ),
        )

        return {
            "user": user_result.get("results", []),
            "session": session_result.get("results", []),
        }

    def build_system_message(self, memories):
        blocks = []

        if memories["user"]:
            lines = [
                f"- {item['memory']}"
                for item in memories["user"]
            ]
            blocks.append("【用户长期记忆】\n" + "\n".join(lines))

        if memories["session"]:
            lines = [
                f"- {item['memory']}"
                for item in memories["session"]
            ]
            blocks.append("【当前会话记忆】\n" + "\n".join(lines))

        if not blocks:
            return None

        return SystemMessage(
            id="mem0-context",
            content=(
                "\n\n".join(blocks)
                + "\n\n请结合以上记忆回答，勿编造。"
            ),
        )

    async def classify_and_persist(self, user_text, assistant_text):
        turn = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        decision = await self.classifier.ainvoke(
            [
                SystemMessage(content=CLASSIFIER_PROMPT),
                HumanMessage(
                    content=(
                        f"用户：{user_text}\n"
                        f"助手：{assistant_text}"
                    )
                ),
            ]
        )

        submitted = []

        if decision.write_user:
            result = await self.client.add(
                turn,
                user_id=self.user_id,
            )
            submitted.append(("user", result))

        if decision.write_session:
            result = await self.client.add(
                turn,
                user_id=self.user_id,
                run_id=self.session_id,
            )
            submitted.append(("session", result))

        return submitted, decision.reason

    async def clear(self):
        await self.client.delete_all(
            user_id=self.user_id,
            run_id=self.session_id,
        )
        await self.client.delete_all(user_id=self.user_id)

async def invoke_with_memory(
    agent,
    redis_store,
    mem0_store,
    session_id,
    user_text,
):
    history = await redis_store.load_messages(session_id)
    print(f"  ↳ Redis 加载 {len(history)} 条历史")

    memories = await mem0_store.search(user_text)
    print(f"  ↳ Mem0 用户层 {len(memories['user'])} 条")
    print(f"  ↳ Mem0 会话层 {len(memories['session'])} 条")

    memory_message = mem0_store.build_system_message(memories)

    invoke_messages = []
    if memory_message is not None:
        invoke_messages.append(memory_message)

    invoke_messages.extend(history)
    invoke_messages.append(HumanMessage(content=user_text))

    result = await agent.ainvoke(
        {"messages": invoke_messages},
        config={"recursion_limit": 30},
    )

    redis_messages = [
        message
        for message in result["messages"]
        if message.id != "mem0-context"
    ]

    await redis_store.save_messages(session_id, redis_messages)
    ttl = await redis_store.ttl(session_id)

    print(
        f"  ↳ Redis 写回 {len(redis_messages)} 条"
        f"（TTL {ttl}s）"
    )

    last_answer = next(
        (
            message
            for message in reversed(result["messages"])
            if isinstance(message, AIMessage)
        ),
        None,
    )

    assistant_text = str(last_answer.content) if last_answer else ""

    submitted, reason = await mem0_store.classify_and_persist(
        user_text,
        assistant_text,
    )
    print(f"  ↳ 分类：{reason}")

    if submitted:
        print(
            "  ↳ Mem0 已提交："
            + "、".join(scope for scope, _ in submitted)
        )
    else:
        print("  ↳ Mem0 未写入")

    return redis_messages, assistant_text

async def main():
    if not os.getenv("MEM0_API_KEY"):
        raise RuntimeError("缺少 MEM0_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY")

    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )

    try:
        await redis_client.ping()
    except:
        await redis_client.aclose()
        raise RuntimeError(
            "Redis 未连接，请先执行：docker compose up -d redis"
        )

    redis_store = RedisMessageStore(
        redis_client,
        KEY_PREFIX,
        MEMORY_TTL,
    )

    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    classifier = model.with_structured_output(
        MemoryDecision,
        method="function_calling",
    )

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "你是会话助手。结合系统消息中的长期/会话记忆回答，"
            "中文简短。有对话摘要则据此继续。"
        ),
        middleware=[
            SummarizationMiddleware(
                model=model,
                summary_prompt=SUMMARY_PROMPT,
                trigger=("messages", 8),
                keep=("messages", 4),
            )
        ]
    )

    try:
        async with AsyncMemoryClient(
            api_key=os.environ["MEM0_API_KEY"]
        ) as mem0_client:
            mem0_store = Mem0MemoryStore(
                mem0_client,
                USER_ID,
                SESSION_ID,
                MEM0_TOP_K,
                classifier,
            )

            print(f"用户 {USER_ID} | 会话 {SESSION_ID}")
            print(
                "输入 exit / quit / :q 退出；"
                ":clear 清空 Redis；"
                ":clear-mem0 清空 Mem0\n"
            )

            previous_count = len(
                await redis_store.load_messages(SESSION_ID)
            )

            while True:
                try:
                    user_text = input("你：").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_text:
                    continue

                if user_text.lower() in {"exit", "quit", ":q"}:
                    break

                if user_text == ":clear":
                    await redis_store.clear(SESSION_ID)
                    previous_count = 0
                    print("已清空 Redis 短期记忆\n")
                    continue

                if user_text == ":clear-mem0":
                    await mem0_store.clear()
                    print("已提交 Mem0 清理请求\n")
                    continue

                redis_messages, assistant_text = (
                    await invoke_with_memory(
                        agent,
                        redis_store,
                        mem0_store,
                        SESSION_ID,
                        user_text,
                    )
                )

                print(f"\n助手：{assistant_text}")
                print(f"Redis 消息数：{len(redis_messages)}")

                if len(redis_messages) < previous_count + 2:
                    print("⚡ 已触发压缩")
                previous_count = len(redis_messages)
                print()

    finally:
        await redis_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())

