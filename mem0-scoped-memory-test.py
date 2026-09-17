import json
import os
import sys

from dotenv import load_dotenv
from mem0 import MemoryClient

USER_ID = "mem0_test_user"
RUN_ID = "mem0_test_session"
AGENT_ID = "mem0_test_agent"

def log(title, data):
    print(f"\n=== {title} ===")
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))

def add_user_memory(client):
    messages = [
        {
            "role": "user",
            "content": "我叫小明，住在杭州，平时喜欢骑行和摄影。",
        },
        {
            "role": "assistant",
            "content": "好的，已记住你的姓名、城市和爱好。",
        },
    ]
    added = client.add(messages, user_id=USER_ID)
    log("用户记忆 - add", added)

def add_session_memory(client):
    messages = [
        {
            "role": "user",
            "content": "这次聊天先帮我把季度总结的大纲列出来，重点写 Q1 的项目复盘。",
        },
        {
            "role": "assistant",
            "content": "明白，我们先围绕 Q1 项目复盘整理季度总结大纲。",
        },
    ]
    added = client.add(messages, user_id=USER_ID, run_id=RUN_ID)
    log("会话记忆 - add", added)

def add_agent_memory(client):
    messages = [
        {
            "role": "user",
            "content": "你现在是旅行规划助手，回答时多给具体建议和备选方案。",
        },
        {
            "role": "assistant",
            "content": "好的，我会以旅行规划助手的身份，提供具体建议和备选方案。",
        },
    ]
    added = client.add(messages, agent_id=AGENT_ID)
    log("Agent 记忆 - add", added)

def search_user_memory(client):
    searched = client.search(
        "用户住在哪里，有什么爱好",
        filters={"user_id": USER_ID},
        top_k=5,
    )
    log(
        "用户记忆 - search",
        [item["memory"] for item in searched.get("results", [])],
    )

    listed = client.get_all(
        filters={"user_id": USER_ID},
        page_size=5,
    )
    log(
        "用户记忆 - get_all",
        [item["memory"] for item in listed.get("results", [])],
    )


def search_session_memory(client):
    session_filter = {
        "AND": [
            {"user_id": USER_ID},
            {"run_id": RUN_ID},
        ]
    }
    searched = client.search(
        "这次对话要先做什么",
        filters=session_filter,
        top_k=5,
    )
    log(
        "会话记忆 - search",
        [item["memory"] for item in searched.get("results", [])],
    )

    listed = client.get_all(
        filters=session_filter,
        page_size=5,
    )
    log(
        "会话记忆 - get_all",
        [item["memory"] for item in listed.get("results", [])],
    )

def search_agent_memory(client):
    searched = client.search(
        "这个 Agent 的角色和回答方式",
        filters={"agent_id": AGENT_ID},
        top_k=5,
    )
    log(
        "Agent 记忆 - search",
        [item["memory"] for item in searched.get("results", [])],
    )

    listed = client.get_all(
        filters={"agent_id": AGENT_ID},
        page_size=5,
    )
    log(
        "Agent 记忆 - get_all",
        [item["memory"] for item in listed.get("results", [])],
    )


def main():
    load_dotenv()
    api_key = os.getenv("MEM0_API_KEY")

    if not api_key:
        raise ValueError("缺少 MEM0_API_KEY")

    client = MemoryClient(api_key=api_key)
    # 命令行参数获取
    args = sys.argv[1:]
    if "--cleanup" in args:
        client.delete_all(user_id=USER_ID)
        client.delete_all(user_id=USER_ID, run_id=RUN_ID)
        client.delete_all(agent_id=AGENT_ID)
        log(
            "清理完成",
            {"USER_ID": USER_ID, "RUN_ID": RUN_ID, "AGENT_ID": AGENT_ID},
        )
        return

    action = args[0] if args else "add"

    if action == "add":
        add_user_memory(client)
        add_session_memory(client)
        add_agent_memory(client)
        print("\nadd 已提交，等记忆处理完成后再运行 search。")
    elif action == "search":
        search_user_memory(client)
        search_session_memory(client)
        search_agent_memory(client)
    else:
        raise ValueError(f"未知命令：{action}；可用：add、search、--cleanup")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n执行失败：{error}", file=sys.stderr)
        suggestion = getattr(error, "suggestion", None)
        if suggestion:
            print(f"建议：{suggestion}", file=sys.stderr)
        raise SystemExit(1)
