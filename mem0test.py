import json
import os
import sys

from dotenv import load_dotenv
from mem0 import MemoryClient


USER_ID = "demo-user"

def log(title, data):
    print(f"\n=== {title} ===")
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))

def main():
    load_dotenv()
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中设置 MEM0_API_KEY")

    client = MemoryClient(api_key=api_key)

    conversation = [
        {"role": "user", "content": "我是素食主义者，而且对坚果过敏。"},
        {"role": "assistant", "content": "好的，我会记住你的饮食偏好。"},
        {"role": "user", "content": "我住在北京，平时喜欢跑步。"},
        {"role": "assistant", "content": "已记录：北京、爱好跑步。"},
    ]

    added = client.add(conversation, user_id=USER_ID)
    log("添加记忆", added)

    search_result = client.search(
        "用户的饮食限制是什么？中文回答",
        filters={"user_id": USER_ID},
        top_k=5,
    )

    log("搜索记忆", search_result)

    all_memories = client.get_all(
        filters={"user_id": USER_ID},
        page_size=10,
    )
    log("列出全部记忆", all_memories)

    results = all_memories.get("results") or search_result.get("results") or []
    first_memory = results[0] if results else None

    if first_memory and first_memory.get("id"):
        memory_id = first_memory["id"]

        memory = client.get(memory_id)
        log("获取单条记忆", memory)

        memory_text = memory.get("memory") or first_memory.get("memory") or ""

        updated = client.update(
            memory_id,
            text=f"{memory_text}（已通过示例脚本更新）",
        )
        log("更新记忆", updated)

        history = client.history(memory_id)
        log("记忆变更历史", history)

    if "--cleanup" in sys.argv[1:]:
        deleted = client.delete_all(user_id=USER_ID)
        log("清理测试数据", deleted)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"执行失败：{error}", file=sys.stderr)
        raise SystemExit(1)
    

