#!/usr/bin/env python3
"""霜信消息检查脚本 - 供定时任务调用"""
import json, sys, urllib.request

SUPABASE_URL = "https://deuvpiwjzkfmeswzztlf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRldXZwaXdqemtmbWVzd3p6dGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM5MzA0ODUsImV4cCI6MjA4OTUwNjQ4NX0.ImERt2pZDxVmCLQwmEJ_QlgCn7978AIa_GNsqfQ3lf8"
CONV_ID = "b229cee7-ae82-44e7-8241-e762138f484f"
LUOLUO_ID = "30ec59c7-89c0-4dc3-9fb5-e755f52ecfe7"
QINGYAN_ID = "b18b15db-be91-4e37-a450-65cb77aa54aa"
TIMESTAMP_FILE = "/home/z/my-project/download/frost-signal-last-check.txt"

def api(method, path, body=None):
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode()
            if raw:
                return json.loads(raw)
            return {"ok": True}
    except Exception as e:
        return {"error": str(e)}

def get_last_check():
    try:
        with open(TIMESTAMP_FILE) as f:
            return f.read().strip()
    except:
        return None

def save_last_check(ts):
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(ts)

def check_new_messages():
    last = get_last_check()
    path = f"messages?select=id,sender_id,content,created_at&conversation_id=eq.{CONV_ID}&sender_id=eq.{LUOLUO_ID}&order=created_at.desc&limit=20"
    if last:
        path += f"&created_at=gt.{last}"
    return api("GET", path)

def get_recent_context(n=10):
    """获取最近n条消息作为上下文"""
    path = f"messages?select=id,sender_id,content,created_at&conversation_id=eq.{CONV_ID}&order=created_at.desc&limit={n}"
    msgs = api("GET", path)
    if isinstance(msgs, list):
        return list(reversed(msgs))
    return []

def send_message(content):
    result = api("POST", "messages", {
        "conversation_id": CONV_ID,
        "sender_id": QINGYAN_ID,
        "content": content
    })
    # 更新对话时间戳
    api("PATCH", f"conversations?id=eq.{CONV_ID}", {"updated_at": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z')})
    return result

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"

    if action == "check":
        msgs = check_new_messages()
        if isinstance(msgs, list) and len(msgs) > 0:
            print(f"NEW_MESSAGES:{len(msgs)}")
            for m in msgs:
                sender = "络络" if m["sender_id"] == LUOLUO_ID else "清言"
                print(f"  [{m['created_at']}] {sender}: {m['content']}")
            # 更新最后检查时间
            save_last_check(msgs[0]["created_at"])
        else:
            print("NO_NEW_MESSAGES")

    elif action == "context":
        msgs = get_recent_context()
        print("=== 最近对话 ===")
        for m in msgs:
            sender = "络络" if m["sender_id"] == LUOLUO_ID else "清言"
            print(f"[{m['created_at']}] {sender}: {m['content']}")

    elif action == "send" and len(sys.argv) > 2:
        content = sys.argv[2]
        result = send_message(content)
        print(f"SENT: {result}")

    elif action == "mark_read":
        # 标记已读（更新最后检查时间为当前时间）
        msgs = api("GET", f"messages?select=created_at&conversation_id=eq.{CONV_ID}&order=created_at.desc&limit=1")
        if isinstance(msgs, list) and len(msgs) > 0:
            save_last_check(msgs[0]["created_at"])
            print(f"MARKED: {msgs[0]['created_at']}")
        else:
            print("NO_MESSAGES_TO_MARK")
