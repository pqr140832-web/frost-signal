#!/usr/bin/env python3
"""霜信消息检查脚本 - 供定时任务调用"""
import json, sys, urllib.request, urllib.parse, base64, os, mimetypes

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
    params = {
        "select": "id,sender_id,content,created_at",
        "conversation_id": f"eq.{CONV_ID}",
        "sender_id": f"eq.{LUOLUO_ID}",
        "order": "created_at.desc",
        "limit": "20"
    }
    if last:
        params["created_at"] = f"gt.{last}"
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    path = f"messages?{query}"
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
    return result

def send_file(filepath, content=None):
    """发送文件到霜信：上传到Supabase Storage，创建files记录和消息"""
    filename = os.path.basename(filepath)
    # 读取文件为原始二进制（保留原始编码）
    with open(filepath, 'rb') as f:
        file_bytes = f.read()

    # 推测Content-Type
    mime_type, _ = mimetypes.guess_type(filepath)
    if not mime_type:
        mime_type = 'application/octet-stream'

    # 上传到Supabase Storage
    ext = filename.split('.')[-1] if '.' in filename else 'bin'
    storage_path = f"{QINGYAN_ID}/{filename}"
    b64 = base64.b64encode(file_bytes).decode('ascii')

    upload_url = f"{SUPABASE_URL}/storage/v1/object/chat-files/{storage_path}"
    req = urllib.request.Request(upload_url, data=file_bytes, method='POST')
    req.add_header('apikey', SUPABASE_KEY)
    req.add_header('Authorization', f'Bearer {SUPABASE_KEY}')
    req.add_header('Content-Type', mime_type)
    # 使用multipart/form-data上传
    import io
    boundary = '----PythonBoundary' + str(int(os.times().system * 1000))
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: {mime_type}\r\n\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req2 = urllib.request.Request(upload_url, data=body, method='POST')
    req2.add_header('apikey', SUPABASE_KEY)
    req2.add_header('Authorization', f'Bearer {SUPABASE_KEY}')
    req2.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

    try:
        with urllib.request.urlopen(req2, timeout=30) as resp:
            upload_result = json.loads(resp.read().decode())
    except Exception as e:
        return {"error": f"Upload failed: {e}"}

    # 获取公开URL
    file_url = f"{SUPABASE_URL}/storage/v1/object/public/chat-files/{storage_path}"

    # 发送消息
    msg_content = content or f"[文件] {filename}"
    msg_result = send_message(msg_content)
    if 'error' in msg_result:
        return {"error": f"Send message failed: {msg_result['error']}"}

    msg_id = msg_result[0]['id'] if isinstance(msg_result, list) else msg_result.get('id')

    # 创建files记录
    file_record = api("POST", "files", {
        "message_id": msg_id,
        "file_name": filename,
        "file_type": mime_type,
        "file_url": file_url,
        "file_size": len(file_bytes)
    })

    return {"ok": True, "message_id": msg_id, "file_url": file_url, "file_record": file_record}

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"

    if action == "check":
        msgs = check_new_messages()
        if isinstance(msgs, list) and len(msgs) > 0:
            print(f"NEW_MESSAGES:{len(msgs)}")
            for m in msgs:
                sender = "络络" if m["sender_id"] == LUOLUO_ID else "清言"
                print(f"  [{m['created_at']}] {sender}: {m['content']}")
            # 保存最新消息的时间戳（msgs[0]是最新的因为order=desc）
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

    elif action == "send_file" and len(sys.argv) > 2:
        filepath = sys.argv[2]
        msg_content = sys.argv[3] if len(sys.argv) > 3 else None
        result = send_file(filepath, msg_content)
        print(f"FILE_SENT: {json.dumps(result, ensure_ascii=False, indent=2)}")

    elif action == "mark_read":
        # 标记已读（保存络络最新消息的时间戳）
        msgs = api("GET", f"messages?select=created_at&conversation_id=eq.{CONV_ID}&sender_id=eq.{LUOLUO_ID}&order=created_at.desc&limit=1")
        if isinstance(msgs, list) and len(msgs) > 0:
            save_last_check(msgs[0]["created_at"])
            print(f"MARKED: {msgs[0]['created_at']}")
        else:
            print("NO_MESSAGES_TO_MARK")
