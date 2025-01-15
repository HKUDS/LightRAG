import requests
import json
import sseclient

def test_non_stream_chat():
    """测试非流式调用 /api/chat 接口"""
    url = "http://localhost:9621/api/chat"
    
    # 构造请求数据
    data = {
        "model": "lightrag:latest",
        "messages": [
            {
                "role": "user",
                "content": "孙悟空"
            }
        ],
        "stream": False
    }
    
    # 发送请求
    response = requests.post(url, json=data)
    
    # 打印响应
    print("\n=== 非流式调用响应 ===")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_stream_chat():
    """测试流式调用 /api/chat 接口"""
    url = "http://localhost:9621/api/chat"
    
    # 构造请求数据
    data = {
        "model": "lightrag:latest",
        "messages": [
            {
                "role": "user",
                "content": "孙悟空有什么法力，性格特征是什么"
            }
        ],
        "stream": True
    }
    
    # 发送请求并获取 SSE 流
    response = requests.post(url, json=data, stream=True)
    client = sseclient.SSEClient(response)
    
    print("\n=== 流式调用响应 ===")
    output_buffer = []
    try:
        for event in client.events():
            try:
                data = json.loads(event.data)
                if data.get("done", False):  # 如果是完成标记
                    if "total_duration" in data:  # 最终的性能统计消息
                        print("\n=== 性能统计 ===")
                        print(json.dumps(data, ensure_ascii=False, indent=2))
                        break
                else:  # 正常的内容消息
                    message = data.get("message", {})
                    content = message.get("content", "")
                    if content:  # 只收集非空内容
                        output_buffer.append(content)
            except json.JSONDecodeError:
                print("Error decoding JSON from SSE event")
    finally:
        response.close()  # 确保关闭响应连接
        
    # 一次性打印所有收集到的内容
    print("".join(output_buffer))

if __name__ == "__main__":
    # 先测试非流式调用
    test_non_stream_chat()
    
    # 再测试流式调用
    test_stream_chat()
