import json; with open("chunks_with_pages.json", "r", encoding="utf-8") as file: data = json.load(file); print("块总数:", len(data)); print("示例页码:"); for i in range(5): print(f"Chunk {i+1}, ID: {data[i][\"chunk_id\"]}, Page Numbers: {data[i][\"page_numbers\"]}")
