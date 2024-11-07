# Palmier

This repository is a fork of LightRAG. Palmier uses LightRAG as the underlying RAG engine.

## How to run

1. Install dependencies
```
pip install -r requirements.txt
```

2. Run the server
```
python palmier/main.py
```

## Endpoint

### Index
```
curl -X POST "http://127.0.0.1:8020/v1/index"
     -H "Content-Type: application/json, X-Github-Token: your_github_token" 
     -d '{"repo": "owner/repo", "branch": "main"}' 
```
```
curl -X GET "http://127.0.0.1:8020/v1/index/status/{owner}/{repo}"
```
### Query
```
curl -X POST "http://127.0.0.1:8020/v1/query"
     -H "Content-Type: application/json"
     -d '{"repo": "owner/repo", "query": "your query here", "mode": "hybrid", "response_type": "Multiple Paragraphs", "top_k": 60}'
```
### Health
```
curl -X GET "http://127.0.0.1:8020/v1/health"
```


