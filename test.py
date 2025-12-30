from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
model.eval().cuda()

query = "什么是长文本重排？"
documents = [
    "重排模型（Reranker）用于对初筛后的文档进行精细打分。",
    "GTE-Rerank-v2 支持 8k tokens，适合处理长文档。",
    "今天天气不错，适合出门旅游。"
]

# 构造输入
pairs = [[query, doc] for doc in documents]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=8192).to('cuda')

# 推理
with torch.no_grad():
    scores = model(**inputs).logits.view(-1).float()
    # 转换为概率（可选）
    # scores = torch.sigmoid(scores) 
print(scores)