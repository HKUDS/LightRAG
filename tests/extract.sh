# 设置必要的环境变量
export LLM_BINDING_API_KEY=sk-dzbnefdirmeybvkynbbzkpckyybwzsgnaixgnkzmlvmjjtzn
export LLM_BINDING_HOST=https://api.siliconflow.cn/v1
export LLM_MODEL=Qwen/Qwen2.5-32B-Instruct

# 运行脚本（所有参数都是必须的）
python entity_extract.py -i demo3.json -o output.cypher -c config.yaml
