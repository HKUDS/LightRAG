#!/bin/bash

set -e

# 日志记录工具函数
log_error() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: $1" >&2
}

export PYTHONPATH="${PYTHONPATH}:/app/api/"


# 启动服务
if [[ "${DEBUG}" == "true" ]]; then
  python /app/api/app.py --host=${SERVER_BIND_ADDRESS:-0.0.0.0} --port=${SERVER_BIND_PORT:-9621} --reload
  # uvicorn  app:app --host=${SERVER_BIND_ADDRESS:-0.0.0.0} --port=${SERVER_BIND_PORT:-9621} --reload
else
  python /app/api/app.py --host=${SERVER_BIND_ADDRESS:-0.0.0.0} --port=${SERVER_BIND_PORT:-9621} --workers=${WORKERS:-1}
  # uvicorn  app:app --host=${SERVER_BIND_ADDRESS:-0.0.0.0} --port=${SERVER_BIND_PORT:-9621}
fi