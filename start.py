#!/usr/bin/env python3
"""
LightRAG 一键启动脚本
自动完成：依赖安装 → 前端构建 → 终止旧进程 → 启动服务
"""

import subprocess
import sys
import os
import time
import signal
import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# 修复 Windows 控制台 GBK 编码问题：强制 stdout/stderr 使用 UTF-8
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(r"D:\code\LightRAG")
WEBUI_DIR = PROJECT_DIR / "lightrag_webui"
ENV_FILE = PROJECT_DIR / ".env"
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
SERVER_PORT = 9621


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────

def _resolve_shell(cmd, shell):
    """Determine whether shell=True is needed and normalize the command."""
    if shell is not None:
        return cmd, shell

    # On Windows, .cmd/.bat files (npm, bun, npx, etc.) require shell=True.
    # For all other commands, prefer shell=False for safety.
    if platform.system() != "Windows":
        return cmd, False

    # If it's already a string, use shell=True on Windows.
    if isinstance(cmd, str):
        return cmd, True

    # Check if the command is a .cmd/.bat wrapper that needs shell=True.
    executable = cmd[0] if isinstance(cmd, list) else cmd
    if executable.endswith((".cmd", ".bat")):
        return cmd, True

    # For commands like "npm", "bun", "npx" — check if they exist as .exe.
    import shutil
    exe_path = shutil.which(executable)
    if exe_path is None:
        # Not found; try with shell=True in case it's a newly installed .cmd.
        return cmd, True
    if exe_path.lower().endswith((".cmd", ".bat")):
        # It's a .cmd wrapper — needs shell=True.
        return cmd, True

    # Regular .exe — shell=False is fine.
    return cmd, False


def run(cmd, cwd=None, env=None, check=True, shell=None):
    """运行命令，实时输出，出错时退出。"""
    print(f"\n  >>> {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    cmd, use_shell = _resolve_shell(cmd, shell)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        subprocess.run(cmd, cwd=cwd or PROJECT_DIR, env=merged_env,
                       check=check, shell=use_shell)
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] 命令执行失败，脚本终止。")
        sys.exit(1)


def run_noerr(cmd, cwd=None, env=None, shell=None):
    """运行命令，忽略错误（用于检查类操作）。"""
    cmd, use_shell = _resolve_shell(cmd, shell)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        return subprocess.run(cmd, cwd=cwd or PROJECT_DIR, env=merged_env,
                              capture_output=True, text=True, shell=use_shell)
    except Exception:
        return None


def is_command_available(name):
    """检查命令行工具是否可用。"""
    result = run_noerr(["where", name] if platform.system() == "Windows" else ["which", name])
    return result is not None and result.returncode == 0


def is_bun_available():
    return is_command_available("bun")


def kill_port(port):
    """终止占用指定端口的进程（Windows）。"""
    print(f"\n[INFO] 检查端口 {port} 是否被占用...")
    if platform.system() == "Windows":
        result = run_noerr(
            f"netstat -ano | findstr :{port}",
            shell=True
        )
        if result and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            pids = set()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    pid = parts[-1]
                    pids.add(pid)
            for pid in pids:
                print(f"  [INFO] 终止进程 PID={pid} (占用端口 {port})...")
                run_noerr(["taskkill", "/F", "/PID", pid])
            if pids:
                time.sleep(1)
                print(f"  [OK] 已终止 {len(pids)} 个进程。")
            else:
                print(f"  [INFO] 端口 {port} 未被占用。")
        else:
            print(f"  [INFO] 端口 {port} 未被占用。")
    else:
        # Linux/macOS
        result = run_noerr(["lsof", "-ti", f":{port}"])
        if result and result.stdout.strip():
            for pid in result.stdout.strip().split("\n"):
                print(f"  [INFO] 终止进程 PID={pid} (占用端口 {port})...")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(1)
            print(f"  [OK] 已终止占用端口的进程。")
        else:
            print(f"  [INFO] 端口 {port} 未被占用。")


# ──────────────────────────────────────────────
#  步骤 1：环境准备
# ──────────────────────────────────────────────

def step_env():
    """检查 .env 文件是否存在。"""
    print("=" * 60)
    print("  [1/7] 检查环境配置 (.env)")
    print("=" * 60)
    if ENV_FILE.exists():
        print(f"  [OK] .env 文件已存在: {ENV_FILE}")
    else:
        print(f"  [WARN] .env 文件不存在，将从 env.example 复制...")
        env_example = PROJECT_DIR / "env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, ENV_FILE)
            print(f"  [OK] 已创建 .env，请根据需要修改 LLM 配置。")
            print(f"  [WARN] 请编辑 {ENV_FILE} 填入你的 API Key 后重新运行。")
        else:
            print(f"  [ERROR] env.example 也不存在！请检查项目完整性。")
            sys.exit(1)


# ──────────────────────────────────────────────
#  步骤 2：安装 Python 依赖 (uv sync)
# ──────────────────────────────────────────────

def step_python_deps():
    """使用 uv 同步 Python 依赖。"""
    print("\n" + "=" * 60)
    print("  [2/7] 安装 Python 依赖 (uv sync)")
    print("=" * 60)

    if not is_command_available("uv"):
        print("  [INFO] uv 未安装，正在自动安装...")
        if platform.system() == "Windows":
            run(["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"])
            # uv 安装后可能在当前 shell 中不可用，尝试找到它
            uv_path = Path.home() / ".cargo" / "bin" / "uv.exe"
            if uv_path.exists():
                run([str(uv_path), "sync", "--extra", "api", "--extra", "offline-llm", "--extra", "offline-storage"])
                return
        else:
            run(["curl", "-LsSf", "https://astral.sh/uv/install.sh"], shell=True)
            print("  [INFO] 请重新运行此脚本以使用新安装的 uv。")
            sys.exit(0)

    run(["uv", "sync", "--extra", "api", "--extra", "offline-llm", "--extra", "offline-storage"])
    print("  [OK] Python 依赖安装完成。")


# ──────────────────────────────────────────────
#  步骤 3：安装 Bun (前端运行时)
# ──────────────────────────────────────────────

def step_bun():
    """确保 Bun 已安装。优先用 npm 安装，失败则回退到 npm 构建。"""
    print("\n" + "=" * 60)
    print("  [3/7] 检查 Bun (前端构建工具)")
    print("=" * 60)

    global _use_npm_fallback
    _use_npm_fallback = False

    if is_bun_available():
        bun_ver = run_noerr(["bun", "--version"])
        print(f"  [OK] Bun 已安装: {bun_ver.stdout.strip() if bun_ver else 'unknown'}")
        return

    # 尝试通过 npm 安装 bun（Windows 上更可靠）
    print("  [INFO] Bun 未安装，尝试通过 npm 安装...")
    if is_command_available("npm"):
        result = run_noerr(["npm", "install", "-g", "bun"])
        if result and result.returncode == 0:
            time.sleep(1)
            if is_bun_available():
                bun_ver = run_noerr(["bun", "--version"])
                print(f"  [OK] Bun 安装成功: {bun_ver.stdout.strip() if bun_ver else '?'}")
                return
        print("  [WARN] npm 安装 bun 失败，将使用 npm 构建前端。")

    # 如果 npm 也没有，尝试 PowerShell 安装 bun
    if not is_command_available("npm"):
        print("  [INFO] 尝试通过 PowerShell 安装 Bun...")
        result = run_noerr(
            ["powershell", "-Command",
             "irm bun.sh/install.ps1 | iex"],
            shell=False
        )
        time.sleep(2)
        if is_bun_available():
            print("  [OK] Bun 安装成功。")
            return

    # 最终回退：使用 npm
    if is_command_available("npm"):
        print("  [INFO] 将使用 npm 作为回退方案构建前端。")
        _use_npm_fallback = True
    else:
        print("  [FATAL] 未找到 Bun 或 npm，无法构建前端。请先安装 Node.js。")
        sys.exit(1)


# ──────────────────────────────────────────────
#  步骤 4：构建前端
# ──────────────────────────────────────────────

def step_frontend_build():
    """安装前端依赖并构建。"""
    print("\n" + "=" * 60)
    print("  [4/7] 构建前端 (lightrag_webui)")
    print("=" * 60)

    use_npm = _use_npm_fallback

    if use_npm:
        # ── npm 回退路径 ──
        print("  [INFO] 使用 npm 安装依赖 + 构建前端...")
        print("  [INFO] npm install (首次可能较慢，请耐心等待)...")
        run(["npm", "install"], cwd=WEBUI_DIR)
        print("  [INFO] npx vite build...")
        run(["npx", "vite", "build"], cwd=WEBUI_DIR)
    else:
        if not is_bun_available():
            print("  [FATAL] Bun 不可用且 npm 回退未启用。")
            sys.exit(1)
        # 安装前端依赖
        print("  [INFO] 安装前端依赖 (bun install)...")
        run(["bun", "install", "--frozen-lockfile"], cwd=WEBUI_DIR)
        # 构建前端
        print("  [INFO] 构建前端产物 (bun run build)...")
        run(["bun", "run", "build"], cwd=WEBUI_DIR)

    dist_dir = WEBUI_DIR / "dist"
    # Vite 构建产物也可能输出到 lightrag/api/webui/
    pkg_webui = PROJECT_DIR / "lightrag" / "api" / "webui"
    built = False
    if dist_dir.exists():
        print(f"  [OK] 前端构建完成: {dist_dir}")
        built = True
    elif pkg_webui.exists() and list(pkg_webui.glob("index.html")):
        print(f"  [OK] 前端构建完成: {pkg_webui}")
        built = True
    if not built:
        print(f"  [ERROR] 前端构建失败，未找到构建产物。")
        sys.exit(1)


# ──────────────────────────────────────────────
#  步骤 5：终止旧进程
# ──────────────────────────────────────────────

def step_kill_old():
    """终止之前可能残留的服务进程。"""
    print("\n" + "=" * 60)
    print("  [5/7] 终止旧进程")
    print("=" * 60)
    kill_port(SERVER_PORT)


# ──────────────────────────────────────────────
#  步骤 6：创建数据库（如果不存在）
# ──────────────────────────────────────────────

def _load_dotenv():
    """简单解析 .env 中的 key=value（忽略注释和引号）。"""
    vars_ = {}
    if not ENV_FILE.exists():
        return vars_
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            vars_[key] = val
    return vars_


def step_create_database():
    """如果使用 PostgreSQL，自动创建数据库。"""
    print("\n" + "=" * 60)
    print("  [6/7] 检查/创建数据库")
    print("=" * 60)

    env_vars = _load_dotenv()

    # 检查是否使用了 PostgreSQL 存储
    storage_types = [
        env_vars.get("LIGHTRAG_KV_STORAGE", ""),
        env_vars.get("LIGHTRAG_DOC_STATUS_STORAGE", ""),
        env_vars.get("LIGHTRAG_GRAPH_STORAGE", ""),
        env_vars.get("LIGHTRAG_VECTOR_STORAGE", ""),
    ]
    uses_pg = any("PG" in s for s in storage_types)

    if not uses_pg:
        print("  [INFO] 未使用 PostgreSQL 存储，跳过数据库创建。")
        return

    pg_host = env_vars.get("POSTGRES_HOST", "localhost")
    pg_port = env_vars.get("POSTGRES_PORT", "5432")
    pg_user = env_vars.get("POSTGRES_USER", "lightrag")
    pg_password = env_vars.get("POSTGRES_PASSWORD", "")
    pg_db = env_vars.get("POSTGRES_DATABASE", "lightrag")

    print(f"  [INFO] PostgreSQL: {pg_host}:{pg_port}")
    print(f"  [INFO] 目标数据库: {pg_db}")

    try:
        import asyncpg
    except ImportError:
        print("  [WARN] asyncpg 未安装，正在安装...")
        run(["uv", "pip", "install", "asyncpg"])

    async def _ensure_db():
        """连接到 postgres 库，检查并创建目标数据库。"""
        # 先连到默认的 postgres 数据库
        conn = await asyncpg.connect(
            host=pg_host,
            port=int(pg_port),
            user=pg_user,
            password=pg_password,
            database="postgres",
        )
        try:
            # 检查目标数据库是否存在
            row = await conn.fetchrow(
                "SELECT 1 FROM pg_database WHERE datname = $1", pg_db
            )
            if row:
                print(f"  [OK] 数据库 '{pg_db}' 已存在。")
            else:
                print(f"  [INFO] 数据库 '{pg_db}' 不存在，正在创建...")
                await conn.execute(f'CREATE DATABASE "{pg_db}"')
                print(f"  [OK] 数据库 '{pg_db}' 创建成功。")
        finally:
            await conn.close()

        # 连到目标数据库，启用 pgvector 扩展
        conn2 = await asyncpg.connect(
            host=pg_host,
            port=int(pg_port),
            user=pg_user,
            password=pg_password,
            database=pg_db,
        )
        try:
            await conn2.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print(f"  [OK] pgvector 扩展已就绪。")
        except Exception:
            print(f"  [INFO] pgvector 扩展可能未安装，服务器创建表时会自动处理。")
        finally:
            await conn2.close()

    try:
        import asyncio
        asyncio.run(_ensure_db())
    except Exception as e:
        print(f"  [ERROR] 数据库操作失败: {e}")
        print(f"  [WARN] 请确认 PostgreSQL 服务可访问，且用户名密码正确。")
        print(f"  [WARN] 服务器启动后会自动尝试连接。")
        # 不退出，让 lightrag-server 自己处理连接问题


# ──────────────────────────────────────────────
#  步骤 7：启动服务
# ──────────────────────────────────────────────

def step_start():
    """启动 LightRAG 服务。"""
    print("\n" + "=" * 60)
    print("  [7/7] 启动 LightRAG 服务")
    print("=" * 60)

    if not VENV_PYTHON.exists():
        print(f"  [FATAL] 虚拟环境不存在: {VENV_PYTHON}")
        sys.exit(1)

    server_exe = PROJECT_DIR / ".venv" / "Scripts" / "lightrag-server.exe"
    if not server_exe.exists():
        # 尝试 source install (可编辑安装)
        server_exe = PROJECT_DIR / ".venv" / "Scripts" / "lightrag-server.exe"

    print(f"\n  {'─' * 50}")
    print(f"    LightRAG 正在启动...")
    print(f"    API 地址:  http://localhost:{SERVER_PORT}")
    print(f"    WebUI:     http://localhost:{SERVER_PORT}")
    print(f"    API 文档:  http://localhost:{SERVER_PORT}/docs")
    print(f"  {'─' * 50}\n")

    # 启动 lightrag-server，将工作目录设为项目根目录
    os.chdir(PROJECT_DIR)

    try:
        subprocess.run(
            [str(server_exe)],
            cwd=PROJECT_DIR,
            env={
                **os.environ,
                "PATH": str(VENV_PYTHON.parent) + os.pathsep + os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
            check=True,
            shell=False,
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] 服务已停止。")
    except Exception as e:
        print(f"\n[ERROR] 服务异常退出: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    print("""
  ╔══════════════════════════════════════════════╗
  ║     🚀  LightRAG 一键启动脚本              ║
  ║     项目: HKUDS/LightRAG                   ║
  ╚══════════════════════════════════════════════╝
""")
    os.chdir(PROJECT_DIR)

    step_env()
    step_python_deps()
    step_bun()
    step_frontend_build()
    step_kill_old()
    step_create_database()
    step_start()


if __name__ == "__main__":
    main()
