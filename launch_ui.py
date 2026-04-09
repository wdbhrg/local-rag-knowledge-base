"""
launch_ui.py —— 「一键启动网页」的小启动器（由 start.bat 调用）

为什么需要它：
  - Streamlit 默认固定端口；若 8501 已被占用会直接报错。
  - 这里先在 8510–8539（或环境变量指定）里挑一个本机空闲端口，再带参数启动 streamlit。

流程摘要：
  1) 切到项目根目录（本文件所在目录）
  2) 优先 .conda_env\\python.exe，其次 .venv\\Scripts\\python.exe，否则当前解释器
  3) subprocess 执行：python -m streamlit run app.py --server.port <端口>

环境变量：
  - KB_UI_PORT 或 ST_PORT：强制指定端口（纯数字）
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys

# 本脚本所在目录 = 项目根（app.py、 knowledge_base.sqlite 预期在这里）
ROOT = os.path.dirname(os.path.abspath(__file__))


def pick_port(start: int = 8510, span: int = 30) -> int:
    """
    端口选择：用户指定 → 否则在 [start, start+span) 里试 bind；
    再不行就让操作系统分配随机可用端口（bind 0）。
    """
    raw = os.environ.get("KB_UI_PORT") or os.environ.get("ST_PORT") or ""
    if raw.strip().isdigit():
        return int(raw.strip())
    for port in range(start, start + span):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))  # 能绑上就说明这个端口此刻空闲
            return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> int:
    """返回进程退出码：streamlit 退出多少，我们就退回多少（给 bat 判断是否成功）。"""
    os.chdir(ROOT)
    conda_py = os.path.join(ROOT, ".conda_env", "python.exe")
    venv_py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(conda_py):
        py = conda_py
    elif os.path.isfile(venv_py):
        py = venv_py
    else:
        py = sys.executable

    port = pick_port()
    url = f"http://127.0.0.1:{port}"
    print(f"[KB-UI] Open in browser: {url}")
    app_py = os.path.join(ROOT, "app.py")
    cmd = [
        py,
        "-m",
        "streamlit",
        "run",
        app_py,
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
    ]
    # subprocess.call：阻塞直到用户关掉 Streamlit；返回值是子进程退出码
    return int(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
