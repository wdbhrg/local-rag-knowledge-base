"""
================================================================================
app.py —— Streamlit 网页：侧边栏配模型 + 「问答」与「资料库」两页
================================================================================

Streamlit 怎么理解：你写的是普通 Python 脚本，从上到下执行；控件（输入框、按钮）
会触发「重新运行整页脚本」，通过 st.session_state 记住用户上次填的内容。

本文件做的事：
  - 连接 knowledge_base.sqlite（@st.cache_resource 长生命周期的连接）
  - 问答页：retrieve() 检索 → ask_llm_stream() 流式显示
  - 资料库页：列表、上传导入、删索引、全量重建；其中 reindex_all / upsert_document
    可在**后台线程**执行，界面用 fragment 轮询进度，避免长文档解析卡住页面。

依赖：ask_kb.py（检索与模型）、build_kb.py（建库与索引维护函数）。
================================================================================
"""

import json
import os
import subprocess
import sys
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from ask_kb import (
    ask_llm,
    ask_llm_stream,
    citation_snippet_for_ui,
    contexts_with_displayable_citations,
    escape_numeric_citations_for_markdown,
    retrieve,
    retrieve_with_notes,
    route_question,
    scoped_miss_looks_like_library_path_issue,
    self_rag_verify,
    skip_retrieval_for_route,
    strip_model_reference_section,
)
from ask_kb import _kb_feature_self_rag
from build_kb import (
    clear_configured_library_root,
    default_library_root,
    delete_document_from_index,
    effective_library_root,
    reindex_all,
    scan_files,
    upsert_document,
    write_configured_library_root,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_LOCAL_DEFAULT_BASE = "http://127.0.0.1:11434/v1"
_LOCAL_DEFAULT_MODEL = "kb-rag"


def _ollama_hosts_to_try() -> list[str]:
    hosts: list[str] = []
    raw = (os.environ.get("OLLAMA_HOST") or "").strip().rstrip("/")
    if raw:
        if not raw.lower().startswith("http"):
            raw = "http://" + raw
        hosts.append(raw)
    for h in (
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
    ):
        if h not in hosts:
            hosts.append(h)
    return hosts


def _requests_get_local(url: str, timeout: float = 6.0) -> requests.Response:
    sess = requests.Session()
    if hasattr(sess, "trust_env"):
        sess.trust_env = False
    return sess.get(
        url,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )


def _decode_cli_stdout(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def _parse_ollama_list_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "models" in data:
            return [str(m.get("name", "")).strip() for m in data["models"] if m.get("name")]
        if isinstance(data, list):
            return [
                str(x.get("name", "")).strip()
                for x in data
                if isinstance(x, dict) and x.get("name")
            ]
    except json.JSONDecodeError:
        pass
    names: list[str] = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t.upper().startswith("NAME"):
            continue
        if "\t" in t:
            first = t.split("\t", 1)[0].strip()
        else:
            parts = t.split()
            first = parts[0] if parts else ""
        if first and not first.upper() == "NAME":
            names.append(first)
    return names


def _ollama_cli_model_names() -> list[str]:
    bundled = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"
    )
    exe = bundled if bundled and os.path.isfile(bundled) else "ollama"
    run_kw: dict[str, Any] = {"capture_output": True, "timeout": 25}
    if sys.platform == "win32":
        run_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run([exe, "list"], **run_kw)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if p.returncode != 0:
        return []
    raw = p.stdout
    if raw is None:
        return []
    text = _decode_cli_stdout(raw)
    return [n for n in _parse_ollama_list_text(text) if n]


def _host_from_openai_base(base_v1: str) -> str:
    b = (base_v1 or "").rstrip("/")
    if b.lower().endswith("/v1"):
        return b[:-3]
    return b


def _endpoint_label_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse

        u = urlparse((url or "").strip())
        return u.netloc or "custom"
    except Exception:
        return "custom"


def _local_openai_base_candidates() -> list[tuple[str, str]]:
    """(OpenAI base …/v1, short UI tag)."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    extra = os.environ.get("KB_LOCAL_PROBE_BASES", "").strip()
    if extra:
        for part in extra.split(","):
            p = part.strip().rstrip("/")
            if not p:
                continue
            if not p.lower().startswith("http"):
                p = "http://" + p
            if not p.lower().endswith("/v1"):
                p = p + "/v1"
            key = p.rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                out.append((p.rstrip("/"), _endpoint_label_from_url(p)))
    for host in _ollama_hosts_to_try():
        u = host.rstrip("/") + "/v1"
        key = u.lower()
        if key not in seen:
            seen.add(key)
            out.append((u, "Ollama"))
    for base, tag in (
        ("http://127.0.0.1:1234/v1", "LM Studio"),
        ("http://127.0.0.1:8080/v1", "llama.cpp"),
        ("http://127.0.0.1:8000/v1", "vLLM"),
        ("http://127.0.0.1:5000/v1", "WebUI"),
        ("http://127.0.0.1:5001/v1", "KoboldCpp"),
    ):
        key = base.lower()
        if key not in seen:
            seen.add(key)
            out.append((base, tag))
    return out


def _probe_v1_model_ids(base_v1: str) -> list[str]:
    base = (base_v1 or "").rstrip("/")
    url = f"{base}/models"
    try:
        r = _requests_get_local(url, timeout=4.0)
        if r.status_code != 200:
            return []
        data = r.json()
    except (OSError, requests.RequestException, ValueError, TypeError):
        return []
    out: list[str] = []
    for item in (data or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or item.get("name") or "").strip()
        if mid and mid not in out:
            out.append(mid)
    return out


def _probe_ollama_api_tags_models(host_root: str) -> list[str]:
    url = host_root.rstrip("/") + "/api/tags"
    try:
        r = _requests_get_local(url, timeout=4.0)
        if r.status_code != 200:
            return []
        data = r.json()
    except (OSError, requests.RequestException, ValueError, TypeError):
        return []
    names: list[str] = []
    for m in (data or {}).get("models") or []:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _probe_all_local_llm_models() -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen_pair: set[tuple[str, str]] = set()
    cli_used = False
    for base_v1, tag in _local_openai_base_candidates():
        base_v1 = base_v1.rstrip("/")
        ids = list(_probe_v1_model_ids(base_v1))
        if not ids:
            host = _host_from_openai_base(base_v1)
            if ":11434" in host.replace("localhost", "127.0.0.1").lower():
                ids = list(_probe_ollama_api_tags_models(host))
        if not ids and not cli_used:
            host = _host_from_openai_base(base_v1)
            if ":11434" in host.replace("localhost", "127.0.0.1").lower():
                cli_names = _ollama_cli_model_names()
                if cli_names:
                    ids = list(cli_names)
                    cli_used = True
        for mid in ids:
            key = (base_v1, mid)
            if key in seen_pair:
                continue
            seen_pair.add(key)
            found.append(
                {"base": base_v1, "model": mid, "label": f"[{tag}] {mid}"}
            )
    found.sort(key=lambda x: x["label"].lower())
    return found


def _any_local_endpoint_reachable() -> bool:
    for base_v1, _tag in _local_openai_base_candidates():
        base_v1 = base_v1.rstrip("/")
        try:
            r = _requests_get_local(f"{base_v1}/models", timeout=2.5)
            if r.status_code == 200:
                return True
        except (OSError, requests.RequestException):
            pass
        host = _host_from_openai_base(base_v1)
        if ":11434" in host.replace("localhost", "127.0.0.1").lower():
            try:
                r = _requests_get_local(host.rstrip("/") + "/api/tags", timeout=2.5)
                if r.status_code == 200:
                    return True
            except (OSError, requests.RequestException):
                pass
    return False


def _local_llm_probe_diag_lines() -> list[str]:
    lines: list[str] = []
    for base_v1, tag in _local_openai_base_candidates():
        base_v1 = base_v1.rstrip("/")
        url = f"{base_v1}/models"
        try:
            r = _requests_get_local(url, timeout=4.0)
            if r.status_code == 200:
                try:
                    n = len((r.json() or {}).get("data") or [])
                except (TypeError, ValueError):
                    n = 0
                lines.append(f"[{tag}] {base_v1}/models：HTTP 200，约 {n} 条")
            else:
                lines.append(f"[{tag}] {base_v1}/models：HTTP {r.status_code}")
        except requests.RequestException as e:
            lines.append(f"[{tag}] {base_v1}/models：请求失败（{e.__class__.__name__}）")
        except (OSError, ValueError) as e:
            lines.append(f"[{tag}] {base_v1}/models：失败（{e}）")
        host = _host_from_openai_base(base_v1)
        if ":11434" in host.replace("localhost", "127.0.0.1").lower():
            tags_url = host.rstrip("/") + "/api/tags"
            try:
                r2 = _requests_get_local(tags_url, timeout=4.0)
                if r2.status_code == 200:
                    try:
                        n2 = len((r2.json() or {}).get("models") or [])
                    except (TypeError, ValueError):
                        n2 = 0
                    lines.append(f"[Ollama] {tags_url}：HTTP 200，{n2} 个模型")
                else:
                    lines.append(f"[Ollama] {tags_url}：HTTP {r2.status_code}")
            except requests.RequestException as e:
                lines.append(f"[Ollama] {tags_url}：{e.__class__.__name__}")
            except (OSError, ValueError) as e:
                lines.append(f"[Ollama] {tags_url}：{e}")
    cli_n = len(_ollama_cli_model_names())
    lines.append(f"ollama list：解析到 {cli_n} 个名称（仅 11434 且无 /v1/models 时作兜底）")
    lines.append(
        "说明：LM Studio / llama.cpp / vLLM 等需开启 OpenAI 兼容服务；"
        "自定义端口可在环境变量 KB_LOCAL_PROBE_BASES 中追加，例如 http://127.0.0.1:9999/v1"
    )
    if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
        lines.append("提示：已设置 HTTP(S)_PROXY；请求本机时已尽量不走代理。")
    return lines


@st.cache_data(ttl=30)
def _local_llm_select_entries(_refresh_token: int) -> list[dict[str, Any]]:
    rows = _probe_all_local_llm_models()
    if not rows:
        if _any_local_endpoint_reachable():
            hint = "（本机推理服务已连通，但未列出模型）请确认已加载模型并开放 /v1/models"
        else:
            hint = (
                "（未检测到本机 OpenAI 兼容接口）将使用默认地址与 kb-rag；"
                "可启动 LM Studio / Ollama / llama.cpp 后点「重新检测」"
            )
        return [
            {
                "base": _LOCAL_DEFAULT_BASE,
                "model": _LOCAL_DEFAULT_MODEL,
                "label": hint,
            }
        ]
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "base": row["base"],
                "model": row["model"],
                "label": row["label"],
            }
        )
    return out


def _entries_by_label(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["label"]: e for e in entries}


def _default_model_backend() -> str:
    raw = os.environ.get("KB_UI_MODEL_BACKEND", "").strip().lower()
    if raw in ("ollama", "local", "l"):
        return "local"
    if raw in ("cloud", "remote", "api"):
        return "cloud"
    b = (os.environ.get("OPENAI_BASE_URL") or "").strip().lower()
    if not b:
        return "local"
    if "127.0.0.1" in b or "localhost" in b or "[::1]" in b:
        return "local"
    if any(p in b for p in (":11434", ":1234", ":8080", ":8000", ":5000", ":5001")):
        return "local"
    return "cloud"


def _default_openai_base() -> str:
    """读取环境变量里的默认 API 地址；侧边栏首次打开时用来预填。"""
    return (
        os.environ.get("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        .strip()
        .rstrip("/")
    )


def _default_openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", "").strip()


def _default_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ARK_API_KEY", "")
    ).strip()


def _default_int(env_name: str, fallback: int) -> int:
    try:
        return max(1, int(os.environ.get(env_name, str(fallback))))
    except ValueError:
        return fallback


def _default_float(env_name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(env_name, str(fallback)))
    except ValueError:
        return fallback


def ensure_ui_config_state() -> None:
    """
    st.session_state 像 dict：给每个控件 key 存默认值，避免用户没点开侧边栏就 KeyError。

    只在「还没有这个 key」时写入，这样用户改过表单后不会被覆盖。
    """
    if "ui_model_backend" not in st.session_state:
        st.session_state["ui_model_backend"] = _default_model_backend()
    else:
        _bm = str(st.session_state.get("ui_model_backend", ""))
        if _bm == "ollama":
            st.session_state["ui_model_backend"] = "local"
    if "local_llm_refresh" not in st.session_state:
        st.session_state["local_llm_refresh"] = int(
            st.session_state.pop("ollama_refresh", 0) or 0
        )
    if "ui_local_llm_pick" not in st.session_state and "ui_ollama_pick" in st.session_state:
        st.session_state["ui_local_llm_pick"] = st.session_state.pop("ui_ollama_pick")
    if "ui_openai_base_url" not in st.session_state:
        st.session_state["ui_openai_base_url"] = _default_openai_base()
    if "ui_openai_model" not in st.session_state:
        st.session_state["ui_openai_model"] = _default_openai_model()
    if "ui_openai_api_key" not in st.session_state:
        st.session_state["ui_openai_api_key"] = _default_api_key()
    if "ui_kb_context_chars" not in st.session_state:
        st.session_state["ui_kb_context_chars"] = _default_int("KB_CONTEXT_CHARS", 2400)
    if "ui_kb_max_tokens" not in st.session_state:
        st.session_state["ui_kb_max_tokens"] = _default_int("KB_MAX_TOKENS", 4096)
    if "ui_kb_temperature" not in st.session_state:
        st.session_state["ui_kb_temperature"] = _default_float("KB_TEMPERATURE", 0.35)
    if "ui_library_root" not in st.session_state:
        st.session_state["ui_library_root"] = str(effective_library_root())
    if "ui_library_mkdir" not in st.session_state:
        st.session_state["ui_library_mkdir"] = False


def apply_ui_config_to_os() -> None:
    """
    ask_kb 里用 os.environ 读配置；把侧边栏表单同步到环境变量后，
    同一进程内后面的 ask_llm_stream / ask_llm 才能读到最新值。
    """
    backend = str(st.session_state.get("ui_model_backend", "cloud"))
    if backend == "local":
        token = int(st.session_state.get("local_llm_refresh", 0))
        entries = _local_llm_select_entries(token)
        by_l = _entries_by_label(entries)
        pick = str(st.session_state.get("ui_local_llm_pick", "") or "")
        row = by_l.get(pick)
        if row:
            os.environ["OPENAI_BASE_URL"] = str(row["base"]).strip().rstrip("/")
            os.environ["OPENAI_MODEL"] = str(row["model"]).strip()
        else:
            d = (
                entries[0]
                if entries
                else {"base": _LOCAL_DEFAULT_BASE, "model": _LOCAL_DEFAULT_MODEL}
            )
            os.environ["OPENAI_BASE_URL"] = str(d["base"]).strip().rstrip("/")
            os.environ["OPENAI_MODEL"] = str(d["model"]).strip()
        os.environ["OPENAI_API_KEY"] = ""
    else:
        base = str(st.session_state.get("ui_openai_base_url", "")).strip().rstrip("/")
        if base:
            os.environ["OPENAI_BASE_URL"] = base
        else:
            os.environ.pop("OPENAI_BASE_URL", None)
        model = str(st.session_state.get("ui_openai_model", "")).strip()
        if model:
            os.environ["OPENAI_MODEL"] = model
        else:
            os.environ.pop("OPENAI_MODEL", None)
        os.environ["OPENAI_API_KEY"] = str(
            st.session_state.get("ui_openai_api_key", "")
        ).strip()
    try:
        os.environ["KB_CONTEXT_CHARS"] = str(
            int(st.session_state.get("ui_kb_context_chars", 2400))
        )
    except (TypeError, ValueError):
        os.environ["KB_CONTEXT_CHARS"] = "2400"
    try:
        os.environ["KB_MAX_TOKENS"] = str(
            int(st.session_state.get("ui_kb_max_tokens", 4096))
        )
    except (TypeError, ValueError):
        os.environ["KB_MAX_TOKENS"] = "4096"
    try:
        os.environ["KB_TEMPERATURE"] = str(
            float(st.session_state.get("ui_kb_temperature", 0.35))
        )
    except (TypeError, ValueError):
        os.environ["KB_TEMPERATURE"] = "0.35"


def _save_library_path_from_ui() -> None:
    raw = str(st.session_state.get("ui_library_root", "")).strip()
    mkdir_ok = bool(st.session_state.get("ui_library_mkdir", False))
    if not raw:
        st.sidebar.error("路径不能为空。")
        return
    try:
        p = Path(raw).expanduser().resolve()
    except OSError as exc:
        st.sidebar.error(f"路径无效：{exc}")
        return
    if not p.exists():
        if mkdir_ok:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                st.sidebar.error(f"无法创建目录：{exc}")
                return
        else:
            st.sidebar.error("目录不存在。请检查路径，或勾选「不存在则自动创建」后重试。")
            return
    if not p.is_dir():
        st.sidebar.error("该路径不是文件夹。")
        return
    write_configured_library_root(p)
    os.environ["KB_LIBRARY_ROOT"] = str(p)
    st.session_state["ui_library_root"] = str(p)
    st.sidebar.success("资料路径已保存。")
    st.rerun()


def _reset_library_path_to_default() -> None:
    clear_configured_library_root()
    os.environ.pop("KB_LIBRARY_ROOT", None)
    st.session_state["ui_library_root"] = str(default_library_root())
    st.sidebar.success("已恢复为默认资料目录。")
    st.rerun()


def _tk_pick_directory(initial: str | None) -> tuple[str | None, bool]:
    """
    打开本机文件夹选择对话框。仅在运行 Streamlit 的机器上可用（本机双击启动时与用户同一台电脑）。

    返回 (所选路径或 None, 是否成功调用 Tk)。第二项为 False 表示环境无图形/Tk，应用输入框粘贴路径。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None, False

    initialdir = None
    if initial:
        ip = Path(initial.strip()).expanduser()
        try:
            ip = ip.resolve()
        except OSError:
            ip = None
        if ip is not None:
            if ip.is_dir():
                initialdir = str(ip)
            elif ip.parent.is_dir():
                initialdir = str(ip.parent)
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    picked: str | None = None
    try:
        picked = filedialog.askdirectory(
            initialdir=initialdir,
            title="选择资料库根目录",
        )
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return (picked if picked else None), True


def _norm_path_key_str(path_str: str) -> str:
    """与索引库 documents.path（posix）对齐的键，便于与磁盘路径匹配。"""
    try:
        return str(Path(path_str).expanduser().resolve().as_posix())
    except OSError:
        return str(Path(path_str).expanduser().as_posix())


def _norm_path_key_file(fp: Path) -> str:
    try:
        return str(fp.resolve().as_posix())
    except OSError:
        return str(fp.as_posix())


def resolved_materials_root() -> Path:
    """
    资料根目录：优先使用侧边栏里用户填写/选择且**已存在**的文件夹；
    否则回退到 effective_library_root()（环境变量 / kb_library_root.txt / 内置默认）。
    这样不点「保存资料路径」也能立刻按当前选中文件夹浏览与重建。
    """
    raw = str(st.session_state.get("ui_library_root", "")).strip()
    if raw:
        try:
            p = Path(raw).expanduser().resolve()
            if p.is_dir():
                return p
        except OSError:
            pass
    return effective_library_root()


def safe_filename(name: str) -> str:
    """用户上传文件只取「basename」，防止路径把戏把文件写到奇怪的地方。"""
    base = Path(name).name.strip()
    if not base or base in (".", ".."):
        return "upload.dat"
    return base


def _sidebar_shell_css() -> None:
    """压缩侧栏字间距与控件高度，与窄侧栏协调；兼容明暗主题用半透明线分割。"""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.25rem;
        }
        [data-testid="stSidebar"] .sb-zone {
            margin: 0 0 0.35rem 0;
        }
        [data-testid="stSidebar"] .sb-head {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            margin: 0 0 0.45rem 0;
            padding: 0 0 0.35rem 0;
            border-bottom: 1px solid rgba(128, 128, 128, 0.3);
        }
        [data-testid="stSidebar"] .sb-head-ic { font-size: 0.9rem; line-height: 1; }
        [data-testid="stSidebar"] .sb-head-tx {
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        [data-testid="stSidebar"] .stTextInput label div {
            font-size: 0.74rem !important;
        }
        [data-testid="stSidebar"] .stTextInput input {
            font-size: 0.8rem !important;
            padding: 0.35rem 0.5rem !important;
        }
        [data-testid="stSidebar"] .stCheckbox label span {
            font-size: 0.72rem !important;
        }
        [data-testid="stSidebar"] .stNumberInput label div,
        [data-testid="stSidebar"] .stSlider label div {
            font-size: 0.74rem !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            padding: 0.2rem 0.4rem !important;
            font-size: 0.76rem !important;
            min-height: 2rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
            gap: 0.35rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sb_head(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="sb-zone sb-head"><span class="sb-head-ic">{icon}</span>'
        f'<span class="sb-head-tx">{title}</span></div>',
        unsafe_allow_html=True,
    )


# Streamlit 页面元信息；必须尽量写在脚本靠前（在其它 st 控件之前）
st.set_page_config(page_title="知识库问答", page_icon="📚", layout="wide")
st.title("📚 知识库问答")
st.caption(
    "基于本地资料；侧栏可选「云端 API」或「本机 OpenAI 兼容接口」。正文请在关键论述后用论文式角标 [1][2]…；"
    "页面底部为统一引用列表。"
)


def _db_cache_key(db_path: Path) -> tuple[str, int, int]:
    """用于缓存资料列表：库文件变更后 mtime/size 变，缓存自动失效。"""
    s = db_path.stat()
    return (str(db_path.resolve()), s.st_mtime_ns, s.st_size)


@st.cache_data(show_spinner="正在加载资料列表…")
def fetch_library_rows(db_key: tuple[str, int, int]) -> list[tuple]:
    """
    @st.cache_data：函数入参相同则直接返回缓存结果，加快刷新。

    db_key 里带了数据库文件的 mtime/size：文件一变，键就变，缓存自动失效，列表永远是新的。
    """
    import sqlite3

    conn = sqlite3.connect(db_key[0], check_same_thread=False, timeout=15.0)
    conn.execute("PRAGMA busy_timeout=8000")
    conn.text_factory = str
    try:
        rows = conn.execute(
            """
            SELECT d.path, d.title, d.ext, d.mtime, COUNT(c.id) AS nchunks
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.id
            GROUP BY d.id
            ORDER BY d.path
            """
        ).fetchall()
    finally:
        conn.close()
    return rows


# 与 README 约定一致：库文件放在项目根目录（本文件同级）
db_path = Path("knowledge_base.sqlite")
if not db_path.exists():
    st.error("未找到 knowledge_base.sqlite。请先双击 start.bat 建库，或命令行执行：start.bat rebuild")
    st.stop()


@st.cache_resource
def _db_connection(path: str):
    """整个 Streamlit 进程复用同一个 sqlite 连接（配合 check_same_thread=False）。"""
    return sqlite3_connect(path)


def sqlite3_connect(path: str):
    """单独抽出，方便一眼看到「连接 + text_factory」配置。"""
    import sqlite3

    conn = sqlite3.connect(path, check_same_thread=False, timeout=15.0)
    conn.execute("PRAGMA busy_timeout=8000")
    try:
        cache_kib = int(os.environ.get("KB_SQLITE_CACHE_KIB", "32768"))
        cache_kib = max(2048, min(2_097_152, cache_kib))
    except ValueError:
        cache_kib = 32768
    conn.execute(f"PRAGMA cache_size={-cache_kib}")
    try:
        mmap_mb = int(os.environ.get("KB_SQLITE_MMAP_MB", "0"))
        mmap_mb = max(0, min(2048, mmap_mb))
        if mmap_mb > 0:
            conn.execute(f"PRAGMA mmap_size={mmap_mb * 1024 * 1024}")
    except Exception:
        pass
    conn.text_factory = str
    return conn


def _reindex_job_state() -> dict:
    key = "_async_reindex_job"
    if key not in st.session_state:
        st.session_state[key] = {
            "lock": threading.Lock(),
            "status": "idle",
            "summary": None,
            "error": None,
            "current_i": 0,
            "total": 0,
            "current_path": "",
        }
    return st.session_state[key]


def _import_job_state() -> dict:
    key = "_async_import_job"
    if key not in st.session_state:
        st.session_state[key] = {
            "lock": threading.Lock(),
            "status": "idle",
            "error": None,
            "target_label": "",
        }
    return st.session_state[key]


def _indexing_jobs_busy() -> bool:
    rj = _reindex_job_state()
    ij = _import_job_state()
    with rj["lock"]:
        r_busy = rj["status"] == "running"
    with ij["lock"]:
        i_busy = ij["status"] == "running"
    return r_busy or i_busy


def _reindex_worker(db_abs: str, root: Path, job: dict) -> None:
    import sqlite3

    lock = job["lock"]
    try:

        def cb(i: int, total: int, path: str) -> None:
            with lock:
                job["current_i"] = i
                job["total"] = total
                job["current_path"] = path

        conn = sqlite3.connect(db_abs, check_same_thread=False)
        conn.text_factory = str
        try:
            summary = reindex_all(conn, root, progress_callback=cb)
        finally:
            conn.close()
        with lock:
            job["summary"] = summary
            job["status"] = "done"
    except Exception as exc:
        with lock:
            job["error"] = str(exc)
            job["status"] = "error"


def _import_worker(db_abs: str, target: Path, job: dict) -> None:
    import sqlite3

    lock = job["lock"]
    try:
        conn = sqlite3.connect(db_abs, check_same_thread=False)
        conn.text_factory = str
        try:
            upsert_document(conn, target)
        finally:
            conn.close()
        with lock:
            job["status"] = "done"
    except Exception as exc:
        with lock:
            job["error"] = str(exc)
            job["status"] = "error"


def _invalidate_after_index_mutations() -> None:
    """后台线程提交写入后，刷新连接缓存与资料列表缓存。"""
    _db_connection.clear()
    fetch_library_rows.clear()


def _schedule_library_toast(payload: dict) -> None:
    """后台任务完成时写入，由资料库页在下一次全页 rerun 时展示（fragment 内 st.success 会在 rerun 后消失）。"""
    st.session_state["_pending_lib_toast"] = payload


def _flush_pending_library_toasts() -> None:
    pending = st.session_state.pop("_pending_lib_toast", None)
    if not pending:
        return
    kind = pending.get("kind")
    if kind == "reindex_ok":
        summary = pending.get("summary") or {}
        if summary.get("errors"):
            for fp, err in summary["errors"]:
                st.warning(f"{fp}: {err}")
        st.success(
            f"全量重建完成：处理文件 {summary.get('processed', 0)}，文档 {summary.get('documents', 0)}，"
            f"片段 {summary.get('chunks', 0)}，关联边 {summary.get('graph_edges', 0)}"
        )
    elif kind == "reindex_err":
        st.error(f"全量重建失败：{pending.get('message', '')}")
    elif kind == "import_ok":
        st.success(f"已保存并索引：{pending.get('path', '')}")
    elif kind == "import_err":
        st.error(f"建索引失败：{pending.get('message', '')}")


@st.fragment(run_every=timedelta(seconds=0.5))
def _library_background_jobs_ui() -> None:
    """轮询后台索引任务；重建与单文件导入在独立线程中执行，不阻塞主脚本。"""
    rj = _reindex_job_state()
    ij = _import_job_state()

    summary_done = None
    reindex_err = None
    with rj["lock"]:
        rs = rj["status"]
        ri, rt, rp = rj["current_i"], rj["total"], rj["current_path"]
        if rj["status"] == "done" and rj["summary"] is not None:
            summary_done = rj["summary"]
            rj["status"] = "idle"
            rj["summary"] = None
        elif rj["status"] == "error" and rj["error"]:
            reindex_err = rj["error"]
            rj["status"] = "idle"
            rj["error"] = None

    import_done = False
    import_err = None
    import_label = ""
    import_ilab = ""
    with ij["lock"]:
        ims = ij["status"]
        import_ilab = ij["target_label"]
        if ij["status"] == "done":
            import_done = True
            import_label = ij["target_label"]
            ij["status"] = "idle"
            ij["target_label"] = ""
        elif ij["status"] == "error" and ij["error"]:
            import_err = ij["error"]
            ij["status"] = "idle"
            ij["error"] = None

    if rs == "running" and rt > 0:
        st.progress(min(1.0, ri / rt))
        st.caption(f"全量重建：{ri} / {rt} · {Path(rp).name if rp else '…'}")
    elif rs == "running":
        st.caption("全量重建：正在准备文件列表…")

    if ims == "running":
        st.info(f"后台正在写入索引：{import_ilab or '…'}")

    if summary_done is not None:
        _invalidate_after_index_mutations()
        _schedule_library_toast({"kind": "reindex_ok", "summary": summary_done})
        st.rerun()

    if reindex_err is not None:
        _invalidate_after_index_mutations()
        _schedule_library_toast({"kind": "reindex_err", "message": reindex_err})
        st.rerun()

    if import_done:
        _invalidate_after_index_mutations()
        _schedule_library_toast({"kind": "import_ok", "path": import_label})
        st.rerun()

    if import_err is not None:
        _invalidate_after_index_mutations()
        _schedule_library_toast({"kind": "import_err", "message": import_err})
        st.rerun()


conn = _db_connection(str(db_path.resolve()))

ensure_ui_config_state()

# ---------- 侧边栏：资料路径、模型与生成参数（写入 session_state + 同步到 os.environ）----------
with st.sidebar:
    _sidebar_shell_css()
    _sb_head("📂", "资料库位置")
    st.text_input(
        "资料目录",
        key="ui_library_root",
        help="浏览或粘贴路径后立即参与列表、问答与重建；「保存」写入 kb_library_root.txt。",
        label_visibility="visible",
    )
    _br1, _br2 = st.columns(2)
    with _br1:
        if st.button("浏览…", use_container_width=True, key="lib_btn_pick_folder"):
            cur = str(st.session_state.get("ui_library_root", "")).strip()
            picked, tk_ok = _tk_pick_directory(cur or None)
            if picked:
                st.session_state["ui_library_root"] = picked
                st.rerun()
            elif not tk_ok:
                st.sidebar.error("无法打开图形选文件夹，请手动粘贴路径。")
    with _br2:
        if st.button("保存", use_container_width=True, key="lib_btn_save_root"):
            _save_library_path_from_ui()
    if st.button("恢复默认", use_container_width=True, key="lib_btn_reset_root"):
        _reset_library_path_to_default()
    st.checkbox(
        "保存时自动建目录",
        key="ui_library_mkdir",
        help="勾选后，保存路径时若文件夹不存在会尝试创建。",
    )
    st.caption("环境变量 KB_LIBRARY_ROOT 优先；默认见 build_kb。")

    st.divider()
    _sb_head("🔌", "模型接口")
    st.radio(
        "模型来源",
        options=["cloud", "local"],
        format_func=lambda x: "云端 OpenAI 兼容" if x == "cloud" else "本机 OpenAI 兼容",
        key="ui_model_backend",
        horizontal=True,
    )
    _backend = str(st.session_state.get("ui_model_backend", "cloud"))
    if _backend == "local":
        st.caption(
            "自动探测本机 **Ollama / LM Studio / llama.cpp / vLLM** 等（/v1/models）；"
            "Ollama 亦可运行 **start.bat setup** 创建 kb-rag。"
        )
        _tok = int(st.session_state.get("local_llm_refresh", 0))
        _entries = _local_llm_select_entries(_tok)
        _labels = [e["label"] for e in _entries]
        st.selectbox(
            "本机模型",
            options=_labels,
            key="ui_local_llm_pick",
            help="探测常见本机端口；自定义地址可设 KB_LOCAL_PROBE_BASES（逗号分隔的 …/v1）。",
        )
        if st.button("重新检测本机模型", use_container_width=True, key="btn_local_llm_refresh"):
            st.session_state["local_llm_refresh"] = _tok + 1
            st.session_state.pop("ui_local_llm_pick", None)
            _local_llm_select_entries.clear()
            st.rerun()
        with st.expander("连接与诊断", expanded=False):
            for _line in _local_llm_probe_diag_lines():
                st.text(_line)
    else:
        st.text_input(
            "API Base URL",
            key="ui_openai_base_url",
            help="OpenAI 兼容地址。豆包示例：https://ark.cn-beijing.volces.com/api/v3",
        )
        st.text_input(
            "模型 ID",
            key="ui_openai_model",
            help="接入点 ID，如 ep-xxxx（OPENAI_MODEL）",
        )
        st.text_input(
            "API Key",
            type="password",
            key="ui_openai_api_key",
            help="仅当前会话请求使用，不入库（OPENAI_API_KEY）",
        )

    st.divider()
    _sb_head("🎛️", "生成与上下文")
    st.number_input(
        "片段字数上限",
        min_value=500,
        max_value=6000,
        step=100,
        key="ui_kb_context_chars",
        help="KB_CONTEXT_CHARS：每条检索片段喂给模型的最大字符数",
    )
    st.number_input(
        "回复 tokens 上限",
        min_value=256,
        max_value=8192,
        step=256,
        key="ui_kb_max_tokens",
        help="KB_MAX_TOKENS：模型生成长度上限",
    )
    st.slider(
        "温度",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="ui_kb_temperature",
        help="KB_TEMPERATURE：越高越发散",
    )

    st.divider()
    if st.button("用环境变量重置表单", use_container_width=True):
        for k in (
            "ui_model_backend",
            "local_llm_refresh",
            "ollama_refresh",
            "ui_local_llm_pick",
            "ui_ollama_pick",
            "ui_openai_base_url",
            "ui_openai_model",
            "ui_openai_api_key",
            "ui_kb_context_chars",
            "ui_kb_max_tokens",
            "ui_kb_temperature",
            "ui_library_root",
            "ui_library_mkdir",
        ):
            st.session_state.pop(k, None)
        _local_llm_select_entries.clear()
        ensure_ui_config_state()
        st.rerun()

apply_ui_config_to_os()

# 资料根目录：侧栏若为有效文件夹则立即采用；否则环境变量 / kb_library_root.txt / 内置默认
materials_root = resolved_materials_root()

_flush_pending_library_toasts()

# ---------- 顶部导航：问答 vs 资料库 ----------
nav = st.radio(
    "页面",
    ["问答", "资料库"],
    horizontal=True,
    label_visibility="collapsed",
    key="ui_main_nav",
)

if _indexing_jobs_busy():
    _library_background_jobs_ui()

if nav == "资料库":
    # ---------- 资料库页：查看列表、删除、上传、全量重建 ----------
    st.subheader("资料库管理")
    st.info(f"**当前资料根目录**：`{materials_root}`")

    raw_sidebar = str(st.session_state.get("ui_library_root", "")).strip()
    if raw_sidebar:
        try:
            if not Path(raw_sidebar).expanduser().resolve().is_dir():
                st.warning(
                    "侧边栏中的路径不是有效文件夹，已回退为已保存配置或默认目录。"
                    f"当前实际使用：`{materials_root}`"
                )
        except OSError:
            st.warning(f"侧边栏路径无法解析，当前使用：`{materials_root}`")

    index_rows = conn.execute(
        """
        SELECT d.path, d.title, d.ext, d.mtime, COUNT(c.id) AS nchunks
        FROM documents d
        LEFT JOIN chunks c ON c.doc_id = d.id
        GROUP BY d.id
        """
    ).fetchall()
    index_map = {
        _norm_path_key_str(p): (p, title, ext, mtime, nchunks)
        for p, title, ext, mtime, nchunks in index_rows
    }

    table: list[dict[str, str | int]] = []
    if materials_root.is_dir():
        for fp in scan_files(materials_root):
            k = _norm_path_key_file(fp)
            rec = index_map.get(k)
            n_ch = int(rec[4]) if rec else 0
            if not rec:
                note = "尚未入库；全量重建后根据路径匹配。"
            elif n_ch == 0:
                note = (
                    "有文档记录但 0 片段：多为上次快速跳过未校验片段、正文未提取（扫描版 PDF）"
                    "或空文件。请点击「全量重建索引」；若该行报错见上文。"
                )
            else:
                note = "—"
            table.append(
                {
                    "文件名": fp.name,
                    "扩展名": fp.suffix.lower(),
                    "已索引": "是" if rec else "否",
                    "片段数": n_ch,
                    "索引标题": (rec[1] or "—") if rec else "—",
                    "说明": note,
                    "完整路径": str(fp.resolve().as_posix()),
                }
            )
        st.caption(f"主表：在当前资料根目录下共扫描到 **{len(table)}** 个可索引文件（递归）。")
        st.dataframe(table, use_container_width=True, hide_index=True)
        if len(table) == 0:
            st.info("该文件夹下暂无 epub / pdf / txt。放入书籍后点击「全量重建索引」，或使用下方上传。")
    else:
        st.warning(
            f"资料根目录不存在或不是文件夹：`{materials_root}`。"
            "请在侧边栏选定有效路径，必要时勾选「不存在则自动创建」后保存。"
        )

    lib_rows = fetch_library_rows(_db_cache_key(db_path))
    if not lib_rows and not table and materials_root.is_dir():
        st.warning("索引仍为空。将文件放入当前文件夹后点击「全量重建索引」以建立全文检索。")

    st.markdown("#### 从知识库删除")
    del_options = [r[0] for r in lib_rows]
    _chunks_by_path = {r[0]: r[4] for r in lib_rows}

    def _fmt_path(p: str) -> str:
        n = _chunks_by_path.get(p, 0)
        return f"{Path(p).name}（{n} 段）"

    to_del = st.multiselect(
        "勾选要移除的资料（默认同时删除磁盘文件，可取消下方勾选仅清索引）",
        options=del_options,
        format_func=_fmt_path,
        key="lib_multiselect_delete",
    )
    rm_disk = st.checkbox("同时删除磁盘文件", value=True, key="lib_rm_disk")
    if st.button("执行删除", type="primary", key="lib_btn_delete"):
        if not to_del:
            st.warning("请先选择至少一条。")
        else:
            for pstr in to_del:
                delete_document_from_index(conn, pstr)
                if rm_disk:
                    fp = Path(pstr)
                    if fp.is_file():
                        try:
                            fp.unlink()
                        except OSError as exc:
                            st.error(f"无法删除文件 {fp}: {exc}")
            fetch_library_rows.clear()
            st.success(f"已处理 {len(to_del)} 条。")
            st.rerun()

    st.divider()
    st.markdown("#### 添加资料")
    up = st.file_uploader("上传 epub / pdf / txt", type=["epub", "pdf", "txt"], key="lib_uploader")
    dest_choice = st.radio("保存到", ["资料库根目录", "PDF 子目录"], horizontal=True, key="lib_dest")
    default_name = safe_filename(up.name) if up else ""
    save_name = st.text_input("保存文件名", value=default_name, key="lib_save_name")
    allow_overwrite = st.checkbox("允许覆盖同名文件", value=False, key="lib_overwrite")

    if st.button("导入并写入索引", key="lib_btn_import"):
        if not up:
            st.warning("请先选择文件。")
        elif _indexing_jobs_busy():
            st.warning("已有后台索引任务在进行中（全量重建或另一导入），请待其完成后再试。")
        else:
            fname = safe_filename(save_name or up.name)
            ext = Path(fname).suffix.lower()
            if ext not in {".epub", ".pdf", ".txt"}:
                st.error("仅支持 .epub / .pdf / .txt")
                st.stop()
            dest_dir = materials_root / "PDF" if dest_choice == "PDF 子目录" else materials_root
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = (dest_dir / fname).resolve()
            try:
                # 确保 target 在资料根目录之下，防止 `..` 之类逃出目录
                target.relative_to(materials_root.resolve())
            except ValueError:
                st.error("非法路径。")
                st.stop()
            if target.exists() and not allow_overwrite:
                st.error("文件已存在。请改名或勾选「允许覆盖同名文件」。")
                st.stop()
            try:
                target.write_bytes(up.getvalue())
            except OSError as exc:
                st.error(f"写入失败：{exc}")
                st.stop()
            job = _import_job_state()
            with job["lock"]:
                job["error"] = None
                job["status"] = "running"
                job["target_label"] = str(target)
            threading.Thread(
                target=_import_worker,
                args=(str(db_path.resolve()), target, job),
                daemon=True,
            ).start()
            st.info(f"已在后台为「{target.name}」建立索引；进度见下方，完成后会自动刷新列表。")

    st.divider()
    st.markdown("#### 全量重建索引")
    st.caption("与 `build_kb.py` / `start.bat rebuild` 相同：扫描资料根目录下全部 epub/pdf/txt，移除已不存在文件对应索引，并更新内容变更。")
    st.caption("重建在**后台线程**运行，可继续浏览侧栏；下方会显示进度（大 PDF/EPUB 不再卡死整页）。")
    if st.button("全量重建索引", key="lib_btn_reindex"):
        if _indexing_jobs_busy():
            st.warning("已有后台索引任务在进行中，请待其完成后再试。")
        else:
            job = _reindex_job_state()
            with job["lock"]:
                job["summary"] = None
                job["error"] = None
                job["current_i"] = 0
                job["total"] = 0
                job["current_path"] = ""
                job["status"] = "running"
            threading.Thread(
                target=_reindex_worker,
                args=(str(db_path.resolve()), materials_root.resolve(), job),
                daemon=True,
            ).start()
            st.info("已在后台开始全量重建，进度见下方；完成后会自动刷新列表。")

    if st.button("刷新资料列表缓存", key="lib_btn_refresh_list", help="若列表未及时更新，可手动刷新"):
        fetch_library_rows.clear()
        st.rerun()

else:
    # ---------- 问答页：检索 + 流式/非流式生成 + 底部引用列表 ----------
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "输入你的问题",
            placeholder="例如：请系统说明 RAG 各阶段要点与模块划分",
            height=100,
            key="chat_question",
        )
        ask_btn = st.button("开始提问", type="primary", key="chat_ask")
    with col2:
        top_k = st.slider("检索片段数", min_value=4, max_value=16, value=4, step=1, key="chat_topk")
        st.caption("更多片段通常回答更充实，模型耗时略增。")
        st.caption(
            f"检索范围：**侧栏「资料目录」**（当前：`{materials_root}`）。与建库时的 `--root` 不一致时，命中可能为空。"
        )

    if ask_btn:
        # 点一次「开始提问」会 rerun；用 button 而不是每字都触发，省 API 调用
        if not question.strip():
            st.warning("请先输入问题。")
            st.stop()

        apply_ui_config_to_os()

        q = question.strip()
        try:
            st.toast("已收到问题，正在处理…", icon="⏳")
        except Exception:
            pass
        route = route_question(q)
        route_label = {"kb": "资料库", "general": "常识/通用", "code": "代码/技术"}.get(
            route, route
        )

        retrieval_skipped = skip_retrieval_for_route(route, q)
        cached_ans: str | None = None
        if retrieval_skipped:
            contexts = []
            search_q = q
            rewrite_note = None
        else:
            ack = st.empty()
            ack.info("⏳ **已收到**，正在检索资料库…")
            contexts, search_q, rewrite_note, cached_ans = retrieve_with_notes(
                conn, q, top_k, library_root=materials_root
            )
            ack.empty()

        # 无片段且非寒暄：强制按「资料库」系统提示走模型，避免 route=general 时落入「无书库的通用助手」而完全不提知识库。
        llm_route = route
        if not contexts and not retrieval_skipped:
            llm_route = "kb"

        if not contexts and not retrieval_skipped:
            if scoped_miss_looks_like_library_path_issue(conn, search_q, materials_root):
                st.warning(
                    "在**当前侧栏资料目录**下没有检索到片段，但**整库**里仍有 FTS 命中样本且均不在该目录下。"
                    "请将侧栏「资料目录」设为建库时的同一根路径并点「保存」，或把书放入该目录后「全量重建索引」。"
                )
            else:
                st.info(
                    "未检索到可引用片段。下方将说明可能原因（关键词、索引是否为空等）；确认已建库且路径一致。"
                )

        meta_parts = [f"路由：{route_label}"]
        if rewrite_note:
            meta_parts.append(rewrite_note)
        st.caption(" · ".join(meta_parts))

        st.subheader("回答")
        answer_box = st.empty()
        text = ""
        if cached_ans:
            st.success("语义缓存命中：已跳过检索与模型调用。")
            text = strip_model_reference_section(cached_ans.strip())
            answer_box.markdown(
                escape_numeric_citations_for_markdown(text or "（缓存为空）")
            )
        else:
            full: list[str] = []
            try:
                for piece in ask_llm_stream(q, contexts, llm_route):
                    full.append(piece)
                    answer_box.markdown(
                        escape_numeric_citations_for_markdown("".join(full))
                    )
                text = strip_model_reference_section("".join(full).strip())
                if not text:
                    with st.spinner("流式不可用，正在非流式生成..."):
                        text = strip_model_reference_section(
                            ask_llm(q, contexts, llm_route) or ""
                        )
                    answer_box.markdown(
                        escape_numeric_citations_for_markdown(
                            text or "模型调用失败，请检查 API 配置。"
                        )
                    )
                else:
                    answer_box.markdown(
                        escape_numeric_citations_for_markdown(text)
                    )
            except Exception as exc:
                st.warning(f"流式输出中断：{exc}")
                text = strip_model_reference_section("".join(full).strip())
                if not text:
                    with st.spinner("流式不可用，正在非流式生成..."):
                        text = strip_model_reference_section(
                            ask_llm(q, contexts, llm_route) or ""
                        )
                    answer_box.markdown(
                        escape_numeric_citations_for_markdown(
                            text or "模型调用失败，请检查 API 配置。"
                        )
                    )
                else:
                    answer_box.markdown(
                        escape_numeric_citations_for_markdown(text)
                    )

        try:
            from semantic_cache import store_cached_answer

            if not cached_ans and text and not retrieval_skipped:
                store_cached_answer(conn, q, text)
        except Exception:
            pass

        if _kb_feature_self_rag() and text and contexts:
            verdict = self_rag_verify(q, contexts, text)
            if not verdict.get("grounded", True):
                rq = (verdict.get("refined_search") or "").strip() or search_q
                with st.spinner(
                    "Self-RAG：检索与答案一致性不足，正在二次检索并重答…"
                ):
                    contexts2 = retrieve(conn, rq, top_k, library_root=materials_root)
                if contexts2:
                    text2 = strip_model_reference_section(
                        ask_llm(q, contexts2, llm_route) or ""
                    )
                    if text2:
                        text = text2
                        contexts = contexts2
                        answer_box.markdown(
                            escape_numeric_citations_for_markdown(text)
                        )
                        reason = (verdict.get("reason") or "").strip()
                        if reason:
                            st.caption(f"Self-RAG 已更新：{reason}")
                else:
                    st.caption(
                        "Self-RAG 二次检索仍无命中，保留上轮回答；可尝试改写问题关键词。"
                    )

        st.divider()
        st.subheader("引用")
        st.caption(
            "与回答角标 [1]、[2]… 顺序一致。仅列出清洗后仍有正文可展示的片段（参考文献/词汇表等不会占位出现）。"
        )
        show_ctx = contexts_with_displayable_citations(contexts, len(contexts))
        for i, c in enumerate(show_ctx, start=1):
            pth = c.get("path") or ""
            body = citation_snippet_for_ui(str(c.get("text") or ""))
            fname = Path(pth).name if pth else "未知文件"
            chunk_i = c.get("chunk_index")
            title = (c.get("title") or "").strip()
            head = f"[{i}] {fname}"
            if chunk_i is not None:
                head += f" · 片段 {chunk_i}"
            with st.expander(head, expanded=False):
                if title and title != Path(pth).stem:
                    st.caption(f"索引标题：{title}")
                if pth:
                    st.caption(f"`{pth}`")
                st.text(body)
