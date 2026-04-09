"""
================================================================================
ask_kb.py —— 「从知识库里查资料 + 让大模型回答」
================================================================================

这就是经典的 RAG（Retrieval-Augmented Generation）里后半段 + 前半段检索：

  1) retrieve()：FTS5 + bm25 多取候选，jieba TF-IDF 关键词 + 覆盖率/书名重排，并限制单文档片段数；
     无命中时降级 LIKE；把你的问题对应的最相关几段文字（chunks）交给下游。
  2) _build_messages()：把搜到的段落塞进「提示词」，告诉模型「只能根据这些片段回答」。
  3) ask_llm / ask_llm_stream：通过 HTTP 调用兼容 OpenAI Chat Completions 的接口
     （豆包 Ark、很多国内网关都是这个形状）。
  4) 若没配 API Key 或调用失败：fallback_answer() 只把检索到的原文摘要列出来。

  可选（环境变量，默认开启若干项）：查询重写、路由（kb/general/code）、Self-RAG 二次检索。

命令行：python ask_kb.py --db xxx.sqlite --q "你的问题"

初学者可先读 retrieve()，再读 ask_llm()，最后看 main()。
================================================================================
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import Any, Iterator, Literal

import httpx
import jieba
import jieba.analyse

# 复用 httpx.Client：连接池 + 长超时，避免每次问话新建连接。
_HTTPX: httpx.Client | None = None

# 同一连接只跑一次 chunks 元数据列迁移，避免每次 retrieve 都 PRAGMA/ALTER。
_CHUNK_META_CONN_IDS: set[int] = set()

# 未设置 OPENAI_BASE_URL 时：默认同 kb_project_local，走本机 Ollama；亦可设 KB_DEFAULT_API_BACKEND=cloud 使用下方云端默认址
_DEFAULT_LOCAL_BASE = "http://127.0.0.1:11434/v1"
_DEFAULT_CLOUD_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _resolved_openai_base_url() -> str:
    raw = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if raw:
        return raw.rstrip("/")
    if os.environ.get("KB_DEFAULT_API_BACKEND", "").strip().lower() in (
        "cloud",
        "remote",
        "api",
    ):
        return _DEFAULT_CLOUD_BASE.rstrip("/")
    return _DEFAULT_LOCAL_BASE.rstrip("/")


def _is_ollama_local_base(url: str) -> bool:
    if os.environ.get("KB_LOCAL_LLM", "").strip().lower() in ("1", "true", "yes"):
        return True
    u = url.lower().replace(" ", "")
    return (
        "127.0.0.1:11434" in u
        or "localhost:11434" in u
        or "[::1]:11434" in u
    )


_LOCAL_OPENAI_DEFAULT_PORTS: frozenset[int] = frozenset(
    {11434, 1234, 8080, 8000, 5000, 5001}
)


def _local_openai_port_set() -> frozenset[int]:
    s = set(_LOCAL_OPENAI_DEFAULT_PORTS)
    raw = (os.environ.get("KB_LOCAL_OPENAI_PORTS") or "").strip()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            s.add(int(p, 10))
        except ValueError:
            continue
    return frozenset(s)


def _is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    h = hostname.lower().strip("[]")
    return h in ("127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1")


def _is_local_openai_compatible_base(url: str) -> bool:
    """本机常见 OpenAI 兼容服务（Ollama / LM Studio / llama.cpp 等），无密钥时使用占位 Key。"""
    if os.environ.get("KB_LOCAL_LLM", "").strip().lower() in ("1", "true", "yes"):
        return True
    raw = (url or "").strip()
    if not raw:
        return False
    try:
        from urllib.parse import urlparse

        pr = urlparse(raw)
    except Exception:
        return False
    if not _is_loopback_hostname(pr.hostname):
        return False
    port = pr.port
    if port is None:
        return False
    return port in _local_openai_port_set()


def _effective_api_key(base_url: str) -> str:
    key = (
        os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ARK_API_KEY", "")
    ).strip()
    if key:
        return key
    if _is_local_openai_compatible_base(base_url):
        return (
            os.environ.get("KB_LOCAL_API_KEY_PLACEHOLDER", "ollama").strip()
            or "ollama"
        )
    return ""


def _is_deepseek_r1_model(model: str) -> bool:
    return "deepseek-r1" in (model or "").lower()


def _ollama_use_reasoning_effort_none(model: str) -> bool:
    flag = (os.environ.get("KB_OLLAMA_THINK") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return False
    if flag in ("0", "false", "no", "off"):
        return True
    return _is_deepseek_r1_model(model)


def _deepseek_r1_doc_qa_system_suffix() -> str:
    if os.environ.get("KB_DEEPSEEK_R1_DOC_MODE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""
    return (
        "\n\n【本轮任务定位】用户提供的「可用资料片段」为其本地已索引文献摘录，用途为个人学习、检索与整理。\n"
        "你须严格基于这些片段作答并按要求标注 [n]；凡片段中已出现的主题、术语、数据、步骤与观点，应如实概括或复述，"
        "不得以笼统的合规、道德或「我不能」类话术拒绝仅针对该文献内容的提问。\n"
        "禁止编造片段中不存在的信息；禁止输出与用户问题及片段无关的长篇说教。片段不足以回答时明确写「资料未提及」。"
    )


def _merge_ollama_chat_payload(payload: dict[str, Any], model: str) -> None:
    if not _is_ollama_local_base(_resolved_openai_base_url()):
        return
    if _ollama_use_reasoning_effort_none(model):
        payload["reasoning"] = {"effort": "none"}


def _rag_tuned_extra_system() -> str:
    if os.environ.get("KB_RAG_TUNED", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""
    return (
        "\n补充要求（资料问答）：逐条对照「可用资料片段」编号 [1][2]… 取用内容；"
        "同一句若综合多段，可写 [1][2]。禁止使用资料未给出的页码、章节号或外部书名。"
        "优先复述资料中的术语与步骤；若只能推断，请写明「依据不足，以下为推论」。"
    )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _httpx_client() -> httpx.Client:
    global _HTTPX
    if _HTTPX is None:
        pool = int(os.environ.get("KB_HTTP_POOL_MAXSIZE", "32"))
        pool = max(4, min(128, pool))
        _HTTPX = httpx.Client(
            timeout=httpx.Timeout(180.0, connect=30.0),
            limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
            headers={"User-Agent": "kb_project/1.0", "Accept-Charset": "utf-8"},
        )
    return _HTTPX


def _ensure_chunk_meta_once(conn: sqlite3.Connection) -> None:
    cid = id(conn)
    if cid in _CHUNK_META_CONN_IDS:
        return
    try:
        from build_kb import _ensure_chunk_meta_columns

        _ensure_chunk_meta_columns(conn)
        _CHUNK_META_CONN_IDS.add(cid)
    except Exception:
        pass


def _decode_sse_line(raw: bytes | None) -> str:
    """流式接口常返回 SSE（Server-Sent Events）一行行数据；这里把字节安全 decode 成 str。"""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _kb_context_chars() -> int:
    """每段检索结果截取前多少个字符喂给模型；太长费钱费时间，太短上下文不够。"""
    return max(500, int(os.environ.get("KB_CONTEXT_CHARS", "2400")))


def _kb_max_tokens() -> int:
    return max(256, int(os.environ.get("KB_MAX_TOKENS", "4096")))


def _kb_temperature() -> float:
    return float(os.environ.get("KB_TEMPERATURE", "0.35"))


def _kb_retrieve_oversample() -> int:
    """先多取若干倍候选，再在内存里重排/去冗余，提高召回质量。"""
    try:
        return max(1, min(12, int(os.environ.get("KB_RETRIEVE_OVERSAMPLE", "4"))))
    except ValueError:
        return 4


def _kb_rerank_cover_weight() -> float:
    try:
        return max(0.0, float(os.environ.get("KB_RERANK_COVER_WEIGHT", "1.25")))
    except ValueError:
        return 1.25


def _kb_rerank_title_weight() -> float:
    try:
        return max(0.0, float(os.environ.get("KB_RERANK_TITLE_WEIGHT", "0.35")))
    except ValueError:
        return 0.35


def _kb_rerank_strict_weight() -> float:
    """核心查询词（TF-IDF 前列）在片段中命中比例，用于近似 AND 精度。"""
    try:
        return max(0.0, float(os.environ.get("KB_RERANK_STRICT_WEIGHT", "0.55")))
    except ValueError:
        return 0.55


def _kb_mmr_lambda() -> float:
    """0 关闭；0~1 在重排后做 MMR 去冗余（越大越偏重相关性）。"""
    try:
        return max(0.0, min(1.0, float(os.environ.get("KB_MMR_LAMBDA", "0"))))
    except ValueError:
        return 0.0


def _kb_max_chunks_per_doc() -> int:
    """最终 top_k 里，同一本书最多保留几段，避免上下文被单一大文档占满。"""
    try:
        return max(1, min(8, int(os.environ.get("KB_MAX_CHUNKS_PER_DOC", "3"))))
    except ValueError:
        return 3


def top_k_hard_cap() -> int:
    """与 UI slider 上限对齐的保守上界，供环境变量校验用。"""
    return 32


RouteId = Literal["kb", "general", "code"]


def _kb_feature_query_rewrite() -> bool:
    return os.environ.get("KB_QUERY_REWRITE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _kb_feature_self_rag() -> bool:
    return os.environ.get("KB_SELF_RAG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _kb_feature_routing() -> bool:
    """默认关闭：问答以资料库检索为主；需多链路时再设 KB_ROUTING=1。"""
    return os.environ.get("KB_ROUTING", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _kb_router_use_llm() -> bool:
    return os.environ.get("KB_ROUTER_USE_LLM", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _llm_configured() -> bool:
    base = _resolved_openai_base_url().strip()
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    key = _effective_api_key(base)
    return bool(base and model and key)


def _llm_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    """短提示词_completion（查询重写、路由、Self-RAG 质检），与 ask_llm 共用网关配置。"""
    if not _llm_configured():
        return ""
    base_url = _resolved_openai_base_url().rstrip("/")
    api_key = _effective_api_key(base_url)
    model = os.environ.get("OPENAI_MODEL", "").strip()
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    _merge_ollama_chat_payload(payload, model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client = _httpx_client()
    last_error = None
    for url in _chat_urls(base_url):
        try:
            resp = client.post(url, headers=headers, json=payload, timeout=120.0)
            if resp.status_code == 404:
                last_error = str(resp.status_code)
                continue
            resp.raise_for_status()
            data = json.loads(resp.content.decode("utf-8", errors="replace"))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            return str(content).strip()
        except Exception as exc:
            last_error = str(exc)
    print(f"辅助 LLM 调用失败: {last_error}")
    return ""


_VAGUE_END = re.compile(r"(吗|呢|嘛)\s*$")


def _is_query_likely_vague(q: str) -> bool:
    """短问、信息词少、泛问句等，适合触发检索问句重写。"""
    q = q.strip()
    if not q:
        return False
    if len(q) < 14:
        return True
    try:
        tags = jieba.analyse.extract_tags(q, topK=14, withWeight=False)
    except Exception:
        tags = []
    substantive = [t for t in tags if t not in _FTS_STOPWORDS and len(t) > 1]
    if len(substantive) < 2:
        return True
    if len(q) < 48 and _VAGUE_END.search(q) and len(substantive) < 4:
        return True
    return False


def _llm_rewrite_query_for_search(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是中文检索查询优化器。用户问句可能口语化或指代不清。"
                "请输出**一条**用于全文检索的短查询：保留专有名词与主题词，可补全省略成分；"
                "不要解释、不要引号、单行不超过 120 字。"
            ),
        },
        {"role": "user", "content": question[:1500]},
    ]
    return _llm_chat(messages, max_tokens=200, temperature=0.15)


def prepare_search_query(user_question: str) -> tuple[str, str | None]:
    """
    返回 (用于 FTS 的查询串, 若重写则给出重写后的字符串供 UI 展示，否则 None)。
    """
    q = user_question.strip()
    if not q:
        return q, None
    if not _kb_feature_query_rewrite() or not _llm_configured():
        return q, None
    if not _is_query_likely_vague(q):
        return q, None
    rw = _llm_rewrite_query_for_search(q)
    line = (rw.splitlines() or [""])[0].strip()
    if not line or line == q:
        return q, None
    return line, line


def _route_question_rules(q: str) -> RouteId:
    ql = q.lower()
    hints = (
        "traceback",
        "exception",
        "syntaxerror",
        "indentationerror",
        "def ",
        "class ",
        "import ",
        "报错",
        "调试",
        "npm ",
        "pip ",
        "git ",
        "sql",
        "httperror",
        "json.decode",
        "代码",
        "函数",
        "编程",
        "api接口",
        "```",
        "typescript",
        "javascript",
        "python",
    )
    if any(h in q or h in ql for h in hints):
        return "code"
    if len(q) < 12 and re.match(
        r"^[\s你好您好谢谢不用谢再见哈嘿呀哦嗯哼\d\w…\.。\!！\?？\,，~～]+$", q
    ):
        return "general"
    return "kb"


def _route_question_llm(q: str) -> RouteId | None:
    raw = _llm_chat(
        [
            {
                "role": "system",
                "content": (
                    "Classify for a document RAG app. Output exactly one token: "
                    "KB, GENERAL, or CODE.\n"
                    "KB = default: anything about the user's books/notes/documents, study questions, or facts they likely indexed.\n"
                    "GENERAL = only brief greetings/thanks (no substantive question), or chit-chat clearly unrelated to any document need.\n"
                    "CODE = programming, debugging, APIs, CLI."
                ),
            },
            {"role": "user", "content": q[:2000]},
        ],
        max_tokens=12,
        temperature=0.0,
    ).upper()
    if "CODE" in raw:
        return "code"
    if "GENERAL" in raw:
        return "general"
    if "KB" in raw:
        return "kb"
    return None


def route_question(question: str) -> RouteId:
    """根据问题类型选择处理链路：kb / general / code。"""
    if not _kb_feature_routing():
        return "kb"
    q = question.strip()
    if not q:
        return "kb"
    if _kb_router_use_llm() and _llm_configured():
        r = _route_question_llm(q)
        if r is not None:
            return r
    return _route_question_rules(q)


def skip_retrieval_for_route(route: RouteId, question: str) -> bool:
    """
    极短寒暄类 GENERAL 问题不检索，避免无关文档干扰 Self-RAG 与引用列表。
    """
    if route != "general":
        return False
    q = question.strip()
    if len(q) >= 28:
        return False
    return bool(
        re.match(
            r"^[\s你好您好谢谢谢啦不用谢再见哈嗨嘿哟嗯哼哦噢\d\w。！？\.·…~～,\，]{0,32}$",
            q,
        )
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}


def self_rag_verify(
    question: str, contexts: list[dict[str, Any]], answer: str
) -> dict[str, Any]:
    """
    Self-RAG：判断生成内容是否主要由当前检索片段支持。
    返回 dict: grounded (bool), reason (str), refined_search (str)。
    """
    out: dict[str, Any] = {"grounded": True, "reason": "", "refined_search": ""}
    if not _kb_feature_self_rag() or not _llm_configured():
        return out
    if not contexts:
        return out
    ans = (answer or "").strip()
    if len(ans) < 40:
        return out
    ctx_brief = "\n".join(
        f"[{i}] {(c.get('text') or '')[:380].replace(chr(10), ' ')}"
        for i, c in enumerate(contexts[:10], start=1)
    )
    raw = _llm_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是 RAG 质检员。根据用户问题、检索片段摘要、模型回答，判断回答的核心论断是否"
                    "主要由检索片段支持。只输出严格 JSON，键："
                    "grounded (true/false), reason (简中≤80字), "
                    "refined_search (若需二次检索则给更贴切的短检索句，否则空字符串)。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question[:900]}\n\n检索摘要：\n{ctx_brief}\n\n回答：{ans[:4000]}",
            },
        ],
        max_tokens=220,
        temperature=0.05,
    )
    data = _parse_json_object(raw)
    if not data:
        return out
    grounded = data.get("grounded")
    if isinstance(grounded, str):
        grounded = grounded.strip().lower() in ("true", "1", "yes")
    elif grounded is None:
        grounded = True
    out["grounded"] = bool(grounded)
    out["reason"] = str(data.get("reason") or "")[:120]
    out["refined_search"] = str(data.get("refined_search") or "").strip()[:200]
    return out


def retrieve_with_notes(
    conn: sqlite3.Connection,
    user_question: str,
    top_k: int,
    library_root: Path | str | None = None,
) -> tuple[list[dict[str, Any]], str, str | None, str | None]:
    """
    检索入口：可选查询重写、可选语义缓存。
    返回 (contexts, 实际检索 query, 说明或 None, 缓存答案或 None)；第四项非空时跳过检索与 LLM。
    """
    search_q, rewritten = prepare_search_query(user_question)
    note = f"已优化检索问句：{rewritten}" if rewritten else None
    try:
        from semantic_cache import lookup_cached_answer

        hit = lookup_cached_answer(conn, user_question.strip())
    except Exception:
        hit = None
    if hit:
        extra = "语义缓存命中（跳过检索与模型）"
        note_out = f"{note}；{extra}" if note else extra
        return [], search_q, note_out, hit
    ctx = retrieve(conn, search_q, top_k, library_root=library_root)
    return ctx, search_q, note, None


# 检索查询侧停用：减少「的、是、如何」等对 OR 全匹配的噪声（索引侧仍保留全文）
_FTS_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "有",
        "与",
        "及",
        "或",
        "等",
        "为",
        "以",
        "而",
        "之",
        "也",
        "都",
        "就",
        "要",
        "会",
        "能",
        "可",
        "这",
        "那",
        "其",
        "中",
        "对",
        "从",
        "到",
        "把",
        "被",
        "将",
        "让",
        "使",
        "由",
        "于",
        "吗",
        "呢",
        "吧",
        "啊",
        "么",
        "嘛",
        "很",
        "最",
        "更",
        "还",
        "又",
        "再",
        "什么",
        "怎么",
        "如何",
        "哪些",
        "是否",
        "可以",
        "应该",
        "如果",
        "因为",
        "所以",
        "但是",
        "然后",
        "一个",
        "一些",
        "这种",
        "这样",
        "进行",
        "通过",
        "根据",
        "关于",
        "以及",
        "或者",
        "还有",
        "不是",
        "没有",
        "需要",
        "请",
        "你",
        "我",
        "我们",
        "你们",
        "他们",
        "它们",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "although",
        "though",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "it",
        "its",
        "their",
        "theirs",
    }
)


def _filter_query_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        s = t.strip()
        if not s:
            continue
        if s in _FTS_STOPWORDS:
            continue
        if len(s) == 1 and not s.isalnum():
            continue
        out.append(s)
    return out


@lru_cache(maxsize=256)
def _query_terms_core(text: str) -> tuple[str, ...]:
    """
    用于 MATCH 与重排的查询词：优先 TF-IDF 关键词（比整句分词全 OR 更准），不足再回退分词。
    """
    s = text.strip()
    if not s:
        return tuple()
    try:
        tags = jieba.analyse.extract_tags(s, topK=14, withWeight=False)
    except Exception:
        tags = []
    cand = _filter_query_tokens(list(tags))
    if len(cand) < 2:
        cand = _filter_query_tokens([w.strip() for w in jieba.lcut(s, cut_all=False) if w.strip()])
    if not cand and s:
        return (s,)
    # 过长查询：限制 OR 项数量，优先靠前的关键词（TF-IDF 已排序）
    try:
        max_terms = max(4, min(18, int(os.environ.get("KB_FTS_MAX_TERMS", "14"))))
    except ValueError:
        max_terms = 14
    return tuple(cand[:max_terms])


@lru_cache(maxsize=256)
def fts_query_from_question(text: str) -> str:
    """
    拼 FTS5 的 MATCH 查询串：关键词加双引号、OR 连接。

    使用 extract_tags 抽关键词 + 停用词过滤，减少全句分词全 OR 带来的噪声命中。
    """
    terms = list(_query_terms_core(text))
    quoted = ['"' + t.replace('"', "") + '"' for t in terms]
    if not quoted:
        return text.replace('"', "")
    return " OR ".join(quoted)


@lru_cache(maxsize=256)
def _jieba_keywords(text: str) -> tuple[str, ...]:
    """重排与 LIKE 降级用的关键词（略宽于 FTS 专用词表）。"""
    core = list(_query_terms_core(text))
    if len(core) >= 2:
        return tuple(core)
    words = [
        w.strip()
        for w in jieba.lcut(text.strip(), cut_all=False)
        if w.strip() and len(w.strip()) >= 2 and w.strip() not in _FTS_STOPWORDS
    ]
    if words:
        return tuple(dict.fromkeys(words))
    s = text.strip()
    return (s,) if s else tuple()


def _keyword_coverage(text: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 0.0
    hit = 0
    for k in keywords:
        if len(k) <= 1:
            if k and k in text:
                hit += 1
        elif k in text:
            hit += 1
    return hit / float(len(keywords))


def _title_bonus(title: str, keywords: tuple[str, ...]) -> float:
    if not title or not keywords:
        return 0.0
    t = title.lower()
    hit = sum(1 for k in keywords if len(k) > 1 and k.lower() in t)
    return min(1.0, hit / max(2.0, float(len(keywords))))


def _rerank_score(
    bm25: float,
    bm25_min: float,
    bm25_max: float,
    coverage: float,
    title_b: float,
    cover_w: float,
    title_w: float,
) -> float:
    """bm25 越小越好；归一化后与其它项合成「越大越好」的分值。"""
    span = bm25_max - bm25_min
    if span <= 1e-12:
        norm_bad = 0.5
    else:
        norm_bad = (bm25 - bm25_min) / span
    return (1.0 - norm_bad) + cover_w * coverage + title_w * title_b


def _apply_per_doc_cap(rows: list[dict[str, Any]], top_k: int, max_per_doc: int) -> list[dict[str, Any]]:
    """按分值降序贪心选取，同一 doc 路径不超过 max_per_doc 段。"""
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for r in rows:
        if len(out) >= top_k:
            break
        p = str(r.get("path", ""))
        if counts[p] >= max_per_doc:
            continue
        out.append(r)
        counts[p] += 1
    return out


def _postprocess_ranked_chunks(sorted_by_rr: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """同前缀去重 → 每文档段数上限 → 不足则放宽补满。"""
    max_per_doc = _kb_max_chunks_per_doc()
    deduped: list[dict[str, Any]] = []
    seen_prefix: set[str] = set()
    for p in sorted_by_rr:
        raw = p.get("text") or ""
        prefix = raw[:160].strip()
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)
        p.pop("_rr", None)
        deduped.append(p)

    capped = _apply_per_doc_cap(deduped, top_k, max_per_doc)
    if len(capped) >= top_k:
        return capped[:top_k]

    seen_ids = {c["chunk_id"] for c in capped}
    path_counts: dict[str, int] = {}
    for x in capped:
        path_counts[str(x["path"])] = path_counts.get(str(x["path"]), 0) + 1
    for p in deduped:
        if len(capped) >= top_k:
            break
        if p["chunk_id"] in seen_ids:
            continue
        path = str(p["path"])
        if path_counts.get(path, 0) >= max_per_doc:
            continue
        capped.append(p)
        seen_ids.add(p["chunk_id"])
        path_counts[path] = path_counts.get(path, 0) + 1
    return capped[:top_k]


def _retrieve_fts_reranked(
    conn: sqlite3.Connection,
    match_query: str,
    keywords: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    oversample = _kb_retrieve_oversample()
    fetch_n = min(200, max(top_k * oversample, top_k))
    cover_w = _kb_rerank_cover_weight()
    title_w = _kb_rerank_title_weight()

    try:
        rows = conn.execute(
            """
            SELECT c.id, d.path, COALESCE(d.title, '') AS title, c.chunk_index, c.text_raw,
                   bm25(chunks_fts) AS score,
                   COALESCE(c.meta_json, '') AS meta_json,
                   COALESCE(c.context_text, '') AS context_text
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, fetch_n),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []

    parsed = [
        {
            "chunk_id": r[0],
            "path": r[1],
            "title": r[2],
            "chunk_index": r[3],
            "text": r[4],
            "score": r[5],
            "meta_json": r[6],
            "context_text": r[7],
        }
        for r in rows
    ]
    bm25s = [float(p["score"]) for p in parsed]
    bm25_min, bm25_max = min(bm25s), max(bm25s)

    for p in parsed:
        cov = _keyword_coverage(p["text"], keywords)
        tb = _title_bonus(p["title"], keywords)
        p["_rr"] = _rerank_score(
            float(p["score"]), bm25_min, bm25_max, cov, tb, cover_w, title_w
        )

    parsed.sort(key=lambda x: float(x.get("_rr", 0.0)), reverse=True)
    return _postprocess_ranked_chunks(parsed, top_k)


def _retrieve_fts_ranked_long(
    conn: sqlite3.Connection,
    match_query: str,
    keywords: tuple[str, ...],
    fetch_n: int,
) -> list[dict[str, Any]]:
    """BM25+特征重排的长候选列表（不做去重/每书封顶），供混合检索使用。"""
    try:
        rows = conn.execute(
            """
            SELECT c.id, d.path, COALESCE(d.title, '') AS title, c.chunk_index, c.text_raw,
                   bm25(chunks_fts) AS score,
                   COALESCE(c.meta_json, '') AS meta_json,
                   COALESCE(c.context_text, '') AS context_text
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, fetch_n),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []
    parsed = [
        {
            "chunk_id": r[0],
            "path": r[1],
            "title": r[2],
            "chunk_index": r[3],
            "text": r[4],
            "score": r[5],
            "meta_json": r[6],
            "context_text": r[7],
        }
        for r in rows
    ]
    bm25s = [float(p["score"]) for p in parsed]
    bm25_min, bm25_max = min(bm25s), max(bm25s)
    cover_w = _kb_rerank_cover_weight()
    title_w = _kb_rerank_title_weight()
    for p in parsed:
        cov = _keyword_coverage(p["text"], keywords)
        tb = _title_bonus(p["title"], keywords)
        p["_rr"] = _rerank_score(
            float(p["score"]), bm25_min, bm25_max, cov, tb, cover_w, title_w
        )
    parsed.sort(key=lambda x: float(x.get("_rr", 0.0)), reverse=True)
    for p in parsed:
        p.pop("_rr", None)
    return parsed


def _kb_hybrid_search_enabled() -> bool:
    return os.environ.get("KB_HYBRID_SEARCH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _kb_cross_rerank_enabled() -> bool:
    return os.environ.get("KB_CROSS_RERANK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _kb_rrf_k() -> int:
    try:
        return max(10, min(120, int(os.environ.get("KB_RRF_K", "60"))))
    except ValueError:
        return 60


def _kb_fts_pool_limit() -> int:
    try:
        return max(24, min(200, int(os.environ.get("KB_HYBRID_FTS_POOL", "100"))))
    except ValueError:
        return 100


def _kb_vector_pool_limit() -> int:
    try:
        return max(24, min(200, int(os.environ.get("KB_HYBRID_VECTOR_POOL", "100"))))
    except ValueError:
        return 100


def _kb_hybrid_downgrade_enabled() -> bool:
    return os.environ.get("KB_HYBRID_STRONG_MATCH_DEGRADE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _fts_keyword_strong_hit(
    fts_list: list[dict[str, Any]],
    query: str,
    keywords: tuple[str, ...],
) -> bool:
    """BM25 首条与问句高度重合时视为「强命中」，可减半向量候选池。"""
    if not fts_list:
        return False
    raw = str(fts_list[0].get("text") or "")
    qn = re.sub(r"\s+", "", (query or "").strip().lower())
    tn = re.sub(r"\s+", "", raw.lower())
    if len(qn) >= 10 and qn in tn:
        return True
    if keywords and len(keywords) >= 2 and _keyword_coverage(raw, keywords) >= 1.0:
        return True
    return False


def _enrich_chunk_rows(rows: list[dict[str, Any]]) -> None:
    """解析 meta_json，填充 prompt_text（优先 context_text 大块上下文）。"""
    for c in rows:
        mj = c.get("meta_json")
        if isinstance(mj, str) and mj.strip():
            try:
                c["meta"] = json.loads(mj)
            except json.JSONDecodeError:
                c["meta"] = {}
        else:
            c["meta"] = {}
        c.pop("meta_json", None)
        ct = c.get("context_text")
        if isinstance(ct, str) and ct.strip():
            c["prompt_text"] = ct.strip()
        else:
            c["prompt_text"] = str(c.get("text") or "")


def _kb_rerank_pool_limit() -> int:
    try:
        return max(8, min(48, int(os.environ.get("KB_RERANK_POOL", "20"))))
    except ValueError:
        return 20


def _embedding_model_name_retrieval() -> str:
    return (
        os.environ.get("KB_EMBEDDING_MODEL")
        or os.environ.get("KB_SEMANTIC_MODEL")
        or "paraphrase-multilingual-MiniLM-L12-v2"
    ).strip()[:200]


def _vector_table_nonempty(conn: sqlite3.Connection) -> bool:
    try:
        r = conn.execute("SELECT 1 FROM chunk_vectors LIMIT 1").fetchone()
        return r is not None
    except sqlite3.OperationalError:
        return False


_retrieval_bi_encoder = None


def _get_bi_encoder_retrieval():
    global _retrieval_bi_encoder
    if _retrieval_bi_encoder is not None:
        return _retrieval_bi_encoder
    from sentence_transformers import SentenceTransformer

    _retrieval_bi_encoder = SentenceTransformer(_embedding_model_name_retrieval())
    return _retrieval_bi_encoder


def _kb_vector_unit_eps() -> float:
    """索引与查询向量均为 sentence-transformers L2 归一化时，可跳过重复 normalize。"""
    try:
        return max(1e-6, min(0.1, float(os.environ.get("KB_VECTOR_UNIT_EPS", "0.001"))))
    except ValueError:
        return 1e-3


def _retrieve_vector_candidates(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    scoped_root: Path | None,
) -> list[dict[str, Any]]:
    import numpy as np

    model_name = _embedding_model_name_retrieval()
    try:
        rows = conn.execute(
            """
            SELECT v.chunk_id, v.dim, v.vec, v.model, c.chunk_index, c.text_raw, d.path, COALESCE(d.title, ''),
                   COALESCE(c.meta_json, ''), COALESCE(c.context_text, '')
            FROM chunk_vectors v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.doc_id
            WHERE v.model = ?
            """,
            (model_name,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        try:
            rows = conn.execute(
                """
                SELECT v.chunk_id, v.dim, v.vec, v.model, c.chunk_index, c.text_raw, d.path, COALESCE(d.title, ''),
                       COALESCE(c.meta_json, ''), COALESCE(c.context_text, '')
                FROM chunk_vectors v
                JOIN chunks c ON c.id = v.chunk_id
                JOIN documents d ON d.id = c.doc_id
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    if not rows:
        return []

    enc = _get_bi_encoder_retrieval()
    qv = enc.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    qv = np.asarray(qv, dtype=np.float32).reshape(-1)
    qd = int(qv.shape[0])
    eps = _kb_vector_unit_eps()
    qn = float(np.linalg.norm(qv))
    if abs(qn - 1.0) >= eps:
        qv = qv / max(qn, 1e-12)
    blobs: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for r in rows:
        cid, _dim, blob, _vmod, cidx, text, path, title = r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        meta_json = r[8] if len(r) > 8 else ""
        context_text = r[9] if len(r) > 9 else ""
        if scoped_root is not None and not _doc_path_in_library(path, scoped_root):
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.shape[0] != qd:
            continue
        blobs.append(vec)
        metas.append(
            {
                "chunk_id": int(cid),
                "path": path,
                "title": title or "",
                "chunk_index": int(cidx),
                "text": text,
                "meta_json": meta_json,
                "context_text": context_text,
            }
        )
    if not blobs:
        return []
    mat = np.stack(blobs, axis=0)
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if float(np.max(np.abs(row_norms - 1.0))) >= eps:
        mat = mat / np.maximum(row_norms, 1e-12)
    sims = mat @ qv
    lim = min(limit, sims.shape[0])
    if lim <= 0:
        return []
    if sims.shape[0] <= lim:
        order = np.argsort(-sims, kind="stable")
    else:
        part = np.argpartition(-sims, lim - 1)[:lim]
        order = part[np.argsort(-sims[part], kind="stable")]
    order = order[:lim]
    out: list[dict[str, Any]] = []
    for idx in order:
        i = int(idx)
        sim = float(sims[i])
        row_dict = dict(metas[i])
        row_dict["score"] = float(-sim)
        out.append(row_dict)
    return out


def _hybrid_rrf_merge(
    fts_rows: list[dict[str, Any]],
    vec_rows: list[dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    id_to_row: dict[int, dict[str, Any]] = {}
    order_fts: list[int] = []
    for r in fts_rows:
        cid = int(r["chunk_id"])
        id_to_row[cid] = dict(r)
        order_fts.append(cid)
    order_vec: list[int] = []
    for r in vec_rows:
        cid = int(r["chunk_id"])
        if cid not in id_to_row:
            id_to_row[cid] = dict(r)
        order_vec.append(cid)
    if not order_fts and not order_vec:
        return []
    scores: defaultdict[int, float] = defaultdict(float)
    for i, cid in enumerate(order_fts):
        scores[cid] += 1.0 / float(rrf_k + i + 1)
    for i, cid in enumerate(order_vec):
        scores[cid] += 1.0 / float(rrf_k + i + 1)
    ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return [id_to_row[c] for c in ranked if c in id_to_row]


_cross_encoder_model = None


def _cross_encoder_available() -> bool:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401

        return True
    except ImportError:
        return False


def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    from sentence_transformers import CrossEncoder

    name = (os.environ.get("KB_CROSS_ENCODER_MODEL") or "BAAI/bge-reranker-base").strip()
    _cross_encoder_model = CrossEncoder(name)
    return _cross_encoder_model


def _kb_cross_encoder_batch_size() -> int:
    try:
        return max(8, min(128, int(os.environ.get("KB_CROSS_ENCODER_BATCH", "32"))))
    except ValueError:
        return 32


def _cross_encoder_rerank(
    query: str, rows: list[dict[str, Any]], take: int
) -> list[dict[str, Any]]:
    if not rows or take <= 0:
        return []
    head = rows[:take]
    ce = _get_cross_encoder()
    texts = [(r.get("text") or "")[:2200] for r in head]
    pairs: list[list[str]] = [[query, t] for t in texts]
    try:
        scores = ce.predict(
            pairs,
            show_progress_bar=False,
            batch_size=_kb_cross_encoder_batch_size(),
        )
    except TypeError:
        scores = ce.predict(pairs, show_progress_bar=False)
    decorated = [(float(s), r) for s, r in zip(scores, head)]
    decorated.sort(key=lambda x: x[0], reverse=True)
    return [r for _s, r in decorated]


def _retrieve_hybrid_then_rerank(
    conn: sqlite3.Connection,
    query: str,
    fts_q: str,
    keywords: tuple[str, ...],
    kw_for_rank: tuple[str, ...],
    scoped_root: Path | None,
    top_k: int,
) -> list[dict[str, Any]]:
    fts_lim = _kb_fts_pool_limit()
    vec_lim = _kb_vector_pool_limit()
    fts_list = _retrieve_fts_ranked_long(
        conn, fts_q, kw_for_rank if kw_for_rank else keywords, fts_lim
    )
    if scoped_root is not None:
        fts_list = [x for x in fts_list if _doc_path_in_library(x["path"], scoped_root)]

    if _kb_hybrid_downgrade_enabled() and _fts_keyword_strong_hit(
        fts_list, query, keywords
    ):
        vec_lim = max(12, vec_lim // 2)

    vec_list: list[dict[str, Any]] = []
    try:
        vec_list = _retrieve_vector_candidates(conn, query, vec_lim, scoped_root)
    except Exception as exc:
        print(f"向量检索已跳过: {exc}")

    merged = _hybrid_rrf_merge(fts_list, vec_list, _kb_rrf_k())
    if not merged:
        return []

    rpool = _kb_rerank_pool_limit()
    head_n = min(len(merged), max(rpool, top_k * 2))
    head = merged[:head_n]

    if _kb_cross_rerank_enabled() and _cross_encoder_available() and len(head) > 1:
        try:
            ce_take = min(len(head), rpool)
            head = _cross_encoder_rerank(query, head, ce_take)
        except Exception as exc:
            print(f"Cross-Encoder 重排序失败，使用 RRF 序: {exc}")

    diversify = max(top_k * 4, top_k, 12)
    return _postprocess_ranked_chunks(head, diversify)


def _doc_path_in_library(doc_path: str, library_root: Path) -> bool:
    """判断索引中的 documents.path 是否落在用户指定的资料根目录下（兼容 Windows 大小写）。"""
    try:
        root = library_root.expanduser().resolve()
        doc = Path(doc_path).expanduser().resolve()
    except OSError:
        return False
    try:
        doc.relative_to(root)
        return True
    except ValueError:
        pass
    if os.name == "nt":
        dr = os.path.normcase(str(doc))
        rr = os.path.normcase(str(root)).rstrip("\\/")
        if not rr:
            return False
        return dr == rr or dr.startswith(rr + os.sep)
    return False


def scoped_miss_looks_like_library_path_issue(
    conn: sqlite3.Connection,
    search_query: str,
    library_root: Path | str | None,
) -> bool:
    """
    在「带资料根过滤的检索」无结果时，用少量 FTS 行判断：是否存在命中但样本文档均不在该根下，
    从而提示侧栏路径与建库时不一致。避免再跑一整轮无 scope 的混合检索。
    """
    if library_root is None or not (search_query or "").strip():
        return False
    try:
        root = Path(library_root).expanduser().resolve()
        if not root.is_dir():
            return False
    except OSError:
        return False
    fq = fts_query_from_question(search_query.strip())
    if not fq:
        return False
    try:
        rows = conn.execute(
            """
            SELECT d.path FROM chunks_fts cf
            JOIN chunks c ON c.id = cf.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE cf MATCH ?
            LIMIT 16
            """,
            (fq,),
        ).fetchall()
    except sqlite3.OperationalError:
        return False
    if not rows:
        return False
    any_inside = any(_doc_path_in_library(str(r[0]), root) for r in rows)
    return not any_inside


def _retrieve_like_fallback(
    conn: sqlite3.Connection,
    keywords: tuple[str, ...],
    top_k: int,
    sql_limit: int = 120,
) -> list[dict[str, Any]]:
    """FTS 无命中时的降级：LIKE 粗筛 + 与主路径一致的重排与去冗余。"""
    if not keywords:
        return []
    like_expr = " OR ".join(["c.text_raw LIKE ?"] * len(keywords))
    lim = max(1, min(2000, int(sql_limit)))
    rows2 = conn.execute(
        f"""
        SELECT c.id, d.path, COALESCE(d.title, '') AS title, c.chunk_index, c.text_raw,
               COALESCE(c.meta_json, ''), COALESCE(c.context_text, '')
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE {like_expr}
        LIMIT ?
        """,
        tuple([f"%{k}%" for k in keywords]) + (lim,),
    ).fetchall()
    if not rows2:
        return []
    cover_w = _kb_rerank_cover_weight()
    title_w = _kb_rerank_title_weight()
    parsed: list[dict[str, Any]] = []
    for r in rows2:
        hit = sum(r[4].count(k) for k in keywords)
        parsed.append(
            {
                "chunk_id": r[0],
                "path": r[1],
                "title": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "score": float(-hit),
                "meta_json": r[5] if len(r) > 5 else "",
                "context_text": r[6] if len(r) > 6 else "",
            }
        )
    bm25s = [p["score"] for p in parsed]
    bm25_min, bm25_max = min(bm25s), max(bm25s)
    for p in parsed:
        cov = _keyword_coverage(p["text"], keywords)
        tb = _title_bonus(p["title"], keywords)
        p["_rr"] = _rerank_score(float(p["score"]), bm25_min, bm25_max, cov, tb, cover_w, title_w)
    parsed.sort(key=lambda x: float(x["_rr"]), reverse=True)
    return _postprocess_ranked_chunks(parsed, top_k)


def _kb_graph_rag_enabled() -> bool:
    return os.environ.get("KB_GRAPH_RAG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _kb_graph_slots() -> int:
    try:
        return max(0, min(16, int(os.environ.get("KB_GRAPH_SLOTS", "2"))))
    except ValueError:
        return 2


def _kb_graph_expand_per_seed() -> int:
    try:
        return max(1, min(8, int(os.environ.get("KB_GRAPH_EXPAND_PER_SEED", "2"))))
    except ValueError:
        return 2


def _chunk_graph_table_ready(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_graph_edges'"
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False


def _graph_expand_retrieval(
    conn: sqlite3.Connection,
    primary: list[dict[str, Any]],
    scoped_root: Path | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    LightRAG 风格轻量扩展：在保持总条数为 top_k 的前提下，用若干槽位替换为图上的相邻/关键词邻居块，
    缓解「检索到的块彼此孤立」问题。需已建 chunk_graph_edges 且 KB_GRAPH_RAG=1。
    """
    if not primary or not _kb_graph_rag_enabled() or not _chunk_graph_table_ready(conn):
        return primary
    slots = _kb_graph_slots()
    if slots <= 0 or top_k <= 0:
        return primary
    per = _kb_graph_expand_per_seed()
    seen: set[int] = set()
    for c in primary:
        cid = c.get("chunk_id")
        if cid is not None:
            seen.add(int(cid))
    new_ids: list[int] = []
    cap = max(slots * 4, slots * per, slots + 4)
    for c in primary:
        if len(new_ids) >= cap:
            break
        cid = c.get("chunk_id")
        if cid is None:
            continue
        try:
            rows = conn.execute(
                """
                SELECT dst_chunk_id FROM chunk_graph_edges
                WHERE src_chunk_id = ?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (int(cid), per),
            ).fetchall()
        except sqlite3.OperationalError:
            return primary
        for r in rows:
            did = int(r[0])
            if did in seen:
                continue
            seen.add(did)
            new_ids.append(did)
            if len(new_ids) >= cap:
                break
    if not new_ids:
        return primary
    placeholders = ",".join("?" * len(new_ids))
    try:
        qrows = conn.execute(
            f"""
            SELECT c.id, d.path, COALESCE(d.title, '') AS title, c.chunk_index, c.text_raw,
                   COALESCE(c.meta_json, ''), COALESCE(c.context_text, '')
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders})
            """,
            new_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return primary
    order = {nid: i for i, nid in enumerate(new_ids)}
    extras: list[dict[str, Any]] = []
    for r in qrows:
        path = str(r[1])
        if scoped_root is not None and not _doc_path_in_library(path, scoped_root):
            continue
        extras.append(
            {
                "chunk_id": int(r[0]),
                "path": path,
                "title": str(r[2] or ""),
                "chunk_index": int(r[3]),
                "text": str(r[4] or ""),
                "score": -0.5,
                "meta_json": str(r[5] or ""),
                "context_text": str(r[6] or ""),
            }
        )
    extras.sort(key=lambda x: order.get(int(x["chunk_id"]), 10**9))
    g = min(slots, len(extras))
    if g <= 0:
        return primary
    p_keep = max(0, top_k - g)
    merged = primary[:p_keep] + extras[:g]
    out = contexts_with_displayable_citations(merged, top_k)
    _enrich_chunk_rows(out)
    return out


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 4,
    library_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    默认混合检索：BM25（FTS）与句向量候选经 RRF 融合；对前 KB_RERANK_POOL 条可选 Cross-Encoder 重排，
    再每书封顶、过滤不可展示片段。未建 chunk_vectors、关闭 KB_HYBRID_SEARCH 或依赖缺失时回退仅 FTS。

    FTS 无命中时降级为 LIKE 粗筛。library_root 若有效则只保留其下文档。

    若 KB_GRAPH_RAG=1 且存在 chunk_graph_edges（全量建索引时生成），在保持条数为 top_k 的前提下用若干槽位并入图邻居块。
    """
    q = (query or "").strip()
    if not q:
        return []

    _ensure_chunk_meta_once(conn)

    scoped_root: Path | None = None
    if library_root is not None:
        try:
            cand = Path(library_root).expanduser().resolve()
            if cand.is_dir():
                scoped_root = cand
        except OSError:
            scoped_root = None

    rank_cap = min(200, max(top_k * 8, top_k))

    fts_q = fts_query_from_question(q)
    keywords = tuple(_jieba_keywords(q))
    kw_for_rank = keywords if keywords else _query_terms_core(q)

    res: list[dict[str, Any]] = []
    if _kb_hybrid_search_enabled() and _vector_table_nonempty(conn):
        try:
            res = _retrieve_hybrid_then_rerank(
                conn,
                q,
                fts_q,
                keywords,
                kw_for_rank if kw_for_rank else (q,),
                scoped_root,
                top_k,
            )
        except Exception as exc:
            print(f"混合检索不可用，回退 FTS：{exc}")
            res = []

    if not res:
        res = _retrieve_fts_reranked(
            conn, fts_q, kw_for_rank if kw_for_rank else (q,), rank_cap
        )
    if scoped_root is not None:
        res = [x for x in res if _doc_path_in_library(x["path"], scoped_root)]
    res = contexts_with_displayable_citations(res, top_k)
    _enrich_chunk_rows(res)
    if res:
        return _graph_expand_retrieval(conn, res, scoped_root, top_k)

    like_lim = 120 if scoped_root is None else min(800, max(120, top_k * 40))
    fb = _retrieve_like_fallback(
        conn, keywords if keywords else (q,), rank_cap, sql_limit=like_lim
    )
    if scoped_root is not None:
        fb = [x for x in fb if _doc_path_in_library(x["path"], scoped_root)]
    fb = contexts_with_displayable_citations(fb, top_k)
    _enrich_chunk_rows(fb)
    return _graph_expand_retrieval(conn, fb, scoped_root, top_k)


_REF_HEADING = re.compile(
    r"^#{1,4}\s*(?:引用|参考文献|references)\s*$",
    re.IGNORECASE,
)
_REF_EXACT = re.compile(r"^(?:引用|参考文献|references)\s*$", re.IGNORECASE)


def strip_model_reference_section(text: str) -> str:
    """删掉模型自己在文末生成的「引用/参考文献」标题及之后内容（页面会单独做统一引用列表，避免重复）。"""
    if not text:
        return text
    lines = text.splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        t = line.strip()
        if _REF_HEADING.match(t) or _REF_EXACT.match(t):
            cut = i
            break
    return "\n".join(lines[:cut]).rstrip()


# GFM/CommonMark 将 [1]、[1][2] 视作引用式链接；转义后以字面显示角标。
_MD_NUMERIC_CITATION = re.compile(r"\[\d+\](?!\()")


def escape_numeric_citations_for_markdown(text: str) -> str:
    """避免 Streamlit Markdown 吞掉论文式数字角标。"""
    if not text:
        return text
    return _MD_NUMERIC_CITATION.sub(
        lambda m: "\\[" + m.group(0)[1:-1] + "\\]", text
    )


def format_reference_list(contexts: list[dict[str, Any]]) -> str:
    """命令行模式下在答案后面打印的参考文献块（网页端用自己的展示逻辑）。"""
    """Single bibliography block: one line per source [n] filename."""
    parts = ["引用", ""]
    for i, c in enumerate(contexts, start=1):
        parts.append(f"[{i}] {Path(c['path']).name}")
    return "\n".join(parts)


def _citation_excerpt_normalize(raw: str) -> str:
    if not raw:
        return ""
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = t.strip()
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    return t.rstrip()


def _is_reference_bibliography_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 4:
        return False
    if re.match(r"^(参考文献|引用书目|参考资料|引用文献|书目|主要参考文献)", s):
        return True
    year_tail = bool(re.search(r"(?:[，,．.]|\s)\d{4}\s*$", s))
    has_pub = bool(
        re.search(
            r"(?:出版社|印书馆|出版公司|出版事业(?:股份)?|書店|印刷|三聯|工坊|人民社|编译社)",
            s,
        )
    )
    if has_pub and year_tail:
        return True
    if re.search(r"(?:台北|臺北|北京|上海|南京|香港|广州|武汉|杭州|天津|江苏):", s) and year_tail:
        return True
    if re.search(r"(?:台北|臺北|北京|上海|南京|香港|广州|武汉|杭州|天津|江苏)[：:]", s) and year_tail:
        return True
    if re.search(r"[译編著主][\.．]", s) and year_tail and (s.count(".") + s.count("．")) >= 2:
        return True
    if len(s) < 220 and s.count(".") >= 2 and year_tail and re.search(r"[：:]", s):
        return True
    if re.match(r"^[^，,]{1,30}[,，]\s*\d{4}\s*$", s) and ("出版" in s or "书店" in s or "印书馆" in s):
        return True
    # 截断的著录/题名片段：「作者.题名起首」但无句读、常以与/及/和煞尾（检索切块边界处常见）
    if len(s) <= 52 and not re.search(r"[。！？]", s) and re.match(
        r"^[\u4e00-\u9fff·•]{2,12}[\.．]\S+$", s
    ):
        if re.search(r"(?:与|及|和|——)\s*$", s):
            return True
    return False


def _is_toc_or_copyright_heading_line(s: str) -> bool:
    t = s.strip()
    if t in ("总目录", "版权页", "插图", "插图目录", "插页", "勘误表"):
        return True
    if re.match(r"^目\s*录$", t) and len(t) <= 4:
        return True
    return False


def _is_colon_volume_listing_line(s: str) -> bool:
    """如「毛泽东文集：第一卷」类目录行，非正文论证句。"""
    t = s.strip()
    return bool(
        re.match(
            r"^[^：:\n]{2,55}[：:]\s*第[一二三四五六七八九十百零\d〇○]+卷\s*$",
            t,
        )
    )


def _is_bracket_index_only_line(s: str) -> bool:
    """出版说明里单独一行的脚注/题注编号，如 [5]。"""
    return bool(re.match(r"^\[\d+\]\s*$", s.strip()))


def _is_bracket_with_zheshi_line(s: str) -> bool:
    return bool(re.match(r"^\[\d+\]\s*这是", s.strip()))


def _is_editorial_meta_caption_line(s: str) -> bool:
    """
    文集、选集常见的「这是……（的通报|的批语|起草的……）」脚注式题注，单句说明出处，非完整论述段落。
    """
    t = s.strip()
    if not t.startswith("这是"):
        return False
    if len(t) < 12 or len(t) > 300:
        return False
    if not t.endswith(("。", ".", "．")):
        return False
    markers = (
        "起草",
        "转发",
        "复电",
        "批语",
        "电报",
        "决定",
        "审阅",
        "加写",
        "为中共中央",
        "为中央",
        "通报",
        "指示",
        "报告稿",
        "同意",
    )
    return any(m in t for m in markers)


def _chunk_looks_like_glossary_or_index(text: str) -> bool:
    """
    整段为翻译词汇表、人名对照、术语英汉字典等辅助内容时，不适合作为论文式引用，整块丢弃。
    依据：版块标题关键词 + 英/拉术语行与中文释条成对出现密度 + 拉丁字母行占比与正文句读行占比失衡。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n < 7:
        return False

    head = "\n".join(lines[:min(36, n)])
    header_markers = (
        "翻译词汇表",
        "译者注释",
        "人名专名",
        "用于翻译插件",
        "更多译法详情请见",
        "专有名词解读",
        "更多专有名词",
    )
    has_header = any(m in head for m in header_markers) or any(
        ln in ("词汇表", "译者注释", "人名专名") for ln in lines[:22]
    )

    en_cn_pairs = 0
    for a, b in zip(lines, lines[1:]):
        if len(a) > 95 or len(b) > 110:
            continue
        ascii_letters = sum(1 for c in a if "A" <= c <= "Z" or "a" <= c <= "z")
        if ascii_letters >= 3 and not re.search(r"[\u4e00-\u9fff]{3,}", a):
            if re.search(r"[\u4e00-\u9fff]{2,}", b) or "《" in b:
                en_cn_pairs += 1

    latin_only = 0
    for ln in lines:
        if len(ln) > 130:
            continue
        if re.search(r"[\u4e00-\u9fff]", ln):
            continue
        if re.search(r"[A-Za-zÀ-ÿĀ-ž]{4,}", ln):
            latin_only += 1

    sentence_lines = sum(1 for ln in lines if re.search(r"[。！？]", ln))

    if en_cn_pairs >= 6:
        return True
    if n >= 7 and en_cn_pairs >= 5:
        return True
    if n < 10:
        if has_header and (en_cn_pairs >= 3 or latin_only >= 4):
            return True
        return False
    if has_header and (en_cn_pairs >= 3 or latin_only >= 4):
        return True
    if latin_only >= max(6, n // 4) and sentence_lines <= max(4, n // 6):
        return True
    return False


def _chunk_looks_like_toc_or_copyright_block(text: str) -> bool:
    """总目录、分卷书目、版权页等；多为「书名：第 N 卷」罗列，非论述正文。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blob_head = text[:1600]
    vol = sum(1 for ln in lines if _is_colon_volume_listing_line(ln))
    if vol >= 5:
        return True
    if "总目录" in blob_head or "版权页" in blob_head:
        if vol >= 3:
            return True
        if vol >= 2 and len(lines) <= 24:
            return True
    return False


def _chunk_looks_like_bracket_annotation_index(text: str) -> bool:
    """出版说明中连续 [5]、[6]… 加「这是……」单句题注，非正文段落。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    br = sum(1 for ln in lines if _is_bracket_index_only_line(ln))
    if br < 2:
        return False

    def _zheshi_caption(ln: str) -> bool:
        if _is_editorial_meta_caption_line(ln):
            return True
        t = ln.strip()
        return (
            t.startswith("这是")
            and 12 <= len(t) <= 300
            and t.endswith(("。", "．", "."))
            and not re.search(r"[「『（“]", t[:8])
        )

    zhe = sum(1 for ln in lines if _zheshi_caption(ln))
    if zhe < max(1, br - 1):
        return False
    noise = br + zhe
    return noise >= len(lines) - max(2, len(lines) // 6)


def _line_is_citation_noise(ln: str) -> bool:
    return (
        _is_reference_bibliography_line(ln)
        or _is_toc_or_copyright_heading_line(ln)
        or _is_colon_volume_listing_line(ln)
        or _is_bracket_index_only_line(ln)
        or _is_bracket_with_zheshi_line(ln)
        or _is_editorial_meta_caption_line(ln)
    )


def _whole_text_incomplete_prose_snippet(text: str) -> bool:
    """
    整段剔除书目行后仍像「半句话/半条著录」：无合格句末标点、过短且含明显切块残留。
    """
    s = text.strip()
    if not s:
        return True
    if "\n" in s:
        return False
    if len(s) > 140:
        return False
    if re.search(r"[。！？][」』\"]*\s*$", s) or re.search(r"[.!?][」』\"]*\s*$", s):
        return False
    if not re.search(r"[\u4e00-\u9fff]", s):
        return False
    if re.search(r"[\u4e00-\u9fff]{2,10}[\.．]\S+", s) and not re.search(r"[。！？]", s):
        return True
    if len(s) < 96 and not re.search(r"[。！？]", s) and re.search(r"(?:与|及|和|——)\s*$", s):
        return True
    return False


def _trim_leading_incomplete_sentence(s: str) -> str:
    s = s.strip()
    if len(s) < 12:
        return s
    if re.match(r"^[「『（【“\"]", s):
        return s
    probe = s[: min(90, len(s))]
    first = -1
    for p in ("。", "！", "？"):
        i = probe.find(p)
        if i != -1:
            first = i if first == -1 else min(first, i)
    # 仅剥掉极短的前缀「半句」；勿把首个完整短句（常见于学术定义）整段去掉
    if first != -1 and 0 < first <= 10:
        return s[first + 1 :].lstrip()
    return s


def _trim_trailing_incomplete_sentence(s: str) -> str:
    s = s.rstrip()
    if not s:
        return ""
    if re.search(r"[。！？][」』\"]*\s*$", s) or re.search(r"[.!?][」』\"]*\s*$", s):
        return s
    cut = -1
    for p in ("。", "！", "？"):
        k = s.rfind(p)
        if k > cut:
            cut = k
    if cut >= max(24, len(s) // 5):
        return s[: cut + 1].rstrip()
    return s


def citation_snippet_for_ui(raw: str) -> str:
    """
    引用区 / 模型上下文共用：去掉参考文献式书目行，并尽量让您的展示从完整句开始、在完整句结束。
    剔除后若无正文则返回空串。
    """
    base = _citation_excerpt_normalize(raw)
    if not base:
        return ""
    if _chunk_looks_like_glossary_or_index(base):
        return ""
    if _chunk_looks_like_toc_or_copyright_block(base):
        return ""
    if _chunk_looks_like_bracket_annotation_index(base):
        return ""
    lines = base.split("\n")
    kept = [ln for ln in lines if not _line_is_citation_noise(ln)]
    text = "\n".join(kept).strip()
    if not text:
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras:
        text = "\n\n".join(paras)
    text = _trim_leading_incomplete_sentence(text)
    text = _trim_trailing_incomplete_sentence(text)
    text = text.strip()
    if _whole_text_incomplete_prose_snippet(text):
        return ""
    return text


def contexts_with_displayable_citations(
    candidates: list[dict[str, Any]], target_k: int
) -> list[dict[str, Any]]:
    """
    按原顺序从候选里挑出「清洗后仍有正文」的片段，最多 target_k 条。
    用于检索阶段跳过参考文献块、词汇表等不可展示块，避免 UI / 提示词出现空引用。
    """
    if target_k <= 0:
        return []
    out: list[dict[str, Any]] = []
    for c in candidates:
        if citation_snippet_for_ui(str(c.get("text") or "")):
            out.append(c)
            if len(out) >= target_k:
                break
    return out


def _build_messages(
    question: str, contexts: list[dict[str, Any]], route: RouteId = "kb"
) -> list[dict[str, str]]:
    """
    拼 OpenAI 风格的 messages: [ {role: system, ...}, {role: user, ...} ]。

    route：kb 仅资料；general 可先常识再说明资料是否命中；code 强调代码与命令输出保留格式。
    contexts 可为空（如寒暄 GENERAL）：走无资料提示模式。
    """
    n = _kb_context_chars()
    pairs: list[tuple[dict[str, Any], str]] = []
    max_take = len(contexts)
    for c in contexts:
        raw = str(c.get("prompt_text") or c.get("text") or "")
        ex = citation_snippet_for_ui(raw) or ""
        if ex:
            pairs.append((c, ex[:n]))
            if len(pairs) >= max_take:
                break

    route = route if route in ("kb", "general", "code") else "kb"

    if not pairs:
        if route == "general":
            sys_gen = (
                "你是通用助手。当前未挂载私有书库片段，请根据常识与通识直接回答用户。"
                "回答宜简短友好；若问题需要专书或用户私人文档才能准确作答，请说明无法访问其资料。"
            )
            return [
                {"role": "system", "content": sys_gen},
                {"role": "user", "content": question},
            ]
        sys_kb = (
            "你是资料问答助手。当前没有可用的检索片段。\n"
            "请明确说明「资料库中未检索到相关内容」，并给出用户可如何换关键词重试的建议；不要编造书中细节。"
        )
        return [{"role": "system", "content": sys_kb}, {"role": "user", "content": question}]

    refs = "\n\n".join(
        [
            f"[{i}] 来源资料名称: {Path(c['path']).name} | 片段序号: {c['chunk_index']}\n{ex}"
            for i, (c, ex) in enumerate(pairs, start=1)
        ]
    )

    base = (
        "你是严谨的资料问答助手。\n"
        "行文须采用论文式标注：凡关键论断、定义、机制、数据、阶段划分、术语界定等来自某片段时，"
        "在该句或该分句末尾用方括号角标标出文献序号，与资料块编号一致，例如：……构成三阶段[1]；……索引与检索[2][3]。"
        "一句可多标；综合多处写作[1][2]。未用上的序号不要编造。\n"
        "结构：先「总览/结论」1–3句并带角标；再 ### 小节展开（背景、要点、机制或流程、边界与注意、小结等），小节内同样需要角标。\n"
        "重要：正文结束后不要另起「引用」「参考文献」或任何文献表；不要再列出 [1] 书名——系统会在页面最底部统一生成文献表。"
    )

    if route == "code":
        system = (
            base
            + "\n【链路：技术/代码】优先依据资料片段中的定义、接口与示例作答；"
            "若片段含代码、命令行或配置，保留缩进与标点，可用 Markdown 代码块呈现。"
            "事实必须以片段为准，不得编造不存在的 API。若资料未提及则写明「资料未提及」。"
        )
    elif route == "general":
        system = (
            base
            + "\n【链路：常识/通用】优先使用资料片段；若片段与问题明显不相关或不足以回答，"
            "可简短补充通识，并明确区分「资料中未涉及」与「常识补充」两部分。不得把常识冒充为书中原文。"
        )
    else:
        system = (
            base
            + "\n只能依据用户提供的资料片段作答关键事实，不得编造。若资料未提及则写明「资料未提及」。"
        )

    tune_extra = _rag_tuned_extra_system()
    if tune_extra:
        system += tune_extra
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    if _is_deepseek_r1_model(model):
        system += _deepseek_r1_doc_qa_system_suffix()

    tune_user = ""
    if tune_extra:
        tune_user = (
            "【硬性要求】正文里凡是采自下方片段的句子，必须在句末写上对应编号角标，格式为半角 [1] 或 [1][2]，与片段标题行编号一致；"
            "不要省略角标，不要用（1）、【1】替代。\n\n"
        )
    user = f"问题：{question}\n\n{tune_user}可用资料片段：\n{refs}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _chat_urls(base_url: str) -> list[str]:
    """不同厂商的 base_url 末尾是否为 /api/v3 路径略有差异；404 时自动换一个候选 URL 再试。"""
    candidate_urls = [f"{base_url}/chat/completions"]
    if base_url.endswith("/api/v3"):
        candidate_urls.append(f"{base_url}/openai/chat/completions")
    return candidate_urls


def ask_llm_stream(
    question: str, contexts: list[dict[str, Any]], route: RouteId = "kb"
) -> Iterator[str]:
    """
    流式调用：边生成边 yield 小段文本（网页上打字机效果）。

    解析 SSE：跳过心跳/注释行，找 `data: {...}` JSON，从 choices[].delta.content 取增量。
    """
    base_url = _resolved_openai_base_url().rstrip("/")
    api_key = _effective_api_key(base_url)
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not base_url or not api_key or not model:
        return
    stream_timeout = (
        600.0 if _is_local_openai_compatible_base(base_url) else 180.0
    )
    payload = {
        "model": model,
        "temperature": _kb_temperature(),
        "max_tokens": _kb_max_tokens(),
        "stream": True,
        "messages": _build_messages(question, contexts, route),
    }
    _merge_ollama_chat_payload(payload, model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client = _httpx_client()
    last_error = None
    for url in _chat_urls(base_url):
        try:
            with client.stream(
                "POST", url, headers=headers, json=payload, timeout=stream_timeout
            ) as resp:
                if resp.status_code == 404:
                    last_error = f"{resp.status_code}"
                    continue
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = _decode_sse_line(raw).strip("\r")
                    if line.startswith(":"):
                        continue  # SSE 注释/心跳，忽略
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break  # OpenAI 流式结束标记
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for ch in obj.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            yield piece
            return
        except Exception as exc:
            last_error = str(exc)
    print(f"模型流式调用失败，将尝试非流式。错误: {last_error}")


def ask_llm(
    question: str, contexts: list[dict[str, Any]], route: RouteId = "kb"
) -> str:
    """一次性等非流式返回完整答案；app.py 在流式失败时会回退到这里。"""
    base_url = _resolved_openai_base_url().rstrip("/")
    api_key = _effective_api_key(base_url)
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not base_url or not api_key or not model:
        return ""
    req_timeout = (
        600.0 if _is_local_openai_compatible_base(base_url) else 180.0
    )
    payload = {
        "model": model,
        "temperature": _kb_temperature(),
        "max_tokens": _kb_max_tokens(),
        "messages": _build_messages(question, contexts, route),
    }
    _merge_ollama_chat_payload(payload, model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client = _httpx_client()
    last_error = None
    for url in _chat_urls(base_url):
        try:
            resp = client.post(url, headers=headers, json=payload, timeout=req_timeout)
            if resp.status_code == 404:
                last_error = f"{resp.status_code} {resp.text}"
                continue
            resp.raise_for_status()
            data = json.loads(resp.content.decode("utf-8", errors="replace"))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            return strip_model_reference_section(str(content).strip())
        except Exception as exc:
            last_error = str(exc)
    print(f"模型调用失败，已回退到检索摘要模式。错误: {last_error}")
    return ""


def fallback_answer(question: str, contexts: list[dict[str, Any]]) -> str:
    """不调模型时的「保底答案」：只展示检索到的片段摘要 + 引用文件名。"""
    lines = [f"问题：{question}", "", "已检索到以下高相关资料片段："]
    for i, c in enumerate(contexts, start=1):
        snippet = c["text"].replace("\n", " ")
        lines.append(f"[{i}] {snippet[:220] + ('...' if len(snippet) > 220 else '')}")
    lines.append("")
    lines.append("引用：")
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] {Path(c['path']).name}（chunk={c['chunk_index']}）")
    return "\n".join(lines)


def main() -> None:
    """脚本直接运行时的入口：打开库 → 检索 → 调 LLM 或 fallback → 打印。"""
    parser = argparse.ArgumentParser(description="Ask questions from local KB with citations.")
    parser.add_argument("--db", default="knowledge_base.sqlite", help="SQLite DB path.")
    parser.add_argument("--q", required=True, help="Question text.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks.")
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    rq = route_question(args.q)
    skip = skip_retrieval_for_route(rq, args.q)
    if skip:
        contexts: list[dict[str, Any]] = []
    else:
        contexts, _, _, _ = retrieve_with_notes(conn, args.q, args.top_k)
    conn.close()
    if not contexts and rq != "general":
        print("没有检索到相关内容。你可以尝试换个问法，或先运行 build_kb.py。")
        return
    answer = ask_llm(args.q, contexts, rq)
    if answer:
        print(answer)
        print()
        print(format_reference_list(contexts))
    else:
        print(fallback_answer(args.q, contexts))


if __name__ == "__main__":
    main()
