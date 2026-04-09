"""
可选 FastAPI 网关：SSE 流式输出（与 Streamlit 并存）。

  set KB_SQLITE_PATH=knowledge_base.sqlite
  pip install fastapi uvicorn
  uvicorn api_server:app --host 127.0.0.1 --port 8765

POST /api/chat/stream  JSON: {"question":"...","top_k":4}
响应：text/event-stream，每行 data: {"delta":"..."} ，结束 data: [DONE]
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ask_kb import ask_llm_stream, prepare_search_query, retrieve
from build_kb import effective_library_root

app = FastAPI(title="local-rag-knowledge-base", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat/stream")
def chat_stream(body: dict) -> StreamingResponse:
    q = (body.get("question") or "").strip()
    top_k = int(body.get("top_k") or 4)
    db_path = Path(
        body.get("db")
        or os.environ.get("KB_SQLITE_PATH")
        or "knowledge_base.sqlite"
    ).expanduser().resolve()
    if not q:
        def err():
            yield 'data: {"error":"empty question"}\n\n'

        return StreamingResponse(err(), media_type="text/event-stream")

    conn = sqlite3.connect(str(db_path))
    conn.text_factory = str
    try:
        sq, _ = prepare_search_query(q)
        lib = effective_library_root()
        root = lib if lib.is_dir() else None
        ctx = retrieve(conn, sq, top_k, library_root=root)
    finally:
        conn.close()

    def gen():
        for piece in ask_llm_stream(q, ctx, "kb"):
            yield "data: " + json.dumps({"delta": piece}, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
