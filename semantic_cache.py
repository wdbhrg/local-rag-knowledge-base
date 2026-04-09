"""
语义缓存：SQLite 存「问题向量 + 答案」，新问题与缓存问句余弦相似度 ≥ 阈值则跳过检索与 LLM。
支持按 query 规范化后 SHA-256 去重（同模型下 UPSERT）；精确哈希命中时无需加载嵌入模型。
依赖 sentence-transformers + numpy（与 chunk 向量同一套可选依赖）。
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time

_CACHE_MODEL = None


def _vector_unit_eps() -> float:
    try:
        return max(1e-6, min(0.1, float(os.environ.get("KB_VECTOR_UNIT_EPS", "0.001"))))
    except ValueError:
        return 1e-3


def _kb_semantic_cache_enabled() -> bool:
    return os.environ.get("KB_SEMANTIC_CACHE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _kb_semantic_cache_write_enabled() -> bool:
    return os.environ.get("KB_SEMANTIC_CACHE_WRITE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _similarity_threshold() -> float:
    try:
        return max(0.85, min(0.999, float(os.environ.get("KB_SEMANTIC_CACHE_THRESHOLD", "0.95"))))
    except ValueError:
        return 0.95


def _max_entries() -> int:
    try:
        return max(32, min(5000, int(os.environ.get("KB_SEMANTIC_CACHE_MAX", "400"))))
    except ValueError:
        return 400


def _embedding_model() -> str:
    return (
        os.environ.get("KB_SEMANTIC_CACHE_MODEL")
        or os.environ.get("KB_EMBEDDING_MODEL")
        or os.environ.get("KB_SEMANTIC_MODEL")
        or "paraphrase-multilingual-MiniLM-L12-v2"
    ).strip()


def _normalize_query_key(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)


def _hash_query(s: str) -> str:
    """用于去重与精确命中的稳定键（同义空白/大小写视为同一键）。"""
    return hashlib.sha256(_normalize_query_key(s).encode("utf-8")).hexdigest()


def _get_model():
    global _CACHE_MODEL
    if _CACHE_MODEL is not None:
        return _CACHE_MODEL
    from sentence_transformers import SentenceTransformer

    _CACHE_MODEL = SentenceTransformer(_embedding_model())
    return _CACHE_MODEL


def ensure_semantic_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            answer_text TEXT NOT NULL,
            created_ts REAL NOT NULL,
            query_hash TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sem_cache_created ON semantic_query_cache(created_ts)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sem_cache_model_dim ON semantic_query_cache(model, dim)"
    )
    _migrate_semantic_cache_hash(conn)
    conn.commit()


def _migrate_semantic_cache_hash(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(semantic_query_cache)")}
    if "query_hash" not in cols:
        try:
            conn.execute("ALTER TABLE semantic_query_cache ADD COLUMN query_hash TEXT")
        except sqlite3.OperationalError:
            pass
    rows = conn.execute(
        "SELECT id, query_text FROM semantic_query_cache WHERE query_hash IS NULL OR query_hash = ''"
    ).fetchall()
    for rid, qt in rows:
        conn.execute(
            "UPDATE semantic_query_cache SET query_hash = ? WHERE id = ?",
            (_hash_query(str(qt or "")), int(rid)),
        )
    try:
        conn.execute(
            """
            DELETE FROM semantic_query_cache WHERE id NOT IN (
                SELECT MAX(id) FROM semantic_query_cache GROUP BY model, query_hash
            )
            """
        )
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_sem_cache_model_qhash "
            "ON semantic_query_cache(model, query_hash)"
        )
    except sqlite3.OperationalError:
        pass


def _vec_to_blob(vec) -> tuple[bytes, int]:
    import numpy as np

    a = np.asarray(vec, dtype=np.float32).reshape(-1)
    return a.tobytes(), int(a.shape[0])


def _blob_cosine(a: bytes, b: bytes, dim: int) -> float:
    import numpy as np

    x = np.frombuffer(a, dtype=np.float32, count=dim)
    y = np.frombuffer(b, dtype=np.float32, count=dim)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def lookup_cached_answer(conn: sqlite3.Connection, question: str) -> str | None:
    if not _kb_semantic_cache_enabled() or not (question or "").strip():
        return None
    qstrip = question.strip()
    mname = _embedding_model()[:200]
    hq = _hash_query(qstrip)
    try:
        ensure_semantic_cache_table(conn)
        row = conn.execute(
            """
            SELECT answer_text FROM semantic_query_cache
            WHERE model = ? AND query_hash = ?
            LIMIT 1
            """,
            (mname, hq),
        ).fetchone()
        if row and str(row[0] or "").strip():
            return str(row[0]).strip()
    except Exception:
        pass

    try:
        n0 = int(conn.execute("SELECT COUNT(*) FROM semantic_query_cache").fetchone()[0])
        if n0 <= 0:
            return None
        enc = _get_model()
        qv, dim = _vec_to_blob(
            enc.encode(qstrip, normalize_embeddings=True, show_progress_bar=False)
        )
    except Exception:
        return None

    rows = conn.execute(
        """
        SELECT vec, dim, model, answer_text FROM semantic_query_cache
        WHERE dim = ? AND model = ?
        ORDER BY created_ts DESC
        LIMIT ?
        """,
        (dim, mname, _max_entries()),
    ).fetchall()
    if not rows:
        return None
    thr = _similarity_threshold()
    eps = _vector_unit_eps()
    try:
        import numpy as np

        q = np.frombuffer(qv, dtype=np.float32, count=dim).astype(np.float64, copy=False)
        qn = float(np.linalg.norm(q))
        if abs(qn - 1.0) >= eps:
            q = q / max(qn, 1e-12)
        mats: list = []
        anss: list[str] = []
        for blob, rdim, _mname, ans in rows:
            if rdim != dim or len(blob) != dim * 4:
                continue
            v = np.frombuffer(blob, dtype=np.float32, count=dim).astype(np.float64, copy=False)
            vn = float(np.linalg.norm(v))
            if abs(vn - 1.0) < eps:
                mats.append(v)
            else:
                mats.append(v / max(vn, 1e-12))
            anss.append(str(ans or ""))
        if not mats:
            return None
        M = np.stack(mats, axis=0)
        sims = M @ q
        j = int(np.argmax(sims))
        if float(sims[j]) >= thr:
            return anss[j] if anss[j] else None
    except Exception:
        best = (0.0, "")
        for blob, rdim, _mname, ans in rows:
            if rdim != dim or len(blob) != dim * 4:
                continue
            sim = _blob_cosine(qv, blob, dim)
            if sim > best[0] and sim >= thr:
                best = (sim, str(ans or ""))
        return best[1] if best[1] else None
    return None


def store_cached_answer(conn: sqlite3.Connection, question: str, answer: str) -> None:
    if not _kb_semantic_cache_write_enabled() or not _kb_semantic_cache_enabled():
        return
    q = (question or "").strip()
    ans = (answer or "").strip()
    if not q or not ans or len(ans) < 8:
        return
    try:
        ensure_semantic_cache_table(conn)
        enc = _get_model()
        vec, dim = _vec_to_blob(
            enc.encode(q, normalize_embeddings=True, show_progress_bar=False)
        )
        mname = _embedding_model()[:200]
        hq = _hash_query(q)
        ts = time.time()
        try:
            conn.execute(
                """
                INSERT INTO semantic_query_cache(
                    query_text, model, dim, vec, answer_text, created_ts, query_hash
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(model, query_hash) DO UPDATE SET
                    query_text = excluded.query_text,
                    vec = excluded.vec,
                    answer_text = excluded.answer_text,
                    created_ts = excluded.created_ts,
                    dim = excluded.dim
                """,
                (q[:4000], mname, dim, vec, ans[:120000], ts, hq),
            )
        except sqlite3.OperationalError:
            conn.execute(
                "DELETE FROM semantic_query_cache WHERE model = ? AND query_hash = ?",
                (mname, hq),
            )
            conn.execute(
                """
                INSERT INTO semantic_query_cache(
                    query_text, model, dim, vec, answer_text, created_ts, query_hash
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (q[:4000], mname, dim, vec, ans[:120000], ts, hq),
            )
        n = int(conn.execute("SELECT COUNT(*) FROM semantic_query_cache").fetchone()[0])
        cap = _max_entries()
        if n > cap:
            conn.execute(
                """
                DELETE FROM semantic_query_cache WHERE id IN (
                    SELECT id FROM semantic_query_cache ORDER BY created_ts ASC LIMIT ?
                )
                """,
                (n - cap,),
            )
        conn.commit()
    except Exception:
        pass
