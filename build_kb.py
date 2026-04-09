"""
================================================================================
build_kb.py —— 「把书变成可搜索的数据库」
================================================================================

整体在做什么（RAG 的第一步：Indexing / 索引）：
  你在文件夹里放了 epub、pdf、txt → 本脚本读取文字 → 切成小段 → 存进 SQLite，
  并创建「全文检索」表，后面 ask_kb.py / app.py 才能按问题把相关段落捞出来。

建议阅读顺序（对入门者最顺）：
  1) 拉到文件最下面的 main()：命令行参数、循环处理每个文件。
  2) upsert_document()：单本书「插入或更新」到数据库的完整流程。
  3) chunk_text() / chunk_text_semantic()：固定句界切块或基于句向量「语义断崖」切块（KB_CHUNK_STRATEGY）。
  4) init_db()：两张普通表 documents / chunks + 一张 fts5 虚拟表 chunks_fts。

关键名词（记不住也没关系，对照代码看）：
  - chunk（片段）：一长篇文章被切成多段文字，每段一行存在 chunks 表里。
  - FTS5：SQLite 自带的全文搜索引擎；MATCH 查询要靠 text_fts 里的分词结果。
  - jieba：中文分词库；英文可按「字符」理解为把句子拆成更小的检索单元。
  - sha256：文件的「内容指纹」；内容没变就跳过重复解析，加快重建。
================================================================================
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

import jieba
import jieba.analyse
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader


# 只处理这三种后缀；其它文件即使放在目录里也会被忽略
SUPPORTED_EXTS = {".epub", ".pdf", ".txt"}
# 避免把项目自己的依赖说明文件误当成「书籍」索引进去
IGNORE_FILENAMES = {"requirements-kb.txt", "requirements.txt"}


def _mtime_fingerprint_match(stored: float, disk: float) -> bool:
    """磁盘 mtime 与库中记录的比对；放宽 2s 以兼容 FAT 等低精度时间戳。"""
    return math.isclose(float(stored), float(disk), rel_tol=0.0, abs_tol=2.0)


def _configure_connection_for_bulk_index(conn: sqlite3.Connection) -> None:
    """大批量写入时降低 fsync 频率、增大缓存，显著加快全量重建（异常断电风险略增，属本地索引可接受）。"""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA temp_store=MEMORY")
    except sqlite3.OperationalError:
        pass


def _ensure_documents_fsize_column(conn: sqlite3.Connection) -> None:
    """旧库无 fsize 列时补齐，用于 mtime+size 快速跳过未改动文件。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "fsize" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN fsize INTEGER NOT NULL DEFAULT -1")
        conn.commit()


def _ensure_documents_index_profile_column(conn: sqlite3.Connection) -> None:
    """记录切块策略 sentence / semantic；策略变更时触发重新切块。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "index_profile" not in cols:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN index_profile TEXT NOT NULL DEFAULT 'sentence'"
        )
        conn.commit()

# 无环境变量、无 kb_library_root.txt 时使用的默认资料目录（可与项目不在同一盘）
_DEFAULT_MATERIALS_ROOT = Path(r"E:\Article organization\books")


def default_library_root() -> Path:
    """网页与脚本共用的默认资料根目录（目录可尚未创建）。"""
    try:
        return _DEFAULT_MATERIALS_ROOT.expanduser().resolve()
    except OSError:
        return _DEFAULT_MATERIALS_ROOT.expanduser()


def library_root() -> Path:
    """
    「资料根目录」：默认约定为你的电子书放在 kb_project 文件夹的**上一级**。

    __file__ 是当前脚本路径；.parent 是 kb_project；再 .parent 就是上一级。
    """
    return Path(__file__).resolve().parent.parent


def library_root_config_path() -> Path:
    """与 app 共用的资料路径配置文件，位于 kb_project 目录下。"""
    return Path(__file__).resolve().parent / "kb_library_root.txt"


def read_configured_library_root() -> Path | None:
    """读取网页或脚本写入的自定义资料根目录；文件不存在或无效则返回 None。"""
    cfg = library_root_config_path()
    if not cfg.is_file():
        return None
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return None
    line = (text.strip().splitlines() or [""])[0].strip()
    if not line:
        return None
    try:
        return Path(line).expanduser().resolve()
    except OSError:
        return None


def write_configured_library_root(root: Path) -> None:
    """将资料根目录写入配置文件，供下次启动与 effective_library_root 使用。"""
    path = root.expanduser().resolve()
    library_root_config_path().write_text(str(path) + "\n", encoding="utf-8")


def clear_configured_library_root() -> None:
    """删除配置文件，使 effective_library_root 回退到默认。"""
    p = library_root_config_path()
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass


def effective_library_root() -> Path:
    """
    实际使用的资料根目录，优先级：
      1) 环境变量 KB_LIBRARY_ROOT（适合脚本/自动化）
      2) 项目目录下 kb_library_root.txt（网页「保存资料路径」写入）
      3) default_library_root()（默认 E:\\Article organization\\books）
    环境变量与配置文件中的路径须已存在且为目录；否则回退到下一优先级。
    """
    env = (os.environ.get("KB_LIBRARY_ROOT") or "").strip()
    if env:
        try:
            p = Path(env).expanduser().resolve()
            if p.is_dir():
                return p
        except OSError:
            pass
    cfg = read_configured_library_root()
    if cfg is not None and cfg.is_dir():
        return cfg
    return default_library_root()


def delete_document_from_index(conn: sqlite3.Connection, stored_path: str) -> bool:
    """从数据库里删掉某本书的索引（documents + chunks + 全文表），**不**删除硬盘上的原文件。"""
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (stored_path,)).fetchone()
    if not row:
        return False
    doc_id = row[0]
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    return True


def reindex_all(
    conn: sqlite3.Connection,
    root: Path | None = None,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    「全量对齐」磁盘与数据库：你已经删了/改名了的文件，索引里也删掉；
    还在的文件，若内容变了会重新切块写库（upsert_document）。
    网页里「全量重建索引」按钮最终会调到这个函数。

    progress_callback：可选，每处理一个文件前调用一次 (1-based 序号, 总数, 当前文件路径)，供 UI 后台任务报进度。
    """
    root = root.resolve() if root is not None else effective_library_root().resolve()
    if not root.is_dir():
        return {
            "processed": 0,
            "documents": 0,
            "chunks": 0,
            "errors": [
                (
                    str(root),
                    "资料根目录不存在或不是文件夹，请先创建该目录或在本页侧边栏修改路径后再重建。",
                )
            ],
            "graph_edges": 0,
        }
    files = scan_files(root)
    init_db(conn)
    if not files:
        conn.execute("DELETE FROM documents")
        conn.commit()
        return {"processed": 0, "documents": 0, "chunks": 0, "errors": [], "graph_edges": 0}
    valid_paths = {str(p.as_posix()) for p in files}
    in_clause = ",".join(["?"] * len(valid_paths))
    conn.execute(f"DELETE FROM documents WHERE path NOT IN ({in_clause})", tuple(valid_paths))
    conn.commit()
    _configure_connection_for_bulk_index(conn)
    errors: list[tuple[str, str]] = []
    processed = 0
    total_files = len(files)
    for fi, fp in enumerate(files, start=1):
        if progress_callback is not None:
            progress_callback(fi, total_files, str(fp))
        try:
            conn.execute("SAVEPOINT kb_upsert")
            upsert_document(conn, fp, commit=False)
            conn.execute("RELEASE SAVEPOINT kb_upsert")
            processed += 1
        except Exception as exc:
            try:
                conn.execute("ROLLBACK TO SAVEPOINT kb_upsert")
            except sqlite3.OperationalError:
                try:
                    conn.rollback()
                except sqlite3.OperationalError:
                    pass
            errors.append((str(fp), str(exc)))
    try:
        conn.commit()
    except sqlite3.OperationalError:
        pass
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    graph_edges = 0
    if _kb_build_graph_enabled() and chunks > 0:
        try:
            graph_edges = rebuild_chunk_graph(conn)
        except Exception as exc:
            safe_print(f"[build_kb] chunk 关联图重建失败（可稍后重跑全量索引）: {exc}")
    return {
        "processed": processed,
        "documents": docs,
        "chunks": chunks,
        "errors": errors,
        "graph_edges": graph_edges,
    }

# Windows 控制台默认编码可能不是 UTF-8，打印中文不乱码的一线防守
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_text(text: str) -> str:
    """统一换行、去掉多余空白，让后面切块更稳定。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\u3000", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraph_into_sentences(paragraph: str) -> list[str]:
    """
    将单段文字拆成尽量完整的句子（保留句末标点）。
    优先按中文句末切分，余下部分再按英文句末 + 空格切分。
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    parts: list[str] = []
    used = 0
    for m in re.finditer(r"[^。！？]*[。！？]+", paragraph):
        s = m.group(0).strip()
        if s:
            parts.append(s)
        used = m.end()
    tail = paragraph[used:].strip()
    if tail:
        if re.search(r"[.!?](?:\s|$)", tail):
            for piece in re.split(r"(?<=[.!?])\s+", tail):
                piece = piece.strip()
                if piece:
                    parts.append(piece)
        else:
            parts.append(tail)
    if not parts:
        parts = [paragraph]
    return parts


def _hard_split_long_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    单句超过 chunk_size 时的降级切分：优先在弱标点（；，、；;,.）处断，其次句末，最后硬切并带重叠。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end >= n:
            chunk = text[start:n].strip()
            if chunk:
                out.append(chunk)
            break
        window = text[start:end]
        cut = -1
        for punct in "；;，,、":
            k = window.rfind(punct)
            if k > max(40, chunk_size // 4):
                cut = max(cut, k + 1)
        if cut <= 0:
            for k in range(len(window) - 1, max(0, len(window) - 100), -1):
                if window[k] in "。！？.!?":
                    cut = k + 1
                    break
        if cut <= 0:
            cut = min(chunk_size, n - start)
        piece = text[start : start + cut].strip()
        if piece:
            out.append(piece)
        if start + cut >= n:
            break
        start = max(start + cut - overlap, start + 1)
    return [x for x in out if x]


def _pack_sentences_into_chunks(
    sentences: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """把一句列表打包成若干条，每条总长不超过 chunk_size，不在句中切断。"""
    chunks: list[str] = []
    buf: list[str] = []
    blen = 0
    for s in sentences:
        slen = len(s)
        if not buf:
            if slen <= chunk_size:
                buf = [s]
                blen = slen
            else:
                chunks.extend(_hard_split_long_sentence(s, chunk_size, overlap))
            continue
        if blen + slen <= chunk_size:
            buf.append(s)
            blen += slen
        else:
            chunks.append("".join(buf))
            if slen <= chunk_size:
                buf = [s]
                blen = slen
            else:
                chunks.extend(_hard_split_long_sentence(s, chunk_size, overlap))
                buf = []
                blen = 0
    if buf:
        chunks.append("".join(buf))
    return chunks


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """
    把一整本书的纯文本切成多条字符串列表。

    默认以 chunk_size（约 900 字）为目标总长，但优先在**完整句边界**打包，而不是「每 N 字硬切一刀」。
    段内先断句再组块；段落之间仍可在总长允许时合并。单句过长时依次尝试弱标点、句末，最后才硬切（带 overlap）。
    """
    text = normalize_text(text)
    if not text:
        return []
    paragraphs: list[str] = []
    for raw in text.split("\n\n"):
        p = re.sub(r"[\n\r\t]+", " ", raw.strip()).strip()
        if p:
            paragraphs.append(p)
    chunks: list[str] = []
    current = ""

    for p in paragraphs:
        sents = _split_paragraph_into_sentences(p)
        para_joined = "".join(sents) if sents else p
        trial = f"{current}\n\n{para_joined}" if current else para_joined
        if len(trial) <= chunk_size:
            current = trial
            continue
        if current:
            chunks.append(current.strip())
            current = ""
        if len(para_joined) <= chunk_size:
            current = para_joined
        else:
            packed = _pack_sentences_into_chunks(sents, chunk_size, overlap)
            for i, pc in enumerate(packed):
                if i < len(packed) - 1:
                    chunks.append(pc)
                else:
                    current = pc
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def tokenize_for_fts(text: str) -> str:
    """
    把一段中文切成词，用空格拼成一长串，给 FTS5 存进 text_fts 列。

    为什么要有这一步：全文检索引擎按「词」匹配；整句不设边界时，中文连在一起不好搜。
    """
    return " ".join([t.strip() for t in jieba.lcut(text) if t.strip()])


def read_txt(path: Path) -> str:
    """尝试多种常见中文编码；老文档经常是 GBK 而不是 UTF-8。"""
    for enc in ("utf-8", "gb18030", "gbk", "big5"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    """逐页把 PDF 里能抠出来的文字拼成一个大字符串（处理不了全图片扫描版 PDF）。"""
    reader = PdfReader(str(path))
    return "\n\n".join([(p.extract_text() or "") for p in reader.pages])


def _epub_item_to_text(item) -> str:
    """从 EPUB 的 HTML 条目中抽出正文；去掉 script/style/nav 减少目录噪声。"""
    if item is None or item.get_type() != ITEM_DOCUMENT:
        return ""
    try:
        raw = item.get_content()
    except Exception:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all("nav"):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def read_epub(path: Path) -> tuple[str, str]:
    """
    读 epub：元数据里拿书名；正文优先按 spine 阅读顺序解析，避免仅靠 get_items()
    时顺序错乱或漏掉 spine 条目（不少书上会因此只有极少量或空正文）。
    """
    book = epub.read_epub(str(path))
    title = "Untitled"
    meta = book.get_metadata("DC", "title")
    if meta and meta[0]:
        title = meta[0][0]

    docs: list[str] = []
    seen: set[str] = set()

    def _resolve_spine_item(key: str):
        it = book.get_item_with_id(key)
        if it is None:
            it = book.get_item_with_href(key)
        return it

    spine = getattr(book, "spine", None) or []
    for entry in spine:
        raw = entry[0] if isinstance(entry, (list, tuple)) else entry
        if raw is None:
            continue
        key = str(raw).strip()
        if not key:
            continue
        item = _resolve_spine_item(key)
        if item is None:
            continue
        uid = item.get_id() or item.get_name() or key
        if uid in seen:
            continue
        seen.add(uid)
        t = _epub_item_to_text(item)
        if t:
            docs.append(t)

    try:
        rest = list(book.get_items_of_type(ITEM_DOCUMENT))
    except Exception:
        rest = []
    if not rest:
        rest = [i for i in book.get_items() if i.get_type() == ITEM_DOCUMENT]
    for item in rest:
        uid = item.get_id() or item.get_name() or ""
        if uid in seen:
            continue
        seen.add(uid)
        t = _epub_item_to_text(item)
        if t:
            docs.append(t)

    return title, "\n\n".join(docs)


def file_hash(path: Path) -> str:
    """对文件二进制内容算 SHA256；用来判断「文件改没改过」。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        print(encoded.decode(sys.stdout.encoding or "utf-8", errors="replace"))


def init_db(conn: sqlite3.Connection) -> None:
    """
    创建三张「逻辑」结构：
      documents —— 每本书一行（路径、标题、扩展名、哈希、修改时间）
      chunks    —— 每本书切出来的许多段；text_raw 给人看，text_fts 给搜索引擎看
      chunks_fts —— FTS5 虚拟表，和 chunks 通过 rowid 关联（content='chunks' 表示同步）
    """
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE NOT NULL, title TEXT, ext TEXT NOT NULL, sha256 TEXT NOT NULL, mtime REAL NOT NULL, fsize INTEGER NOT NULL DEFAULT -1, index_profile TEXT NOT NULL DEFAULT 'sentence')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL, text_raw TEXT NOT NULL, text_fts TEXT NOT NULL, FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE)"
    )
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text_fts, content='chunks', content_rowid='id')")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    _ensure_documents_fsize_column(conn)
    _ensure_documents_index_profile_column(conn)
    _ensure_chunk_vectors_table(conn)
    _ensure_chunk_graph_table(conn)
    _ensure_chunk_meta_columns(conn)


def _ensure_chunk_meta_columns(conn: sqlite3.Connection) -> None:
    """meta_json：索引阶段写入标题/路径等，检索时直接使用，避免再读源文件。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "meta_json" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN meta_json TEXT")
    if "context_text" not in cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN context_text TEXT"
        )
    conn.commit()


def _ensure_chunk_vectors_table(conn: sqlite3.Connection) -> None:
    """句向量 BLOB（float32），供混合检索；与 chunks 同行级联删除。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_vectors (
            chunk_id INTEGER PRIMARY KEY NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _ensure_chunk_graph_table(conn: sqlite3.Connection) -> None:
    """chunk 间有向边（邻接、关键词桥接），供 GraphRAG 扩展检索；删 chunk 时级联删边。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_graph_edges (
            src_chunk_id INTEGER NOT NULL,
            dst_chunk_id INTEGER NOT NULL,
            rel TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (src_chunk_id, dst_chunk_id),
            FOREIGN KEY(src_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
            FOREIGN KEY(dst_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_graph_edges_dst ON chunk_graph_edges(dst_chunk_id)"
    )
    conn.commit()


def _kb_build_graph_enabled() -> bool:
    return os.environ.get("KB_BUILD_GRAPH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def rebuild_chunk_graph(conn: sqlite3.Connection) -> int:
    """
    全量重建 chunk 关联图：同书相邻块双向边；同文档内窗口距离≤5 且关键词交集≥2 的块对（keyword 边）。
    在「全量重建索引」末尾调用；单文件 upsert 后图可能暂时陈旧，需全量对齐时刷新。
    """
    _ensure_chunk_graph_table(conn)
    conn.execute("DELETE FROM chunk_graph_edges")
    rows = conn.execute(
        """
        SELECT c.id, c.doc_id, c.chunk_index, c.text_raw
        FROM chunks c
        ORDER BY c.doc_id, c.chunk_index
        """
    ).fetchall()
    by_doc: dict[int, list[tuple[int, str, int]]] = {}
    for cid, did, cidx, text in rows:
        did = int(did)
        by_doc.setdefault(did, []).append((int(cid), text or "", int(cidx)))
    edge_set: set[tuple[int, int, str, float]] = set()
    win = 5
    for _did, parts in by_doc.items():
        parts.sort(key=lambda x: x[2])
        ids = [p[0] for p in parts]
        texts = [p[1] for p in parts]
        n = len(ids)
        if n < 2:
            continue
        for i in range(n - 1):
            a, b = ids[i], ids[i + 1]
            edge_set.add((a, b, "adjacent", 1.0))
            edge_set.add((b, a, "adjacent", 1.0))
        tag_sets = [set(jieba.analyse.extract_tags(t, topK=8)) for t in texts]
        for i in range(n):
            hi = min(n, i + win + 1)
            for j in range(i + 1, hi):
                inter = tag_sets[i] & tag_sets[j]
                if len(inter) < 2:
                    continue
                w = min(1.0, len(inter) / 8.0)
                a, b = ids[i], ids[j]
                edge_set.add((a, b, "keyword", w))
                edge_set.add((b, a, "keyword", w))
    batch = list(edge_set)
    step = 4000
    for i in range(0, len(batch), step):
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunk_graph_edges(src_chunk_id, dst_chunk_id, rel, weight)
            VALUES (?,?,?,?)
            """,
            batch[i : i + step],
        )
    conn.commit()
    return len(batch)


_embedding_for_index = None


def _embedding_model_name() -> str:
    return (
        os.environ.get("KB_EMBEDDING_MODEL")
        or os.environ.get("KB_SEMANTIC_MODEL")
        or "paraphrase-multilingual-MiniLM-L12-v2"
    ).strip()


def _kb_vector_index_write_enabled() -> bool:
    if os.environ.get("KB_VECTOR_INDEX", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        return False
    return True


def _get_embedding_model_for_index():
    global _embedding_for_index
    if _embedding_for_index is not None:
        return _embedding_for_index
    from sentence_transformers import SentenceTransformer

    _embedding_for_index = SentenceTransformer(_embedding_model_name())
    return _embedding_for_index


def _store_chunk_vector(conn: sqlite3.Connection, chunk_id: int, text: str) -> None:
    if not _kb_vector_index_write_enabled() or not text.strip():
        return
    import numpy as np

    model = _get_embedding_model_for_index()
    emb = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vec = np.asarray(emb, dtype=np.float32).tobytes()
    dim = int(emb.shape[0])
    mname = _embedding_model_name()[:200]
    conn.execute(
        "INSERT OR REPLACE INTO chunk_vectors(chunk_id, model, dim, vec) VALUES(?,?,?,?)",
        (chunk_id, mname, dim, vec),
    )


def index_chunk_profile() -> str:
    """当前索引使用的切块配置：sentence | semantic；写入 documents.index_profile。"""
    s = (os.environ.get("KB_CHUNK_STRATEGY") or "sentence").strip().lower()
    if s in ("semantic", "sem", "embedding", "vector"):
        return "semantic"
    return "sentence"


_sentence_encoder = None


def _get_sentence_encoder():
    """懒加载 multilingual MiniLM（语义切块）；首次运行会下载权重。"""
    global _sentence_encoder
    if _sentence_encoder is not None:
        return _sentence_encoder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "语义切块需安装可选依赖：pip install -r requirements-optional.txt"
        ) from exc
    name = os.environ.get(
        "KB_SEMANTIC_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )
    _sentence_encoder = SentenceTransformer(name)
    return _sentence_encoder


def _flatten_document_sentences(text: str) -> list[str]:
    """将全书规范化文本拆成有序句列表（跨段）。"""
    text = normalize_text(text)
    if not text:
        return []
    out: list[str] = []
    for raw in text.split("\n\n"):
        p = re.sub(r"[\n\r\t]+", " ", raw.strip()).strip()
        if p:
            out.extend(_split_paragraph_into_sentences(p))
    return [s.strip() for s in out if s.strip()]


def chunk_text_semantic(
    text: str, chunk_size: int = 900, overlap: int = 120
) -> list[str]:
    """
    语义切块：相邻句 embedding 的余弦距离在全体邻接边中偏大的位置视为「话题转折」，优先断块。
    仍受 chunk_size 约束；超长块再走 _hard_split_long_sentence。

    环境变量：KB_SEMANTIC_DISTANCE_PERCENTILE（默认 78，越大断点越少）、
    KB_SEMANTIC_MIN_CHUNK（默认 150）、KB_SEMANTIC_MODEL。
    """
    sentences = _flatten_document_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= 2:
        return chunk_text(text, chunk_size, overlap)

    import numpy as np

    model = _get_sentence_encoder()
    emb = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    dists = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
    pct = float(os.environ.get("KB_SEMANTIC_DISTANCE_PERCENTILE", "78"))
    pct = max(55.0, min(99.5, pct))
    thresh = float(np.percentile(dists, pct))
    min_chunk = int(os.environ.get("KB_SEMANTIC_MIN_CHUNK", "150"))

    raw_chunks: list[str] = []
    start = 0
    n = len(sentences)
    while start < n:
        end = start + 1
        acc_len = len(sentences[start])
        while end < n:
            gap = float(dists[end - 1])
            next_s = sentences[end]
            next_len = len(next_s)
            if acc_len + next_len > chunk_size:
                break
            if (
                acc_len >= min_chunk
                and gap >= thresh
                and end > start
            ):
                break
            acc_len += next_len
            end += 1
        raw_chunks.append("".join(sentences[start:end]))
        if end >= n:
            break
        next_start = end
        if overlap > 0 and end > start:
            o_budget = 0
            j = end - 1
            while j >= start and o_budget < overlap:
                o_budget += len(sentences[j])
                j -= 1
            next_start = max(start + 1, j + 1)
        start = next_start

    final: list[str] = []
    for c in raw_chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) <= chunk_size:
            final.append(c)
        else:
            final.extend(_hard_split_long_sentence(c, chunk_size, overlap))
    return [x for x in final if x]


def chunk_text_for_index(
    text: str, chunk_size: int = 900, overlap: int = 120
) -> list[str]:
    """建库时统一入口：按 KB_CHUNK_STRATEGY 选句界或语义切块。"""
    if index_chunk_profile() == "semantic":
        return chunk_text_semantic(text, chunk_size, overlap)
    return chunk_text(text, chunk_size, overlap)


def upsert_document(
    conn: sqlite3.Connection,
    path: Path,
    *,
    commit: bool = True,
) -> None:
    """
    upsert = update + insert：有则更新，无则插入。

    若 mtime+文件大小与库一致且 **index_profile（切块策略）未变** 且已有 chunks：快速跳过。
    若全文 SHA256 未变且 chunks 完整：仅更新 mtime/size/index_profile。
    内容或 KB_CHUNK_STRATEGY 变化：删旧 chunks 再按 chunk_text_for_index 写入。
    commit=False 时由调用方在批量结束后统一 commit（全量重建）。
    """
    ext = path.suffix.lower()
    st = path.stat()
    mtime_f = st.st_mtime
    size_i = int(st.st_size)
    rel_path = str(path.expanduser().resolve().as_posix())

    profile = index_chunk_profile()
    row = conn.execute(
        """
        SELECT id, sha256, mtime, COALESCE(fsize, -1), COALESCE(index_profile, 'sentence')
        FROM documents WHERE path = ?
        """,
        (rel_path,),
    ).fetchone()
    doc_id: int | None
    if row is not None:
        doc_id = int(row[0])
        old_sha, old_mtime, old_sz, old_profile = (
            str(row[1]),
            float(row[2]),
            int(row[3]),
            str(row[4] or "sentence"),
        )
        n_exist = int(
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
        )
        if (
            n_exist > 0
            and old_sz == size_i
            and _mtime_fingerprint_match(old_mtime, mtime_f)
            and old_profile == profile
        ):
            return
    else:
        doc_id = None
        old_sha = ""
        old_profile = "sentence"

    sha = file_hash(path)
    if doc_id is not None and old_sha == sha:
        n_exist = int(
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
        )
        if n_exist > 0 and old_profile == profile:
            conn.execute(
                "UPDATE documents SET mtime = ?, fsize = ?, index_profile = ? WHERE id = ?",
                (mtime_f, size_i, profile, doc_id),
            )
            if commit:
                conn.commit()
            return

    if ext == ".txt":
        title, content = path.stem, read_txt(path)
    elif ext == ".pdf":
        title, content = path.stem, read_pdf(path)
    elif ext == ".epub":
        title, content = read_epub(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    content_stripped = normalize_text(content)
    if not content_stripped:
        if ext == ".pdf":
            raise ValueError(
                "未从 PDF 提取到文字（多为纯扫描图片版；可尝试 OCR 工具转为可检索 PDF）"
            )
        if ext == ".epub":
            raise ValueError(
                "未从 EPUB 解析到正文（可能加密、非标准打包或仅含图片；可换来源或解压检查 xhtml）"
            )
    if doc_id is not None:
        conn.execute(
            "UPDATE documents SET title = ?, ext = ?, sha256 = ?, mtime = ?, fsize = ?, index_profile = ? WHERE id = ?",
            (title, ext, sha, mtime_f, size_i, profile, doc_id),
        )
        conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    else:
        cur = conn.execute(
            "INSERT OR IGNORE INTO documents(path, title, ext, sha256, mtime, fsize, index_profile) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (rel_path, title, ext, sha, mtime_f, size_i, profile),
        )
        doc_id = cur.lastrowid or int(
            conn.execute("SELECT id FROM documents WHERE path = ?", (rel_path,)).fetchone()[0]
        )
    try:
        span = max(0, min(4000, int(os.environ.get("KB_SEARCH_SPAN_CHARS", "0") or "0")))
    except ValueError:
        span = 0
    meta = json.dumps(
        {"doc_title": title, "rel_path": rel_path, "ext": ext},
        ensure_ascii=False,
    )
    for i, c in enumerate(chunk_text_for_index(content)):
        if span > 0 and len(c) > span:
            fts_src = c[:span]
            vec_src = fts_src
        else:
            fts_src = c
            vec_src = c
        tok = tokenize_for_fts(fts_src)
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, chunk_index, text_raw, text_fts, meta_json) VALUES(?, ?, ?, ?, ?)",
            (doc_id, i, c, tok, meta),
        )
        chunk_row_id = int(cur.lastrowid)
        conn.execute("INSERT INTO chunks_fts(rowid, text_fts) VALUES(?, ?)", (chunk_row_id, tok))
        try:
            _store_chunk_vector(conn, chunk_row_id, vec_src)
        except Exception:
            pass
    if commit:
        conn.commit()


def scan_files(root: Path) -> list[Path]:
    """递归遍历 root 下所有文件，筛出支持的电子书后缀，并按路径排序（输出顺序稳定）。"""
    root = root.expanduser().resolve()
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            if p.name.startswith("~$"):
                continue
            if p.name.lower() in IGNORE_FILENAMES:
                continue
            files.append(p.resolve())
    return sorted(files)


def main() -> None:
    """
    命令行入口：--root 指向放书的文件夹，--db 指向生成的 .sqlite 文件。

    流程摘要：扫文件 → 连数据库 → 删掉「库里已有但磁盘上已不存在」的路径 → 逐个 upsert。
    """
    parser = argparse.ArgumentParser(description="Build local KB from epub/pdf/txt files.")
    parser.add_argument("--root", default="..", help="Root directory containing source files.")
    parser.add_argument("--db", default="knowledge_base.sqlite", help="Output sqlite database path.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    db_path = Path(args.db).resolve()
    files = scan_files(root)
    if not files:
        safe_print("No supported files found.")
        return
    conn = sqlite3.connect(str(db_path))
    init_db(conn)
    _configure_connection_for_bulk_index(conn)
    valid_paths = {str(p.as_posix()) for p in files}
    conn.execute("DELETE FROM documents WHERE path NOT IN ({})".format(",".join(["?"] * len(valid_paths))) if valid_paths else "DELETE FROM documents", tuple(valid_paths))
    conn.commit()
    processed = 0
    for fp in files:
        try:
            conn.execute("SAVEPOINT kb_upsert")
            upsert_document(conn, fp, commit=False)
            conn.execute("RELEASE SAVEPOINT kb_upsert")
            processed += 1
            safe_print(f"[OK] {fp}")
        except Exception as exc:
            try:
                conn.execute("ROLLBACK TO SAVEPOINT kb_upsert")
            except sqlite3.OperationalError:
                try:
                    conn.rollback()
                except sqlite3.OperationalError:
                    pass
            safe_print(f"[ERR] {fp}: {exc}")
    try:
        conn.commit()
    except sqlite3.OperationalError:
        pass
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    safe_print(json.dumps({"processed_files": processed, "documents": docs, "chunks": chunks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
