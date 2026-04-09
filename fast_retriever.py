"""
================================================================================
fast_retriever.py —— 高性能RAG检索模块
================================================================================

基于db_optimizer的系统性优化，提供极速检索能力：
1. 并行混合检索 - FTS和向量检索并行执行
2. 智能缓存策略 - 多层级缓存减少重复计算
3. 批量处理 - 向量化操作替代循环
4. 预过滤优化 - 提前剪枝减少候选集
5. 内存优化 - 避免不必要的数据拷贝

性能提升预期：
- 纯FTS检索: 2-3x
- 混合检索: 3-5x
- 向量检索: 5-10x (使用IVF索引)

使用方法：
    from fast_retriever import FastRetriever
    retriever = FastRetriever(conn)
    results = retriever.retrieve("查询问题", top_k=4)
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import jieba
import jieba.analyse
import numpy as np

from db_optimizer import (
    BatchRetriever,
    VectorIndexManager,
    get_optimized_connection,
    pooled_connection,
)

# 停用词集合（与ask_kb保持一致）
_FTS_STOPWORDS: frozenset[str] = frozenset({
    "的", "了", "和", "是", "在", "有", "与", "及", "或", "等", "为", "以", "而", "之", "也", "都", "就", "要", "会", "能", "可",
    "这", "那", "其", "中", "对", "从", "到", "把", "被", "将", "让", "使", "由", "于", "吗", "呢", "吧", "啊", "么", "嘛",
    "很", "最", "更", "还", "又", "再", "什么", "怎么", "如何", "哪些", "是否", "可以", "应该", "如果", "因为", "所以", "但是",
    "然后", "一个", "一些", "这种", "这样", "进行", "通过", "根据", "关于", "以及", "或者", "还有", "不是", "没有", "需要", "请",
    "你", "我", "我们", "你们", "他们", "它们",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "and", "but", "if", "or", "because", "until", "while", "although", "though", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "it", "its", "their", "theirs",
})


class FastRetriever:
    """高性能检索器"""
    
    def __init__(self, conn: sqlite3.Connection | None = None, db_path: str | None = None):
        """
        初始化高性能检索器
        
        Args:
            conn: 现有数据库连接（优先使用）
            db_path: 数据库路径（如果conn未提供）
        """
        self._external_conn = conn is not None
        self.conn = conn
        self.db_path = db_path
        self._vector_manager: VectorIndexManager | None = None
        self._batch_retriever: BatchRetriever | None = None
        self._encoder = None
        self._encoder_lock = threading.Lock()
        self._ivf_built = False
        
        if conn:
            self._batch_retriever = BatchRetriever(conn)
    
    def _ensure_connection(self) -> sqlite3.Connection:
        """确保有可用连接"""
        if self.conn is not None:
            return self.conn
        if self.db_path:
            self.conn = get_optimized_connection(self.db_path)
            self._batch_retriever = BatchRetriever(self.conn)
            return self.conn
        raise ValueError("Must provide either conn or db_path")
    
    def _get_encoder(self):
        """懒加载编码器"""
        if self._encoder is not None:
            return self._encoder
        
        with self._encoder_lock:
            if self._encoder is not None:
                return self._encoder
            
            from sentence_transformers import SentenceTransformer
            
            model_name = (
                os.environ.get("KB_EMBEDDING_MODEL")
                or os.environ.get("KB_SEMANTIC_MODEL")
                or "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._encoder = SentenceTransformer(model_name)
            return self._encoder
    
    def build_vector_index(self, n_clusters: int = 16) -> bool:
        """构建向量索引加速检索"""
        conn = self._ensure_connection()
        
        if self._vector_manager is None:
            self._vector_manager = VectorIndexManager(conn)
        
        self._ivf_built = self._vector_manager.build_ivf_index(n_clusters)
        return self._ivf_built
    
    @staticmethod
    @lru_cache(maxsize=512)
    def _extract_keywords(text: str) -> tuple[str, ...]:
        """提取关键词（带缓存）"""
        if not text.strip():
            return tuple()
        
        try:
            tags = jieba.analyse.extract_tags(text, topK=14, withWeight=False)
        except Exception:
            tags = []
        
        # 过滤停用词
        filtered = [t for t in tags if t not in _FTS_STOPWORDS and len(t) > 1]
        
        if len(filtered) < 2:
            # 回退到普通分词
            words = [w.strip() for w in jieba.lcut(text, cut_all=False) if w.strip()]
            filtered = [w for w in words if w not in _FTS_STOPWORDS and len(w) > 1]
        
        # 限制关键词数量
        max_terms = min(14, int(os.environ.get("KB_FTS_MAX_TERMS", "14")))
        return tuple(filtered[:max_terms])
    
    @staticmethod
    @lru_cache(maxsize=512)
    def _build_fts_query(text: str) -> str:
        """构建FTS查询（带缓存）"""
        keywords = FastRetriever._extract_keywords(text)
        if not keywords:
            return text.replace('"', "")
        
        quoted = ['"' + k.replace('"', "") + '"' for k in keywords]
        return " OR ".join(quoted)
    
    def _fts_search(
        self,
        query: str,
        top_k: int,
        keywords: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """FTS检索（优化版）"""
        conn = self._ensure_connection()
        
        oversample = max(1, min(12, int(os.environ.get("KB_RETRIEVE_OVERSAMPLE", "4"))))
        fetch_n = min(200, max(top_k * oversample, top_k))
        
        # 使用优化的查询
        rows = conn.execute(
            """
            SELECT c.id, d.path, COALESCE(d.title, '') AS title, 
                   c.chunk_index, c.text_raw, bm25(chunks_fts) AS score,
                   COALESCE(c.meta_json, '') AS meta_json,
                   COALESCE(c.context_text, '') AS context_text
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, fetch_n),
        ).fetchall()
        
        if not rows:
            return []
        
        # 批量处理结果
        results = []
        bm25_scores = []
        
        for r in rows:
            results.append({
                "chunk_id": r[0],
                "path": r[1],
                "title": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "score": r[5],
                "meta_json": r[6],
                "context_text": r[7],
            })
            bm25_scores.append(float(r[5]))
        
        # 向量化重排
        return self._rerank_results(results, bm25_scores, keywords)
    
    def _vector_search_fast(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """快速向量检索（使用IVF索引）"""
        conn = self._ensure_connection()
        
        # 编码查询
        enc = self._get_encoder()
        qv = enc.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        qv = np.asarray(qv, dtype=np.float32).reshape(-1)
        
        model_name = (
            os.environ.get("KB_EMBEDDING_MODEL")
            or os.environ.get("KB_SEMANTIC_MODEL")
            or "paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 尝试使用IVF索引
        if self._ivf_built and self._vector_manager:
            candidate_ids = self._vector_manager.search_with_ivf(
                qv, model_name, top_k * 4, n_probe=3
            )
            if candidate_ids:
                # 批量获取候选向量
                if self._batch_retriever:
                    vector_data = self._batch_retriever.batch_get_vectors(
                        candidate_ids, model_name
                    )
                else:
                    placeholders = ",".join(["?"] * len(candidate_ids))
                    rows = conn.execute(
                        f"""
                        SELECT chunk_id, vec FROM chunk_vectors
                        WHERE chunk_id IN ({placeholders}) AND model = ?
                        """,
                        candidate_ids + [model_name],
                    ).fetchall()
                    vector_data = [(r[0], r[1]) for r in rows]
                
                if vector_data:
                    return self._compute_similarities(qv, vector_data, top_k)
        
        # 回退到普通向量检索（分批处理避免内存溢出）
        return self._vector_search_batch(query, qv, model_name, top_k)
    
    def _vector_search_batch(
        self,
        query: str,
        qv: np.ndarray,
        model_name: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """分批向量检索（内存优化版）"""
        conn = self._ensure_connection()
        
        batch_size = 1000
        all_results: list[tuple[int, float]] = []
        
        offset = 0
        while True:
            rows = conn.execute(
                """
                SELECT v.chunk_id, v.vec, c.text_raw, d.path, 
                       COALESCE(d.title, ''), c.chunk_index,
                       COALESCE(c.meta_json, ''), COALESCE(c.context_text, '')
                FROM chunk_vectors v
                JOIN chunks c ON c.id = v.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE v.model = ?
                LIMIT ? OFFSET ?
                """,
                (model_name, batch_size, offset),
            ).fetchall()
            
            if not rows:
                break
            
            # 批量计算相似度
            chunk_ids = []
            vectors = []
            metas = []
            
            for r in rows:
                vec = np.frombuffer(r[1], dtype=np.float32)
                if vec.shape[0] == qv.shape[0]:
                    chunk_ids.append(r[0])
                    vectors.append(vec)
                    metas.append({
                        "path": r[3],
                        "title": r[4],
                        "chunk_index": r[5],
                        "text": r[2],
                        "meta_json": r[6],
                        "context_text": r[7],
                    })
            
            if vectors:
                # 向量化计算
                mat = np.stack(vectors)
                # L2归一化
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                mat = mat / np.maximum(norms, 1e-12)
                # 计算余弦相似度
                sims = mat @ qv
                
                # 收集结果
                for i, sim in enumerate(sims):
                    all_results.append((chunk_ids[i], float(sim), metas[i]))
            
            if len(rows) < batch_size:
                break
            offset += batch_size
        
        # 取top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        top_results = all_results[:top_k]
        
        return [
            {
                "chunk_id": r[0],
                "score": -r[1],  # 与BM25保持一致（越小越好）
                **r[2],
            }
            for r in top_results
        ]
    
    def _compute_similarities(
        self,
        qv: np.ndarray,
        vector_data: list[tuple[int, bytes]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """计算相似度并返回top_k"""
        conn = self._ensure_connection()
        
        # 批量获取chunk信息
        chunk_ids = [v[0] for v in vector_data]
        
        if self._batch_retriever:
            chunk_infos = self._batch_retriever.batch_get_chunks(chunk_ids)
        else:
            placeholders = ",".join(["?"] * len(chunk_ids))
            rows = conn.execute(
                f"""
                SELECT c.id, d.path, COALESCE(d.title, '') AS title,
                       c.chunk_index, c.text_raw, COALESCE(c.meta_json, ''),
                       COALESCE(c.context_text, '')
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
            chunk_infos = [
                {
                    "chunk_id": r[0],
                    "path": r[1],
                    "title": r[2],
                    "chunk_index": r[3],
                    "text": r[4],
                    "meta_json": r[5],
                    "context_text": r[6],
                }
                for r in rows
            ]
        
        # 创建id到信息的映射
        info_map = {c["chunk_id"]: c for c in chunk_infos}
        
        # 计算相似度
        results = []
        for cid, blob in vector_data:
            if cid not in info_map:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != qv.shape[0]:
                continue
            # 归一化并计算
            vec_norm = vec / max(np.linalg.norm(vec), 1e-12)
            sim = float(vec_norm @ qv)
            info = info_map[cid]
            results.append({
                "chunk_id": cid,
                "score": -sim,
                **info,
            })
        
        # 排序取top_k
        results.sort(key=lambda x: x["score"])
        return results[:top_k]
    
    def _rerank_results(
        self,
        results: list[dict[str, Any]],
        bm25_scores: list[float],
        keywords: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """重排结果"""
        if not results:
            return []
        
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        span = bm25_max - bm25_min
        
        cover_w = max(0.0, float(os.environ.get("KB_RERANK_COVER_WEIGHT", "1.25")))
        title_w = max(0.0, float(os.environ.get("KB_RERANK_TITLE_WEIGHT", "0.35")))
        
        # 向量化计算覆盖率和标题匹配
        for i, r in enumerate(results):
            text = r["text"]
            title = r["title"]
            
            # 关键词覆盖率
            if keywords:
                hits = sum(1 for k in keywords if k in text)
                coverage = hits / len(keywords)
            else:
                coverage = 0.0
            
            # 标题匹配
            if title and keywords:
                title_hits = sum(1 for k in keywords if len(k) > 1 and k.lower() in title.lower())
                title_bonus = min(1.0, title_hits / max(2.0, len(keywords)))
            else:
                title_bonus = 0.0
            
            # 计算综合得分
            if span <= 1e-12:
                norm_bad = 0.5
            else:
                norm_bad = (r["score"] - bm25_min) / span
            
            r["_rr"] = (1.0 - norm_bad) + cover_w * coverage + title_w * title_bonus
        
        # 排序
        results.sort(key=lambda x: x.get("_rr", 0.0), reverse=True)
        
        # 清理临时字段
        for r in results:
            r.pop("_rr", None)
        
        return results
    
    def _hybrid_rrf_merge(
        self,
        fts_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """RRF融合FTS和向量结果"""
        rrf_k = max(10, min(120, int(os.environ.get("KB_RRF_K", "60"))))
        
        # 构建id到结果的映射
        id_to_row: dict[int, dict[str, Any]] = {}
        scores: dict[int, float] = {}
        
        # 添加FTS结果
        for i, r in enumerate(fts_results):
            cid = int(r["chunk_id"])
            id_to_row[cid] = r
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + i + 1)
        
        # 添加向量结果
        for i, r in enumerate(vector_results):
            cid = int(r["chunk_id"])
            if cid not in id_to_row:
                id_to_row[cid] = r
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + i + 1)
        
        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        
        return [id_to_row[cid] for cid in sorted_ids[:top_k]]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        library_root: Path | str | None = None,
        use_hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """
        高性能检索入口
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            library_root: 资料库根目录（用于过滤）
            use_hybrid: 是否使用混合检索
        
        Returns:
            检索结果列表
        """
        if not query.strip():
            return []
        
        q = query.strip()
        keywords = self._extract_keywords(q)
        fts_q = self._build_fts_query(q)
        
        # 并行执行FTS和向量检索
        fts_results: list[dict[str, Any]] = []
        vector_results: list[dict[str, Any]] = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交FTS任务
            fts_future = executor.submit(self._fts_search, fts_q, top_k * 2, keywords)
            
            # 提交向量任务（如果启用）
            vector_future = None
            if use_hybrid and self._vector_table_nonempty():
                vector_future = executor.submit(self._vector_search_fast, q, top_k * 2)
            
            # 获取结果
            try:
                fts_results = fts_future.result(timeout=30)
            except Exception:
                fts_results = []
            
            if vector_future:
                try:
                    vector_results = vector_future.result(timeout=30)
                except Exception:
                    vector_results = []
        
        # 融合结果
        if use_hybrid and vector_results:
            merged = self._hybrid_rrf_merge(fts_results, vector_results, top_k * 2)
        else:
            merged = fts_results
        
        # 应用资料库过滤
        if library_root is not None:
            merged = self._filter_by_library_root(merged, library_root)
        
        # 去重和截断
        return self._deduplicate_and_cap(merged, top_k)
    
    def _vector_table_nonempty(self) -> bool:
        """检查向量表是否有数据"""
        conn = self._ensure_connection()
        try:
            r = conn.execute("SELECT 1 FROM chunk_vectors LIMIT 1").fetchone()
            return r is not None
        except sqlite3.OperationalError:
            return False
    
    def _filter_by_library_root(
        self,
        results: list[dict[str, Any]],
        library_root: Path | str,
    ) -> list[dict[str, Any]]:
        """按资料库根目录过滤"""
        try:
            root = Path(library_root).expanduser().resolve()
        except OSError:
            return results
        
        filtered = []
        for r in results:
            path = str(r.get("path", ""))
            try:
                doc_path = Path(path).expanduser().resolve()
                doc_path.relative_to(root)
                filtered.append(r)
            except ValueError:
                # Windows大小写不敏感检查
                if os.name == "nt":
                    dr = os.path.normcase(str(doc_path))
                    rr = os.path.normcase(str(root)).rstrip("\\/")
                    if dr == rr or dr.startswith(rr + os.sep):
                        filtered.append(r)
        
        return filtered
    
    def _deduplicate_and_cap(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """去重并限制数量"""
        seen_prefix: set[str] = set()
        seen_ids: set[int] = set()
        out: list[dict[str, Any]] = []
        
        max_per_doc = max(1, min(8, int(os.environ.get("KB_MAX_CHUNKS_PER_DOC", "3"))))
        doc_counts: dict[str, int] = {}
        
        for r in results:
            cid = int(r.get("chunk_id", 0))
            if cid in seen_ids:
                continue
            
            # 前缀去重
            text = str(r.get("text", ""))
            prefix = text[:160].strip()
            if prefix in seen_prefix:
                continue
            
            # 每文档限制
            path = str(r.get("path", ""))
            if doc_counts.get(path, 0) >= max_per_doc:
                continue
            
            seen_ids.add(cid)
            seen_prefix.add(prefix)
            doc_counts[path] = doc_counts.get(path, 0) + 1
            out.append(r)
            
            if len(out) >= top_k:
                break
        
        return out
    
    def close(self) -> None:
        """关闭资源"""
        if not self._external_conn and self.conn:
            self.conn.close()
            self.conn = None


def fast_retrieve(
    query: str,
    db_path: str,
    top_k: int = 4,
    library_root: Path | str | None = None,
    build_index: bool = False,
) -> list[dict[str, Any]]:
    """
    快速检索便捷函数
    
    Args:
        query: 查询文本
        db_path: 数据库路径
        top_k: 返回结果数量
        library_root: 资料库根目录
        build_index: 是否构建向量索引
    
    Returns:
        检索结果列表
    """
    retriever = FastRetriever(db_path=db_path)
    
    try:
        if build_index:
            retriever.build_vector_index()
        
        return retriever.retrieve(query, top_k, library_root)
    finally:
        retriever.close()
