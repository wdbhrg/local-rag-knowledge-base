"""
================================================================================
db_optimizer.py —— 数据库查询性能优化模块
================================================================================

提供数据库索引优化、查询缓存、连接池管理等功能，系统性提升RAG检索速度。

核心优化点：
1. 智能索引管理 - 自动创建/优化数据库索引
2. 向量检索加速 - IVF索引、向量量化、批量相似度计算
3. 查询结果缓存 - LRU缓存热点查询
4. 连接池管理 - 复用数据库连接
5. 异步预加载 - 后台预加载热点数据

使用方法：
    from db_optimizer import get_optimized_connection, VectorIndexManager
    conn = get_optimized_connection("knowledge_base.sqlite")
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

# 连接池 - 线程安全
_connection_pool: dict[str, sqlite3.Connection] = {}
_pool_lock = threading.RLock()

# 查询缓存
_query_cache: OrderedDict[str, Any] = OrderedDict()
_cache_lock = threading.Lock()
_cache_max_size = 128
_cache_ttl_seconds = 300  # 5分钟过期


def _get_cache_key(query: str, params: tuple) -> str:
    """生成查询缓存键"""
    key_data = f"{query}:{json.dumps(params, sort_keys=True, default=str)}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _is_cache_valid(timestamp: float) -> bool:
    """检查缓存是否过期"""
    return time.time() - timestamp < _cache_ttl_seconds


class QueryCache:
    """线程安全的LRU查询缓存"""
    
    def __init__(self, max_size: int = 128, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Any | None:
        """获取缓存值，自动处理过期"""
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if not self._is_valid(timestamp):
                del self._cache[key]
                return None
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            # 淘汰最旧的
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def invalidate_pattern(self, pattern: str) -> int:
        """根据模式使缓存失效"""
        with self._lock:
            keys_to_remove = [k for k in self._cache if pattern in k]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)
    
    def _is_valid(self, timestamp: float) -> bool:
        return time.time() - timestamp < self.ttl
    
    @property
    def stats(self) -> dict[str, int]:
        """缓存统计"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
            }


# 全局查询缓存实例
_query_cache_instance = QueryCache()


def cached_query(ttl: int = 300):
    """查询缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hashlib.sha256(str(args).encode()).hexdigest()}"
            
            # 尝试从缓存获取
            cached = _query_cache_instance.get(cache_key)
            if cached is not None:
                return cached
            
            # 执行查询
            result = func(*args, **kwargs)
            
            # 缓存结果
            _query_cache_instance.set(cache_key, result)
            return result
        return wrapper
    return decorator


class VectorIndexManager:
    """向量索引管理器 - 优化向量检索性能"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ivf_index: dict[str, Any] = {}
        self._index_built = False
        self._lock = threading.Lock()
    
    def build_ivf_index(self, n_clusters: int = 16) -> bool:
        """
        构建IVF（倒排文件）索引加速向量检索
        将向量空间划分为n_clusters个聚类，检索时只需搜索最近聚类
        """
        try:
            import numpy as np
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return False
        
        with self._lock:
            if self._index_built:
                return True
            
            # 获取所有向量
            rows = self.conn.execute(
                "SELECT chunk_id, vec, model FROM chunk_vectors"
            ).fetchall()
            
            if len(rows) < n_clusters * 2:
                return False
            
            # 按模型分组
            by_model: dict[str, list[tuple[int, np.ndarray]]] = {}
            for cid, blob, model in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                by_model.setdefault(model, []).append((cid, vec))
            
            # 为每个模型构建IVF索引
            for model, vectors in by_model.items():
                if len(vectors) < n_clusters * 2:
                    continue
                
                ids = [v[0] for v in vectors]
                vecs = np.stack([v[1] for v in vectors])
                
                # L2归一化
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.maximum(norms, 1e-12)
                
                # K-Means聚类
                kmeans = MiniBatchKMeans(
                    n_clusters=min(n_clusters, len(vectors) // 2),
                    batch_size=256,
                    max_iter=100,
                    random_state=42,
                )
                labels = kmeans.fit_predict(vecs)
                
                # 构建倒排索引
                inverted_index: dict[int, list[int]] = {}
                for idx, label in enumerate(labels):
                    inverted_index.setdefault(int(label), []).append(ids[idx])
                
                self._ivf_index[model] = {
                    "centroids": kmeans.cluster_centers_,
                    "inverted": inverted_index,
                    "kmeans": kmeans,
                }
            
            self._index_built = True
            return True
    
    def search_with_ivf(
        self,
        query_vec: np.ndarray,
        model: str,
        top_k: int = 100,
        n_probe: int = 3,
    ) -> list[tuple[int, float]]:
        """
        使用IVF索引搜索最近邻
        n_probe: 搜索的聚类数量，越大越精确但越慢
        """
        if model not in self._ivf_index:
            return []
        
        import numpy as np
        
        idx_info = self._ivf_index[model]
        kmeans = idx_info["kmeans"]
        inverted = idx_info["inverted"]
        
        # 找到最近的n_probe个聚类
        query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-12)
        distances = np.linalg.norm(idx_info["centroids"] - query_norm, axis=1)
        nearest_clusters = np.argsort(distances)[:n_probe]
        
        # 收集候选向量ID
        candidate_ids: set[int] = set()
        for cluster_id in nearest_clusters:
            candidate_ids.update(inverted.get(int(cluster_id), []))
        
        return list(candidate_ids)


class ConnectionPool:
    """SQLite连接池 - 复用连接减少开销"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self._pool: OrderedDict[str, sqlite3.Connection] = OrderedDict()
        self._lock = threading.RLock()
        self._in_use: set[str] = set()
    
    def get_connection(self, db_path: str) -> sqlite3.Connection:
        """获取连接，优先复用"""
        with self._lock:
            normalized_path = str(Path(db_path).resolve())
            
            # 查找可用连接
            for path, conn in list(self._pool.items()):
                if path == normalized_path and path not in self._in_use:
                    self._in_use.add(path)
                    self._pool.move_to_end(path)
                    return conn
            
            # 创建新连接
            conn = self._create_optimized_connection(normalized_path)
            
            # 如果池已满，关闭最旧的未使用连接
            while len(self._pool) >= self.max_connections:
                # 找到最旧的未使用连接
                for old_path in list(self._pool.keys()):
                    if old_path not in self._in_use:
                        old_conn = self._pool.pop(old_path)
                        old_conn.close()
                        break
                else:
                    # 所有连接都在使用中，关闭最旧的
                    old_path, old_conn = self._pool.popitem(last=False)
                    old_conn.close()
            
            self._pool[normalized_path] = conn
            self._in_use.add(normalized_path)
            return conn
    
    def release_connection(self, db_path: str) -> None:
        """释放连接回池"""
        with self._lock:
            normalized_path = str(Path(db_path).resolve())
            self._in_use.discard(normalized_path)
    
    def _create_optimized_connection(self, db_path: str) -> sqlite3.Connection:
        """创建优化的数据库连接"""
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        conn.text_factory = str
        
        # 性能优化PRAGMA
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB缓存
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB内存映射
        conn.execute("PRAGMA busy_timeout=10000")
        
        return conn
    
    def close_all(self) -> None:
        """关闭所有连接"""
        with self._lock:
            for conn in self._pool.values():
                conn.close()
            self._pool.clear()
            self._in_use.clear()


# 全局连接池
_connection_pool_instance = ConnectionPool()


@contextmanager
def pooled_connection(db_path: str):
    """连接池上下文管理器"""
    conn = _connection_pool_instance.get_connection(db_path)
    try:
        yield conn
    finally:
        _connection_pool_instance.release_connection(db_path)


def ensure_performance_indexes(conn: sqlite3.Connection) -> None:
    """确保性能关键索引存在"""
    indexes = [
        # chunks表索引
        ("idx_chunks_doc_id", "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)"),
        ("idx_chunks_chunk_index", "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)"),
        
        # chunk_vectors表索引
        ("idx_chunk_vectors_model", "CREATE INDEX IF NOT EXISTS idx_chunk_vectors_model ON chunk_vectors(model)"),
        ("idx_chunk_vectors_dim", "CREATE INDEX IF NOT EXISTS idx_chunk_vectors_dim ON chunk_vectors(dim)"),
        
        # chunk_graph_edges表索引
        ("idx_graph_edges_src", "CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON chunk_graph_edges(src_chunk_id)"),
        ("idx_graph_edges_dst", "CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON chunk_graph_edges(dst_chunk_id)"),
        ("idx_graph_edges_weight", "CREATE INDEX IF NOT EXISTS idx_graph_edges_weight ON chunk_graph_edges(weight DESC)"),
        
        # semantic_query_cache表索引
        ("idx_sem_cache_hash", "CREATE INDEX IF NOT EXISTS idx_sem_cache_hash ON semantic_query_cache(model, query_hash)"),
        ("idx_sem_cache_ts", "CREATE INDEX IF NOT EXISTS idx_sem_cache_ts ON semantic_query_cache(created_ts DESC)"),
        
        # documents表索引
        ("idx_documents_path", "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)"),
    ]
    
    for name, sql in indexes:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # 索引可能已存在
    
    conn.commit()


def get_optimized_connection(db_path: str) -> sqlite3.Connection:
    """
    获取优化的数据库连接
    自动应用性能优化设置和索引
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.text_factory = str
    
    # 基础性能设置
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    # 缓存设置（从环境变量读取或默认值）
    try:
        cache_kib = int(os.environ.get("KB_SQLITE_CACHE_KIB", "65536"))
        cache_kib = max(2048, min(2_097_152, cache_kib))
    except ValueError:
        cache_kib = 65536
    conn.execute(f"PRAGMA cache_size={-cache_kib}")
    
    # 内存映射
    try:
        mmap_mb = int(os.environ.get("KB_SQLITE_MMAP_MB", "256"))
        mmap_mb = max(0, min(2048, mmap_mb))
        if mmap_mb > 0:
            conn.execute(f"PRAGMA mmap_size={mmap_mb * 1024 * 1024}")
    except Exception:
        pass
    
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA busy_timeout=10000")
    
    # 确保索引存在
    ensure_performance_indexes(conn)
    
    return conn


def analyze_query_performance(conn: sqlite3.Connection, query: str) -> dict[str, Any]:
    """分析查询执行计划"""
    try:
        plan = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        return {
            "plan": plan,
            "detail": conn.execute(f"EXPLAIN {query}").fetchall() if hasattr(conn, 'execute') else None,
        }
    except Exception as e:
        return {"error": str(e)}


def vacuum_database(conn: sqlite3.Connection) -> None:
    """整理数据库，回收空间并优化"""
    conn.execute("VACUUM")
    conn.execute("ANALYZE")
    conn.commit()


class BatchRetriever:
    """批量检索器 - 合并多个查询减少数据库往返"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def batch_get_chunks(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        """批量获取chunk信息"""
        if not chunk_ids:
            return []
        
        placeholders = ",".join(["?"] * len(chunk_ids))
        query = f"""
            SELECT c.id, d.path, COALESCE(d.title, '') AS title, 
                   c.chunk_index, c.text_raw, COALESCE(c.meta_json, ''),
                   COALESCE(c.context_text, '')
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders})
        """
        
        rows = self.conn.execute(query, chunk_ids).fetchall()
        
        # 保持原始顺序
        order_map = {cid: i for i, cid in enumerate(chunk_ids)}
        results = []
        for r in rows:
            results.append({
                "chunk_id": r[0],
                "path": r[1],
                "title": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "meta_json": r[5],
                "context_text": r[6],
            })
        
        results.sort(key=lambda x: order_map.get(x["chunk_id"], 999999))
        return results
    
    def batch_get_vectors(self, chunk_ids: list[int], model: str) -> list[tuple[int, bytes]]:
        """批量获取向量"""
        if not chunk_ids:
            return []
        
        placeholders = ",".join(["?"] * len(chunk_ids))
        query = f"""
            SELECT chunk_id, vec FROM chunk_vectors
            WHERE chunk_id IN ({placeholders}) AND model = ?
        """
        
        rows = self.conn.execute(query, chunk_ids + [model]).fetchall()
        return [(r[0], r[1]) for r in rows]


def clear_all_caches() -> None:
    """清除所有缓存"""
    _query_cache_instance.clear()


def get_optimization_stats() -> dict[str, Any]:
    """获取优化统计信息"""
    return {
        "query_cache": _query_cache_instance.stats,
        "connection_pool": {
            "size": len(_connection_pool_instance._pool),
            "in_use": len(_connection_pool_instance._in_use),
            "max": _connection_pool_instance.max_connections,
        },
    }
