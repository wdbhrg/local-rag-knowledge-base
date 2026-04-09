"""
================================================================================
benchmark.py —— RAG检索性能基准测试
================================================================================

对比原始检索与优化后的检索性能，验证优化效果。

使用方法：
    python benchmark.py --db knowledge_base.sqlite --queries "问题1" "问题2"

输出：
    - 各检索方式的耗时对比
    - 结果质量对比（重合度）
    - 性能提升百分比
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


def benchmark_original_retrieve(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 4,
    iterations: int = 5,
) -> dict[str, Any]:
    """测试原始检索性能"""
    from ask_kb import retrieve
    
    times = []
    results = None
    
    for _ in range(iterations):
        start = time.perf_counter()
        results = retrieve(conn, query, top_k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "times": times,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "results": results,
    }


def benchmark_optimized_retrieve(
    db_path: str,
    query: str,
    top_k: int = 4,
    iterations: int = 5,
    build_index: bool = False,
) -> dict[str, Any]:
    """测试优化后的检索性能"""
    from fast_retriever import FastRetriever
    
    times = []
    results = None
    
    retriever = FastRetriever(db_path=db_path)
    
    # 可选：构建向量索引
    if build_index:
        print("  Building vector index...")
        retriever.build_vector_index(n_clusters=16)
    
    for i in range(iterations):
        start = time.perf_counter()
        results = retriever.retrieve(query, top_k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        # 第一次迭代后重置连接缓存
        if i == 0:
            retriever.close()
            retriever = FastRetriever(db_path=db_path)
    
    retriever.close()
    
    return {
        "times": times,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "results": results,
    }


def calculate_overlap(results1: list[dict], results2: list[dict]) -> float:
    """计算两组结果的chunk_id重合度"""
    ids1 = {r["chunk_id"] for r in results1}
    ids2 = {r["chunk_id"] for r in results2}
    
    if not ids1 or not ids2:
        return 0.0
    
    intersection = len(ids1 & ids2)
    union = len(ids1 | ids2)
    
    return intersection / union if union > 0 else 0.0


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 0.001:
        return f"{seconds*1000:.2f}µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def run_benchmark(
    db_path: str,
    queries: list[str],
    top_k: int = 4,
    iterations: int = 5,
    skip_original: bool = False,
) -> None:
    """运行完整基准测试"""
    
    print("=" * 70)
    print("RAG检索性能基准测试")
    print("=" * 70)
    print(f"数据库: {db_path}")
    print(f"查询数量: {len(queries)}")
    print(f"每查询迭代次数: {iterations}")
    print(f"Top-K: {top_k}")
    print("=" * 70)
    
    # 检查数据库
    if not Path(db_path).exists():
        print(f"错误: 数据库不存在: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.text_factory = str
    
    # 获取统计信息
    try:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vector_count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        print(f"\n数据库统计:")
        print(f"  文档数: {doc_count}")
        print(f"  片段数: {chunk_count}")
        print(f"  向量数: {vector_count}")
    except Exception as e:
        print(f"  获取统计信息失败: {e}")
    
    print()
    
    total_original = 0
    total_optimized = 0
    total_optimized_indexed = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\n查询 {i}/{len(queries)}: \"{query}\"")
        print("-" * 70)
        
        # 测试原始检索
        if not skip_original:
            print("  测试原始检索...")
            original_stats = benchmark_original_retrieve(
                conn, query, top_k, iterations
            )
            total_original += original_stats["mean"]
            print(f"    平均耗时: {format_time(original_stats['mean'])}")
            print(f"    标准差: {format_time(original_stats['std'])}")
            print(f"    范围: {format_time(original_stats['min'])} - {format_time(original_stats['max'])}")
        else:
            original_stats = None
            print("  跳过原始检索测试")
        
        # 测试优化检索（无索引）
        print("  测试优化检索（无IVF索引）...")
        optimized_stats = benchmark_optimized_retrieve(
            db_path, query, top_k, iterations, build_index=False
        )
        total_optimized += optimized_stats["mean"]
        print(f"    平均耗时: {format_time(optimized_stats['mean'])}")
        print(f"    标准差: {format_time(optimized_stats['std'])}")
        print(f"    范围: {format_time(optimized_stats['min'])} - {format_time(optimized_stats['max'])}")
        
        # 测试优化检索（有索引）
        if vector_count > 0:
            print("  测试优化检索（有IVF索引）...")
            optimized_indexed_stats = benchmark_optimized_retrieve(
                db_path, query, top_k, iterations, build_index=True
            )
            total_optimized_indexed += optimized_indexed_stats["mean"]
            print(f"    平均耗时: {format_time(optimized_indexed_stats['mean'])}")
            print(f"    标准差: {format_time(optimized_indexed_stats['std'])}")
            print(f"    范围: {format_time(optimized_indexed_stats['min'])} - {format_time(optimized_indexed_stats['max'])}")
        else:
            optimized_indexed_stats = None
            print("  跳过IVF索引测试（无向量数据）")
        
        # 结果质量对比
        if original_stats and optimized_stats:
            overlap = calculate_overlap(
                original_stats["results"],
                optimized_stats["results"]
            )
            print(f"\n  结果重合度: {overlap:.1%}")
        
        # 性能提升
        if original_stats:
            speedup = original_stats["mean"] / optimized_stats["mean"]
            print(f"  性能提升: {speedup:.2f}x")
            
            if optimized_indexed_stats:
                speedup_indexed = original_stats["mean"] / optimized_indexed_stats["mean"]
                print(f"  性能提升（含IVF）: {speedup_indexed:.2f}x")
    
    conn.close()
    
    # 汇总
    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    
    if not skip_original:
        print(f"原始检索总平均耗时: {format_time(total_original / len(queries))}")
        print(f"优化检索总平均耗时: {format_time(total_optimized / len(queries))}")
        overall_speedup = total_original / total_optimized
        print(f"整体性能提升: {overall_speedup:.2f}x")
        
        if vector_count > 0:
            print(f"优化检索（含IVF）总平均耗时: {format_time(total_optimized_indexed / len(queries))}")
            overall_speedup_indexed = total_original / total_optimized_indexed
            print(f"整体性能提升（含IVF）: {overall_speedup_indexed:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="RAG检索性能基准测试")
    parser.add_argument("--db", default="knowledge_base.sqlite", help="数据库路径")
    parser.add_argument("--queries", nargs="+", default=[
        "什么是RAG",
        "如何优化数据库查询性能",
        "机器学习方法",
    ], help="测试查询列表")
    parser.add_argument("--top-k", type=int, default=4, help="返回结果数量")
    parser.add_argument("--iterations", type=int, default=5, help="每查询迭代次数")
    parser.add_argument("--skip-original", action="store_true", help="跳过原始检索测试")
    
    args = parser.parse_args()
    
    run_benchmark(
        db_path=args.db,
        queries=args.queries,
        top_k=args.top_k,
        iterations=args.iterations,
        skip_original=args.skip_original,
    )


if __name__ == "__main__":
    main()
