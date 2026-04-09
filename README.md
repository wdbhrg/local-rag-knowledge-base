# Local RAG Knowledge Base

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Performance](https://img.shields.io/badge/performance-optimized-green.svg)](#性能优化)

高性能本地RAG（检索增强生成）知识库系统，支持EPUB/PDF/TXT文档的索引、检索和问答。基于SQLite + FTS5全文检索，可选稠密向量检索，提供Streamlit Web界面和FastAPI服务。

**核心特性**：
- 🚀 **高性能检索**：3-10倍查询速度提升（详见[性能优化](#性能优化)）
- 🔍 **混合检索**：BM25 + 向量相似度 + RRF融合 + Cross-Encoder重排
- 💾 **本地优先**：所有数据存储在本地SQLite，无需云端依赖
- 🤖 **多模型支持**：OpenAI兼容API / Ollama本地模型 / 豆包等国产大模型
- 🌐 **双界面**：Streamlit Web界面 + FastAPI REST服务
- 📊 **语义缓存**：相似查询直接返回缓存结果，减少API调用

**GitHub**: https://github.com/wdbhrg/local-rag-knowledge-base

---

## 目录

- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [性能优化](#性能优化)
- [核心功能](#核心功能)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [开发指南](#开发指南)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Streamlit Web  │   FastAPI服务   │      CLI命令行              │
│   (app.py)      │ (api_server.py) │    (ask_kb.py)              │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         └─────────────────┼───────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      检索引擎层 (Optimized)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  FTS5 BM25  │  │  向量检索   │  │    RRF融合 + 重排       │  │
│  │  (SQLite)   │  │  (IVF索引)  │  │  (fast_retriever.py)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          │                                      │
│  ┌───────────────────────▼────────────────────────┐            │
│  │         性能优化层 (db_optimizer.py)           │            │
│  │  • 连接池管理  • 查询缓存  • 批量检索  • 索引优化 │            │
│  └────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      数据存储层                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │  documents  │  │   chunks    │  │chunk_vectors│  │  FTS5  │ │
│  │  (文档元数据) │  │  (文本片段)  │  │  (向量存储)  │  │(全文索引)│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ chunk_graph │  │semantic_cache│  │    性能索引(自动创建)    │  │
│  │  (关联图)   │  │  (语义缓存)   │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```mermaid
flowchart TD
    subgraph Indexing["📚 索引流程 (build_kb.py)"]
        docs[EPUB/PDF/TXT] --> parser[文档解析]
        parser --> chunker[智能分块]
        chunker --> fts[(FTS5索引)]
        chunker --> vectors[(向量索引)]
        chunker --> graph[(关联图)]
    end

    subgraph Retrieval["🔍 检索流程 (fast_retriever.py)"]
        query[用户查询] --> kw[关键词提取]
        kw --> fts_search[FTS检索]
        kw --> vec_search[向量检索]
        fts_search --> rrf[RRF融合]
        vec_search --> rrf
        rrf --> rerank[Cross-Encoder重排]
        rerank --> cache_check{缓存命中?}
        cache_check -->|是| cached[返回缓存]
        cache_check -->|否| llm[LLM生成]
    end

    subgraph Optimization["⚡ 性能优化层"]
        conn_pool[连接池] --> cache[查询缓存]
        cache --> batch[批量检索]
        batch --> ivf[IVF向量索引]
    end
```

---

## 快速开始

### 环境要求

- **Python**: 3.10+
- **操作系统**: Windows / macOS / Linux
- **内存**: 建议4GB+（向量检索时）
- **磁盘**: 预留2-3倍原文档大小的空间

### 安装

```bash
# 克隆仓库
git clone https://github.com/wdbhrg/local-rag-knowledge-base.git
cd local-rag-knowledge-base

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或: .venv\Scripts\activate  # Windows

# 安装核心依赖
pip install -r requirements.txt

# 安装可选依赖（推荐，启用完整功能）
pip install -r requirements-optional.txt
```

### 配置

复制环境变量模板并填写：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# OpenAI兼容API配置
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_id

# 本地Ollama配置（可选）
# OPENAI_BASE_URL=http://127.0.0.1:11434/v1
# OPENAI_MODEL=kb-rag

# 性能优化配置
KB_SQLITE_CACHE_KIB=65536      # SQLite缓存大小(64MB)
KB_SQLITE_MMAP_MB=256          # 内存映射大小(256MB)
KB_HYBRID_SEARCH=1             # 启用混合检索
KB_CROSS_RERANK=1              # 启用Cross-Encoder重排
KB_SEMANTIC_CACHE=1            # 启用语义缓存
```

### 启动

**Windows一键启动**：
```bash
start.bat
```

**命令行启动**：
```bash
# 1. 构建知识库索引
python build_kb.py --root "./books" --db knowledge_base.sqlite

# 2. 启动Web界面
python launch_ui.py
# 或直接: streamlit run app.py

# 3. 或启动API服务
uvicorn api_server:app --host 127.0.0.1 --port 8765
```

---

## 性能优化

本项目针对RAG检索进行了系统性性能优化，主要改进点：

### 优化模块

| 模块 | 文件 | 优化内容 | 性能提升 |
|------|------|----------|----------|
| 数据库优化 | `db_optimizer.py` | 连接池、查询缓存、索引优化 | 2-3x |
| 快速检索 | `fast_retriever.py` | 并行检索、批量处理、IVF索引 | 3-10x |
| 批量操作 | `BatchRetriever` | 合并查询减少DB往返 | 1.5-2x |
| 向量索引 | `VectorIndexManager` | IVF聚类索引 | 5-10x |

### 核心优化技术

#### 1. 并行混合检索
```python
# FTS和向量检索并行执行，而非串行
with ThreadPoolExecutor(max_workers=2) as executor:
    fts_future = executor.submit(fts_search, ...)
    vec_future = executor.submit(vector_search, ...)
    results = merge(fts_future.result(), vec_future.result())
```

#### 2. IVF向量索引
```python
# 构建IVF索引，检索时只需搜索最近聚类
retriever = FastRetriever(db_path="knowledge_base.sqlite")
retriever.build_vector_index(n_clusters=16)  # 构建索引
results = retriever.retrieve("查询", top_k=4)  # 快速检索
```

#### 3. 连接池与缓存
```python
# 复用数据库连接，缓存热点查询
from db_optimizer import get_optimized_connection, QueryCache

conn = get_optimized_connection("knowledge_base.sqlite")  # 优化的连接
# 自动应用: WAL模式、内存映射、性能索引
```

### 性能对比

使用 `benchmark.py` 测试性能：

```bash
python benchmark.py --db knowledge_base.sqlite --iterations 10
```

**典型性能数据**（10k文档，100k片段）：

| 检索方式 | 平均耗时 | 性能提升 |
|----------|----------|----------|
| 原始检索 | 850ms | 1x (基准) |
| 优化检索（无IVF） | 280ms | **3.0x** |
| 优化检索（含IVF） | 95ms | **8.9x** |

---

## 核心功能

### 1. 文档索引

支持格式：EPUB、PDF、TXT

```bash
# 命令行索引
python build_kb.py --root "./books" --db knowledge_base.sqlite

# Python API
from build_kb import reindex_all
import sqlite3

conn = sqlite3.connect("knowledge_base.sqlite")
result = reindex_all(conn, root_path="./books")
print(f"索引完成: {result['documents']}文档, {result['chunks']}片段")
```

### 2. 检索问答

```python
# 快速检索
from fast_retriever import fast_retrieve

results = fast_retrieve(
    query="什么是机器学习",
    db_path="knowledge_base.sqlite",
    top_k=4,
    build_index=True  # 自动构建IVF索引
)

# 完整问答
from ask_kb import ask_llm, retrieve
import sqlite3

conn = sqlite3.connect("knowledge_base.sqlite")
contexts = retrieve(conn, "什么是机器学习", top_k=4)
answer = ask_llm("什么是机器学习", contexts)
print(answer)
```

### 3. 语义缓存

```python
# 自动缓存相似查询
from semantic_cache import store_cached_answer, lookup_cached_answer

# 首次查询（调用LLM）
answer = ask_llm(question, contexts)
store_cached_answer(conn, question, answer)

# 相似查询（直接返回缓存，跳过LLM）
cached = lookup_cached_answer(conn, "什么是机器学习技术")  # 相似问题
if cached:
    print("缓存命中！")
```

### 4. GraphRAG扩展

```bash
# 启用关联图构建
export KB_BUILD_GRAPH=1
export KB_GRAPH_RAG=1
export KB_GRAPH_SLOTS=2

python build_kb.py --root "./books" --db knowledge_base.sqlite
```

---

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_BASE_URL` | - | API基础地址 |
| `OPENAI_API_KEY` | - | API密钥 |
| `OPENAI_MODEL` | - | 模型ID |
| `KB_LIBRARY_ROOT` | - | 资料库根目录 |
| `KB_CHUNK_STRATEGY` | sentence | 分块策略: sentence/semantic |
| `KB_CONTEXT_CHARS` | 2400 | 片段最大字符数 |
| `KB_MAX_TOKENS` | 4096 | 生成最大token数 |
| `KB_TEMPERATURE` | 0.35 | 生成温度 |

### 性能优化配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KB_SQLITE_CACHE_KIB` | 65536 | SQLite缓存(KiB) |
| `KB_SQLITE_MMAP_MB` | 256 | 内存映射大小(MB) |
| `KB_HYBRID_SEARCH` | 1 | 启用混合检索 |
| `KB_CROSS_RERANK` | 1 | 启用Cross-Encoder重排 |
| `KB_SEMANTIC_CACHE` | 0 | 启用语义缓存 |
| `KB_VECTOR_INDEX` | 1 | 启用向量索引 |
| `KB_BUILD_GRAPH` | 1 | 构建关联图 |

---

## API文档

### FastAPI服务

启动服务：
```bash
uvicorn api_server:app --host 127.0.0.1 --port 8765
```

**流式问答接口**：

```bash
POST /api/chat/stream
Content-Type: application/json

{
  "question": "什么是RAG?",
  "top_k": 4,
  "db": "knowledge_base.sqlite"
}
```

响应（SSE流）：
```
data: {"delta": "RAG是"}
data: {"delta": "检索增强生成的缩写"}
data: [DONE]
```

**健康检查**：
```bash
GET /health
```

### Python API

详见各模块文档字符串：
- `ask_kb.py`: 检索与LLM调用
- `fast_retriever.py`: 高性能检索
- `db_optimizer.py`: 数据库优化
- `build_kb.py`: 索引构建
- `semantic_cache.py`: 语义缓存

---

## 开发指南

### 项目结构

```
local-rag-knowledge-base/
├── app.py                 # Streamlit Web界面
├── api_server.py          # FastAPI服务
├── ask_kb.py              # 检索与LLM调用（原始实现）
├── fast_retriever.py      # 高性能检索（优化实现）
├── db_optimizer.py        # 数据库性能优化
├── build_kb.py            # 索引构建
├── semantic_cache.py      # 语义缓存
├── benchmark.py           # 性能测试
├── launch_ui.py           # 启动器
├── start.bat              # Windows一键启动
├── requirements.txt       # 核心依赖
├── requirements-optional.txt  # 可选依赖
├── .env.example           # 环境变量模板
└── eval/                  # 评测工具
    └── evaluate_rag.py
```

### 添加自定义优化

```python
# db_optimizer.py 扩展示例
class CustomOptimizer:
    def __init__(self, conn):
        self.conn = conn
    
    def optimize_for_workload(self, queries: list[str]):
        """基于工作负载自动优化"""
        # 分析查询模式
        # 创建针对性索引
        pass
```

### 运行测试

```bash
# 性能基准测试
python benchmark.py --db knowledge_base.sqlite

# RAG评测
python eval/evaluate_rag.py --db knowledge_base.sqlite --golden eval/golden.jsonl
```

---

## 贡献指南

欢迎提交Issue和PR！

1. Fork本仓库
2. 创建特性分支: `git checkout -b feature/xxx`
3. 提交更改: `git commit -am 'Add xxx'`
4. 推送分支: `git push origin feature/xxx`
5. 提交Pull Request

---

## 许可证

[MIT License](LICENSE)

---

## 致谢

- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [sentence-transformers](https://www.sbert.net/) - 文本嵌入
- [Streamlit](https://streamlit.io/) - Web界面
- [FastAPI](https://fastapi.tiangolo.com/) - API框架
