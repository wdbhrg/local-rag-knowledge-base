"""

RAG 评测：基于自建「金标准」JSONL 的客观检索指标 + 可选 RAGAS 全链路分数。



1) 检索（不依赖 LLM）

   - Hit@k：top-k 检索结果中是否命中任一 relevant_snippets（子串匹配，空白规范化后）。

   - MRR：首个含金片段的文档在排序中的倒数排名（未命中为 0）。

   - Precision@k / Recall@k：检索侧微观指标——含任一金片段的块占比；被至少一块覆盖的金片段占比。



2) RAGAS（需 pip install -r requirements-optional.txt，并配置 OPENAI_*）

   默认同时计算：

   - faithfulness：回答是否被检索上下文支撑；

   - context_recall：标准答案中的陈述有多少可由检索到的上下文覆盖（RAGAS LLMContextRecall）；

   - answer_relevancy：回答与问题的相关度（需兼容 OpenAI 的 Embeddings API）。



用法示例：

  .venv\\Scripts\\python eval/evaluate_rag.py --db knowledge_base.sqlite --golden eval/golden.jsonl --top-k 4

  .venv\\Scripts\\python eval/evaluate_rag.py --db knowledge_base.sqlite --golden eval/golden.jsonl --ragas

  .venv\\Scripts\\python eval/evaluate_rag.py ... --ragas --json-out eval/last_scores.json \\

      --min-faithfulness 0.5 --min-context-recall 0.4 --min-answer-relevancy 0.5



Arize Phoenix：本脚本直接输出量化分数；若需链路追踪与实验看板，可将 JSON 结果导入你自建的流水线，

或自行接入 Phoenix / OpenTelemetry（见 README）。

"""

from __future__ import annotations



import argparse

import json

import math

import os

import re

import sqlite3

import sys

from pathlib import Path

from statistics import mean

from typing import Any



ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



from ask_kb import ask_llm, retrieve  # noqa: E402

from build_kb import effective_library_root  # noqa: E402





def _norm(s: str) -> str:

    s = s.replace("\u3000", " ").strip().lower()

    s = re.sub(r"\s+", "", s)

    return s





def _hit_and_mrr(contexts: list[dict], snippets: list[str]) -> tuple[float, float]:

    if not snippets:

        return 1.0, 1.0

    norms = [_norm(s) for s in snippets if s.strip()]

    if not norms:

        return 1.0, 1.0

    texts = [_norm(c.get("text") or "") for c in contexts]

    for i, t in enumerate(texts):

        if any(ns and ns in t for ns in norms):

            return 1.0, 1.0 / float(i + 1)

    return 0.0, 0.0


def _precision_recall_at_k(contexts: list[dict], snippets: list[str]) -> tuple[float, float]:
    """
    基于子串金标准的检索 P/R（非 IR 集合论意义下的纯集合，但与 RAG 金标注常用写法一致）：
    - Precision@k：top-k 块中，正文含至少一条金片段（规范化子串）的块数 / k。
    - Recall@k：金片段中，被至少一块覆盖的比例。
    """
    if not snippets:
        return 1.0, 1.0
    norms = [_norm(s) for s in snippets if s.strip()]
    if not norms:
        return 1.0, 1.0
    texts = [_norm(c.get("text") or "") for c in contexts]
    k = len(contexts)
    if k == 0:
        return 0.0, 0.0
    rel_chunks = sum(1 for t in texts if any(ns and ns in t for ns in norms))
    precision = rel_chunks / float(k)
    matched = sum(1 for ns in norms if ns and any(ns in t for t in texts))
    recall = matched / float(len(norms))
    return precision, recall


def load_golden(path: Path) -> list[dict]:

    rows = []

    raw = path.read_text(encoding="utf-8")

    for line in raw.splitlines():

        line = line.strip()

        if not line or line.startswith("#"):

            continue

        rows.append(json.loads(line))

    return rows





def run_retrieval_eval(

    conn: sqlite3.Connection,

    items: list[dict],

    top_k: int,

    library_root: Path | None,

) -> dict[str, float]:

    hits: list[float] = []

    mrrs: list[float] = []

    precs: list[float] = []

    recalls: list[float] = []

    for it in items:

        q = (it.get("question") or "").strip()

        if not q:

            continue

        snips = it.get("relevant_snippets") or it.get("must_contain") or []

        if isinstance(snips, str):

            snips = [snips]

        ctxs = retrieve(conn, q, top_k, library_root=library_root)

        h, m = _hit_and_mrr(ctxs, snips)

        hits.append(h)

        mrrs.append(m)

        p, r = _precision_recall_at_k(ctxs, snips)

        precs.append(p)

        recalls.append(r)

    n = len(hits) or 1

    return {
        "hit_at_k": mean(hits),
        "mrr": mean(mrrs),
        "precision_at_k": mean(precs),
        "recall_at_k": mean(recalls),
        "n": float(len(hits)),
    }





def _eval_embedding_model() -> str:

    return (

        (os.environ.get("KB_EVAL_EMBEDDING_MODEL") or os.environ.get("OPENAI_EMBEDDING_MODEL") or "")

        .strip()

        or "text-embedding-3-small"

    )





def _result_metric_means(result: Any) -> dict[str, float]:

    out: dict[str, float] = {}

    rdict = getattr(result, "_repr_dict", None)

    if isinstance(rdict, dict) and rdict:

        for k, v in rdict.items():

            try:

                if v is None or (isinstance(v, float) and math.isnan(v)):

                    continue

                out[str(k)] = float(v)

            except (TypeError, ValueError):

                pass

        if out:

            return out

    if hasattr(result, "to_pandas"):

        df = result.to_pandas()

        for col in df.columns:

            try:

                series = df[col]

                if series.dtype == object:

                    continue

                m = float(series.mean())

                if not math.isnan(m):

                    out[str(col)] = m

            except (TypeError, ValueError):

                pass

    return out





def run_ragas_eval(

    conn: sqlite3.Connection,

    items: list[dict],

    top_k: int,

    library_root: Path | None,

    *,

    generate_answers: bool,

    include_answer_relevancy: bool,

) -> dict[str, float] | None:

    try:

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        from ragas import EvaluationDataset, evaluate

        from ragas.llms.base import LangchainLLMWrapper

        from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall

    except ImportError as exc:

        print(f"[RAGAS] 未安装依赖: {exc}", file=sys.stderr)

        print("请执行: pip install -r requirements-optional.txt", file=sys.stderr)

        return None



    model = (os.environ.get("OPENAI_MODEL") or "").strip()

    if not model:

        print("[RAGAS] 需要环境变量 OPENAI_MODEL", file=sys.stderr)

        return None



    try:

        llm = ChatOpenAI(

            model=model,

            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("ARK_API_KEY"),

            base_url=os.environ.get("OPENAI_BASE_URL") or None,

        )

        evaluator_llm = LangchainLLMWrapper(llm)

    except Exception as exc:

        print(f"[RAGAS] 无法构造评测 LLM: {exc}", file=sys.stderr)

        return None



    embeddings = None

    if include_answer_relevancy:

        try:

            embeddings = OpenAIEmbeddings(

                model=_eval_embedding_model(),

                api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("ARK_API_KEY"),

                base_url=os.environ.get("OPENAI_BASE_URL") or None,

            )

        except Exception as exc:

            print(f"[RAGAS] 无法构造 Embeddings，将跳过 answer_relevancy：{exc}", file=sys.stderr)

            include_answer_relevancy = False



    metrics: list = [Faithfulness(), LLMContextRecall()]

    if include_answer_relevancy:

        metrics.append(AnswerRelevancy())



    rows: list[dict] = []

    for it in items:

        q = (it.get("question") or "").strip()

        if not q:

            continue

        ref = (it.get("reference") or it.get("ground_truth") or "").strip()

        ctxs = retrieve(conn, q, top_k, library_root=library_root)

        ctx_texts = [str(c.get("text") or "") for c in ctxs if (c.get("text") or "").strip()]

        if generate_answers:

            ans = (ask_llm(q, ctxs) or "").strip()

        else:

            ans = "\n\n".join(ctx_texts)[:2800]

        if not ref:

            ref = "（金标准未提供 reference；context_recall / 对比类指标会失真）"

        rows.append(

            {

                "user_input": q,

                "retrieved_contexts": ctx_texts,

                "response": ans,

                "reference": ref,

            }

        )



    if not rows:

        return None



    evaluation_dataset = EvaluationDataset.from_list(rows)



    try:

        result = evaluate(

            dataset=evaluation_dataset,

            metrics=metrics,

            llm=evaluator_llm,

            embeddings=embeddings,

            show_progress=False,

        )

    except Exception as exc:

        print(f"[RAGAS] evaluate 失败: {exc}", file=sys.stderr)

        return None



    return _result_metric_means(result)





def _check_min_scores(

    scores: dict[str, float],

    *,

    min_faithfulness: float | None,

    min_context_recall: float | None,

    min_answer_relevancy: float | None,

) -> tuple[bool, list[str]]:

    """

    门禁：与 RAGAS 默认 metric name 对齐（faithfulness / context_recall / answer_relevancy）。

    """

    ok = True

    msgs: list[str] = []

    checks = [

        ("faithfulness", min_faithfulness),

        ("context_recall", min_context_recall),

        ("answer_relevancy", min_answer_relevancy),

    ]

    for key, threshold in checks:

        if threshold is None:

            continue

        if key not in scores:

            ok = False

            msgs.append(f"{key} 缺失，无法与阈值 {threshold:.4f} 比较")

            continue

        if scores[key] < threshold:

            ok = False

            msgs.append(f"{key}={scores[key]:.4f} < {threshold:.4f}")

    return ok, msgs





def main() -> int:

    ap = argparse.ArgumentParser(description="RAG retrieval + optional RAGAS eval")

    ap.add_argument("--db", default="knowledge_base.sqlite", help="SQLite 知识库路径")

    ap.add_argument("--golden", required=True, help="JSONL 金标准（见 eval/golden.example.jsonl）")

    ap.add_argument("--top-k", type=int, default=4)

    ap.add_argument(

        "--library-root",

        default="",

        help="与问答一致的资料根目录；默认 effective_library_root()",

    )

    ap.add_argument(

        "--ragas",

        action="store_true",

        help="运行 RAGAS（需 API Key 与可选依赖；较慢）",

    )

    ap.add_argument(

        "--ragas-generate",

        action="store_true",

        help="RAGAS 前对每条调用 ask_llm 生成 answer（否则用检索拼接 stub）",

    )

    ap.add_argument(

        "--ragas-no-relevancy",

        action="store_true",

        help="不计算 answer_relevancy（无需 Embeddings API）",

    )

    ap.add_argument(

        "--json-out",

        default="",

        help="将检索 + RAGAS（若启用）均分写入 JSON 文件，便于 CI 留档",

    )

    ap.add_argument(

        "--min-faithfulness",

        type=float,

        default=None,

        help="与 --ragas 联用：低于该值则退出码 2（门禁）",

    )

    ap.add_argument(

        "--min-context-recall",

        type=float,

        default=None,

        help="与 --ragas 联用：context_recall 低于该值则退出码 2",

    )

    ap.add_argument(

        "--min-answer-relevancy",

        type=float,

        default=None,

        help="与 --ragas 联用：answer_relevancy 低于该值则退出码 2",

    )

    args = ap.parse_args()

    if not args.ragas and (
        args.min_faithfulness is not None
        or args.min_context_recall is not None
        or args.min_answer_relevancy is not None
    ):
        print("错误：--min-faithfulness / --min-context-recall / --min-answer-relevancy 需与 --ragas 同用", file=sys.stderr)
        return 1

    db_path = Path(args.db).resolve()

    if not db_path.is_file():

        print(f"找不到数据库: {db_path}", file=sys.stderr)

        return 1



    gold_path = Path(args.golden).resolve()

    if not gold_path.is_file():

        print(f"找不到金标准: {gold_path}", file=sys.stderr)

        return 1



    items = load_golden(gold_path)

    if not items:

        print("金标准为空", file=sys.stderr)

        return 1



    lib: Path | None = None

    if (args.library_root or "").strip():

        lib = Path(args.library_root).expanduser().resolve()

        if not lib.is_dir():

            print(f"--library-root 不是目录: {lib}", file=sys.stderr)

            return 1

    else:

        er = effective_library_root()

        lib = er if er.is_dir() else None



    conn = sqlite3.connect(str(db_path))

    conn.text_factory = str

    export: dict[str, Any] = {"top_k": args.top_k, "golden": str(gold_path), "db": str(db_path)}

    try:

        ret = run_retrieval_eval(conn, items, args.top_k, lib)

        export["retrieval"] = ret

        print("=== 检索指标（客观，依赖你标注的 relevant_snippets）===")

        print(

            f"Hit@{args.top_k}: {ret['hit_at_k']:.1%}  "

            f"MRR: {ret['mrr']:.3f}  "

            f"Precision@{args.top_k}: {ret['precision_at_k']:.3f}  "

            f"Recall@{args.top_k}: {ret['recall_at_k']:.3f}  "

            f"n={int(ret['n'])}"

        )

        print(

            "说明：Hit@k 即「至少一条金片段出现在任一检索块中」；"

            "Precision/Recall 为块级与金片段覆盖；非「模型答对」。扩充 golden.jsonl 可提高结论可信度。"

        )



        rg_out: dict[str, float] | None = None

        if args.ragas:

            print("\n=== RAGAS（LLM 判分 + 可选 Embeddings；受 API 与 reference 质量影响）===")

            rg_out = run_ragas_eval(

                conn,

                items,

                args.top_k,

                lib,

                generate_answers=args.ragas_generate,

                include_answer_relevancy=not args.ragas_no_relevancy,

            )

            if rg_out:

                export["ragas"] = rg_out

                for k in sorted(rg_out.keys()):

                    print(f"  {k}: {rg_out[k]:.4f}")



            need_gate = args.min_faithfulness is not None or args.min_context_recall is not None

            if args.min_answer_relevancy is not None and not args.ragas_no_relevancy:

                need_gate = True

            if need_gate:

                if not rg_out:

                    print("[门槛未过] RAGAS 未得到分数（检查依赖与 API）", file=sys.stderr)

                    if args.json_out:

                        outp = Path(args.json_out).resolve()

                        outp.parent.mkdir(parents=True, exist_ok=True)

                        outp.write_text(

                            json.dumps(

                                {**export, "gate_ok": False, "gate_messages": ["no_ragas_scores"]},

                                ensure_ascii=False,

                                indent=2,

                            ),

                            encoding="utf-8",

                        )

                    return 2

                ok_thr, msgs_thr = _check_min_scores(

                    rg_out,

                    min_faithfulness=args.min_faithfulness,

                    min_context_recall=args.min_context_recall,

                    min_answer_relevancy=(

                        args.min_answer_relevancy if not args.ragas_no_relevancy else None

                    ),

                )

                if not ok_thr:

                    for m in msgs_thr:

                        print(f"[门槛未过] {m}", file=sys.stderr)

                    if args.json_out:

                        outp = Path(args.json_out).resolve()

                        outp.parent.mkdir(parents=True, exist_ok=True)

                        outp.write_text(

                            json.dumps(

                                {**export, "gate_ok": False, "gate_messages": msgs_thr},

                                ensure_ascii=False,

                                indent=2,

                            ),

                            encoding="utf-8",

                        )

                    return 2



        if args.json_out:

            outp = Path(args.json_out).resolve()

            outp.parent.mkdir(parents=True, exist_ok=True)

            export["gate_ok"] = True

            outp.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"\n已写入 {outp}")

    finally:

        conn.close()



    return 0





if __name__ == "__main__":

    raise SystemExit(main())


