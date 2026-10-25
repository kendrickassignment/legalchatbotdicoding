import time
from .config import ROOT, read_json, write_json


def retrieval_evaluation(retriever, reranker, cases=None):
    cases = cases or read_json(ROOT / "data/evaluation/retrieval_cases.json")
    rows = []
    for case in cases:
        start = time.perf_counter()
        candidates = retriever.retrieve(case["question"], filters=case.get("filters"))
        ranked = reranker.rerank(case["question"], candidates)
        relevant = lambda c: c["chunk"].metadata["doc_id"] == case["doc_id"] and c["chunk"].metadata["page"] in case["relevant_pages"]
        ranks = [i for i, c in enumerate(ranked, 1) if relevant(c)]
        rows.append({"id": case["id"], "question": case["question"], "candidate_count": len(candidates),
            "candidate_hit": any(relevant(c) for c in candidates), "hit_at_3": bool(ranks),
            "reciprocal_rank": 1 / min(ranks) if ranks else 0.,
            "top1_score": ranked[0]["relevance_score"] if ranked else None,
            "top1_relevant": relevant(ranked[0]) if ranked else False,
            "top3": [{"citation": c["chunk"].metadata["citation"], "score": c["relevance_score"]} for c in ranked],
            "seconds": round(time.perf_counter() - start, 3)})
    report = {"mode": "retrieval_only_no_hyde_no_generation", "case_count": len(rows),
        "hit_at_3": sum(r["hit_at_3"] for r in rows) / len(rows),
        "mrr_at_3": sum(r["reciprocal_rank"] for r in rows) / len(rows), "cases": rows,
        "limitation": "Small development set, not held-out proof of legal answer accuracy or threshold calibration"}
    write_json(ROOT / "artifacts/reports/retrieval_eval.json", report)
    return report


def plot_training():
    import matplotlib.pyplot as plt
    from .config import settings
    fig, ax = plt.subplots(figsize=(10, 5))
    for exp in settings()["experiments"]:
        run = read_json(ROOT / "artifacts/reports" / f"{exp['name']}.json")
        for field, style in [("loss", "--"), ("eval_loss", "-")]:
            points = [(x["step"], x[field]) for x in run["log_history"] if field in x]
            ax.plot([x[0] for x in points], [x[1] for x in points], style, label=f"{exp['name']} {field}")
    ax.set(xlabel="Optimizer step", ylabel="Cross entropy loss", title="Independent SFT experiments: actual training logs")
    ax.legend()
    ax.grid(alpha=.2)
    target = ROOT / "artifacts/reports/loss_curves.png"
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target
