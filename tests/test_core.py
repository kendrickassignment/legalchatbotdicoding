"""Deterministic contract tests. Test doubles are never reported as model inference."""
import pytest
from legal_rag.data import normalize_row, prompt_key
from legal_rag.rewards import format_reward_func, reasoning_length_reward, correctness_reward, final_answer
from legal_rag.retrieval import matches, sigmoid
from legal_rag.pipeline import should_fallback, LegalRAG
from legal_rag.corpus import Chunk, make_chunks


@pytest.mark.parametrize("row,expected", [
    ({"instruction": "Tanya", "input": "Konteks", "output": "Jawab"}, "Tanya\n\nKonteks"),
    ({"input": "Pertanyaan", "output": "Jawab"}, "Pertanyaan"),
    ({"instruction": "Tanya", "input": None}, "Tanya"),
])
def test_actual_dataset_schema(row, expected):
    assert normalize_row(row)["prompt_text"] == expected


def test_prompt_dedup_before_split():
    assert prompt_key("ABC  def ") == prompt_key("abc def")


@pytest.mark.parametrize("text,score", [("<think>x</think>Jawab", 1.), ("<think>x", .2), ("x</think>", .3),
    ("<think>x</think><think>y</think>Jawab", .5), ("<think></think>Jawab", 1.), ("Jawab", 0.)])
def test_shaped_format_reward(text, score):
    assert format_reward_func([text]) == [score]


@pytest.mark.parametrize("body,score", [("", 0.), ("   ", 0.), ("a"*49, .2), ("a"*50, .5), ("a"*199, .5), ("a"*200, 1.)])
def test_length_boundary_and_truncated_think(body, score):
    assert reasoning_length_reward(["<think>" + body]) == [score]
    assert reasoning_length_reward(["<think>" + body + "</think>Jawaban"]) == [score]


def test_ground_truth_must_be_in_final_answer():
    assert correctness_reward(["<think>upah lembur</think>tidak tahu"], ["upah lembur"]) == [0.]
    assert correctness_reward(["<think>x</think>Anda berhak menerima upah lembur"], ["upah lembur"]) == [1.]
    assert final_answer("<think>unfinished") == ""


def test_filter_matches_all_fields():
    assert matches({"year": 2021, "doc_id": "A"}, {"year": 2021, "doc_id": "A"})
    assert not matches({"year": 2021}, {"doc_id": "A"})


def test_fallback_threshold_boundary():
    assert should_fallback([], .45)
    assert should_fallback([{"relevance_score": .44}], .45)
    assert not should_fallback([{"relevance_score": .45}], .45)
    assert sigmoid(0) == .5


def test_parent_child_provenance():
    pages = [{"text": "Kalimat tentang peraturan. "*400, "metadata": {"doc_id": "X", "page": 1}}]
    parents, children = make_chunks(pages)
    assert len(children) > len(parents) > 1
    assert max(len(c.text) for c in children) <= 800
    assert all(c.metadata["parent_id"] in {p.id for p in parents} for c in children)


class StubRetriever:
    def retrieve(self, *a, **kw):
        return [{"chunk": Chunk(str(i), "INTERNAL_SECRET", {"citation": "Test source", "page": i}), "rrf_score": .1} for i in range(5)]


class StubReranker:
    def __init__(self, score):
        self.score = score

    def rerank(self, q, candidates):
        return [{**c, "relevance_score": self.score} for c in candidates[:3]]


def test_no_automatic_confidential_web_egress(monkeypatch):
    def deny(*a, **kw):
        raise AssertionError("web must not be called")
    monkeypatch.setattr("legal_rag.pipeline.duckduckgo_search", deny)
    rag = LegalRAG(StubRetriever(), StubReranker(.1), lambda *a, **kw: "hypothetical", allow_web=True)
    assert rag.ask("secret client")['status'] == "web_requires_public_query"


def test_web_fallback_discards_local_context(monkeypatch):
    seen = []
    def generator(messages, **kw):
        seen.append(str(messages))
        return "<think>Ringkasan</think>Jawab [S1]"
    monkeypatch.setattr("legal_rag.pipeline.duckduckgo_search", lambda query: [{"title": "Official", "url": "https://peraturan.go.id/x", "text": "PUBLIC_SNIPPET", "source_type": "web_search_snippet"}])
    result = LegalRAG(StubRetriever(), StubReranker(.1), generator, allow_web=True).ask("private", public_web_query="peraturan publik")
    assert result["fallback_used"]
    assert "INTERNAL_SECRET" not in seen[-1]
    assert "PUBLIC_SNIPPET" in seen[-1]


def test_fabricated_citation_fails_closed():
    result = LegalRAG(StubRetriever(), StubReranker(.8), lambda *a, **kw: "Jawab [S99]").ask("question")
    assert result["status"] == "citation_validation_failed"


def test_student_model_required():
    from legal_rag.hub import verified_repo
    from legal_rag.config import ROOT
    if not (ROOT / "artifacts/reports/hub.json").exists():
        with pytest.raises(RuntimeError, match="No published"):
            verified_repo()


def test_hybrid_weights_filtering_and_parent_expansion():
    import torch
    import faiss
    import numpy as np
    from legal_rag.retrieval import HybridRetriever
    from legal_rag.config import settings
    class Encoder:
        def encode(self, q, **kw):
            return np.asarray([[0., 1.]] * len(q), dtype="float32")
    parents = [Chunk("A", "lembur hak pekerja", {"doc_id": "A", "parent_id": "A"}),
               Chunk("B", "perizinan usaha", {"doc_id": "B", "parent_id": "B"})]
    children = [Chunk("Ac", "lembur hak pekerja", parents[0].metadata), Chunk("Bc", "perizinan usaha", parents[1].metadata)]
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1., 0.], [0., 1.]], dtype="float32"))
    cfg = {**settings(), "dense_weight": .9, "bm25_weight": .1}
    retriever = HybridRetriever(parents, children, index, Encoder(), cfg)
    assert retriever.retrieve("lembur")[0]["chunk"].id == "B"
    cfg.update(dense_weight=.1, bm25_weight=.9)
    assert retriever.retrieve("lembur")[0]["chunk"].id == "A"
    filtered = retriever.retrieve("lembur", filters={"doc_id": "A"})
    assert [r["chunk"].id for r in filtered] == ["A"]


def test_reranker_uses_raw_logit_once():
    import numpy as np
    from legal_rag.retrieval import Reranker
    class Cross:
        def predict(self, pairs, activation_fct=None, **kw):
            import torch
            assert isinstance(activation_fct, torch.nn.Identity)
            return np.asarray([0., 2.])
    candidates = [{"chunk": Chunk("a", "a", {})}, {"chunk": Chunk("b", "b", {})}]
    ranked = Reranker(Cross()).rerank("q", candidates)
    assert ranked[0]["chunk"].id == "b"
    assert ranked[0]["relevance_score"] == pytest.approx(sigmoid(2.))
    assert ranked[1]["relevance_score"] == .5
