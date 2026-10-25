"""Weighted reciprocal-rank fusion -> parent expansion -> cross-encoder."""
import math
import re
import os
from pathlib import Path
import numpy as np
from .config import ROOT, settings, write_json, read_json
from .corpus import Chunk, serialize_chunks, digest


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def matches(metadata, filters):
    return all(metadata.get(k) == v for k, v in (filters or {}).items())


def sigmoid(x):
    return 1 / (1 + math.exp(-max(-60., min(60., float(x)))))


def faiss_search(vectors, queries, k):
    if os.name == "nt":
        # Some Windows FAISS/PyTorch wheels ship conflicting OpenMP runtimes.
        # Keep real FAISS search in a clean process; do not suppress OpenMP errors.
        import subprocess
        import sys
        import tempfile
        cache = ROOT / "data/cache/faiss_ipc"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            source, target = Path(directory) / "in.npz", Path(directory) / "out.npz"
            np.savez(source, vectors=np.asarray(vectors, dtype="float32"), queries=queries)
            subprocess.run([sys.executable, str(Path(__file__).with_name("faiss_worker.py")), str(source), str(target), str(k)], check=True)
            with np.load(target, allow_pickle=False) as result:
                return result["scores"].copy(), result["indices"].copy()
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype="float32"))
    return index.search(queries, k)


class HybridRetriever:
    def __init__(self, parents, children, index, encoder, cfg=None):
        from rank_bm25 import BM25Okapi
        self.parents = {p.id: p for p in parents}
        self.children, self.index, self.encoder = children, index, encoder
        self.cfg = cfg or settings()
        self.bm25 = BM25Okapi([tokenize(c.text) for c in children])

    @classmethod
    def build(cls, parents, children, destination=None, device="cpu"):
        import torch  # Load the torch OpenMP runtime before FAISS on Windows.
        import faiss
        from sentence_transformers import SentenceTransformer
        cfg = settings()
        corpus_hash = digest(ROOT / "data/processed/pages.jsonl")
        encoder = SentenceTransformer(cfg["embedding_model"], revision=cfg["embedding_revision"], device=device)
        encoder.max_seq_length = 256
        destination = Path(destination or ROOT / "artifacts/index")
        reusable = {}
        if (destination / "children.faiss").exists() and (destination / "config.json").exists():
            previous_cfg = read_json(destination / "config.json")
            if all(previous_cfg.get(key) == cfg.get(key) for key in ["embedding_model", "embedding_revision"]):
                old_index = faiss.read_index(str(destination / "children.faiss"))
                old_vectors = old_index.reconstruct_n(0, old_index.ntotal)
                for row, vector in zip(read_json(destination / "children.json"), old_vectors, strict=True):
                    reusable[(row["id"], row["text"])] = vector
        missing = [c for c in children if (c.id, c.text) not in reusable]
        if missing:
            encoded = encoder.encode(["passage: " + c.text for c in missing], normalize_embeddings=True,
                batch_size=32, show_progress_bar=True, convert_to_numpy=True).astype("float32")
            reusable.update({(c.id, c.text): v for c, v in zip(missing, encoded, strict=True)})
        vectors = np.stack([reusable[(c.id, c.text)] for c in children]).astype("float32")
        if digest(ROOT / "data/processed/pages.jsonl") != corpus_hash:
            raise RuntimeError("Corpus changed while embeddings were being built. Retry against a stable corpus snapshot.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        obj = cls(parents, children, index, encoder, cfg)
        destination.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(destination / "children.faiss"))
        write_json(destination / "parents.json", serialize_chunks(parents))
        write_json(destination / "children.json", serialize_chunks(children))
        write_json(destination / "config.json", cfg)
        write_json(destination / "corpus_fingerprint.json", {"pages_sha256": corpus_hash, "children": len(children), "new_embeddings": len(missing), "reused_embeddings": len(children)-len(missing)})
        return obj

    @classmethod
    def load(cls, source=None, device="cpu"):
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        source = Path(source or ROOT / "artifacts/index")
        cfg = read_json(source / "config.json")
        fingerprint = read_json(source / "corpus_fingerprint.json")
        if fingerprint["pages_sha256"] != digest(ROOT / "data/processed/pages.jsonl"):
            raise RuntimeError("Corpus changed after indexing. Rebuild the index, including OCR changes.")
        encoder = SentenceTransformer(cfg["embedding_model"], revision=cfg["embedding_revision"], device=device)
        encoder.max_seq_length = 256
        return cls([Chunk(**p) for p in read_json(source / "parents.json")],
            [Chunk(**p) for p in read_json(source / "children.json")],
            faiss.read_index(str(source / "children.faiss")), encoder, cfg)

    def retrieve(self, query, hypotheses=None, filters=None, k=None):
        import faiss
        k = k or self.cfg["candidate_k"]
        eligible = [i for i, c in enumerate(self.children) if matches(c.metadata, filters)]
        if not eligible:
            return []
        lexical = self.bm25.get_scores(tokenize(query))
        lexical_order = sorted(eligible, key=lambda i: lexical[i], reverse=True)[:k]
        # Apply metadata constraints BEFORE dense ranking, not after global top-k.
        vectors = self.index.reconstruct_n(0, self.index.ntotal)[eligible]
        queries = [query] + list(hypotheses or [])
        qv = self.encoder.encode(["query: " + q for q in queries], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, orders = faiss_search(vectors, qv, min(k, len(eligible)))
        rankings = [(lexical_order, self.cfg["bm25_weight"])]
        rankings += [([eligible[i] for i in order if i >= 0], self.cfg["dense_weight"] / len(queries)) for order in orders]
        fused = {}
        for order, weight in rankings:
            for rank, idx in enumerate(order, 1):
                fused[idx] = fused.get(idx, 0.) + weight / (self.cfg["rrf_k"] + rank)
        # Parent deduplication prevents overlapping children from flooding context.
        parents = {}
        for idx, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
            c = self.children[idx]
            pid = c.metadata["parent_id"]
            if pid not in parents:
                parents[pid] = {"chunk": self.parents[pid], "rrf_score": score, "matched_child": c.text}
        return list(parents.values())[:k]


class Reranker:
    def __init__(self, model=None):
        if model is None:
            import torch
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(settings()["reranker_model"], revision=settings()["reranker_revision"], device="cpu", max_length=512)
        self.model = model

    def rerank(self, question, candidates, k=None):
        if not candidates:
            return []
        import torch
        # Identity explicitly returns raw logits. Sigmoid below is applied once.
        import inspect
        # ST 3 uses activation_fct; newer ST uses activation_fn.
        params = inspect.signature(self.model.predict).parameters
        activation_key = "activation_fct" if "activation_fct" in params else "activation_fn"
        logits = self.model.predict([(question, c["chunk"].text) for c in candidates],
            **{activation_key: torch.nn.Identity()}, show_progress_bar=False, batch_size=8)
        ranked = [{**c, "relevance_logit": float(np.asarray(logit).reshape(-1)[0]),
                   "relevance_score": sigmoid(np.asarray(logit).reshape(-1)[0])}
                  for c, logit in zip(candidates, logits, strict=True)]
        return sorted(ranked, key=lambda x: x["relevance_score"], reverse=True)[:k or settings()["rerank_k"]]
