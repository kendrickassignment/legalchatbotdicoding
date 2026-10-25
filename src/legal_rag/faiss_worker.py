"""FAISS-only subprocess for Windows wheels with a separate OpenMP runtime."""
import sys
import numpy as np
import faiss


def main():
    source, target, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data = np.load(source, allow_pickle=False)
    vectors, queries = data["vectors"], data["queries"]
    faiss.omp_set_num_threads(2)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, indices = index.search(queries, k)
    np.savez(target, scores=scores, indices=indices)


if __name__ == "__main__":
    main()
