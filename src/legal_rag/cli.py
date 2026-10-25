import argparse
import json


def make_rag():
    # Keep Unsloth first in an inference process too.
    from .generation import StudentGenerator
    generator = StudentGenerator()
    from .retrieval import HybridRetriever, Reranker
    from .pipeline import LegalRAG
    return LegalRAG(HybridRetriever.load(), Reranker(), generator)


def main():
    parser = argparse.ArgumentParser(description="LexID legal RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("extract")
    sub.add_parser("ocr")
    sub.add_parser("index")
    sub.add_parser("evaluate")
    sub.add_parser("compare")
    train = sub.add_parser("train")
    train.add_argument("stage", choices=["sft_a", "sft_b", "grpo"])
    push = sub.add_parser("publish")
    push.add_argument("stage", choices=["sft", "grpo"])
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--public-web-query", default=None)
    sub.add_parser("chat")
    sub.add_parser("ui")
    args = parser.parse_args()
    if args.command == "preflight":
        from .training import preflight
        print(json.dumps(preflight(False), indent=2))
    elif args.command == "extract":
        from .corpus import extract_corpus
        extract_corpus()
    elif args.command == "ocr":
        from .ocr import ocr_sparse_pages
        print(ocr_sparse_pages())
    elif args.command == "index":
        from .corpus import load_pages, make_chunks
        from .retrieval import HybridRetriever
        parents, children = make_chunks(load_pages())
        HybridRetriever.build(parents, children)
    elif args.command == "evaluate":
        from .retrieval import HybridRetriever, Reranker
        from .evaluation import retrieval_evaluation
        print(json.dumps(retrieval_evaluation(HybridRetriever.load(), Reranker()), indent=2, ensure_ascii=False))
    elif args.command == "train":
        if args.stage == "grpo":
            from .training import run_grpo
            run_grpo()
        else:
            from .training import run_sft
            run_sft(args.stage)
    elif args.command == "compare":
        from .training import compare_experiments
        print(compare_experiments())
    elif args.command == "publish":
        from .hub import publish
        print(publish(args.stage))
    else:
        rag = make_rag()
        from .interfaces import interactive_loop, launch_gradio
        if args.command == "chat":
            interactive_loop(rag)
        elif args.command == "ui":
            launch_gradio(rag)
        else:
            print(json.dumps(rag.ask(args.question, public_web_query=args.public_web_query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
