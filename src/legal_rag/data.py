"""The assigned dataset only; deduplicate before deterministic splitting."""
import hashlib
import re
from .config import ROOT, settings, write_json

SFT_SYSTEM = "Anda asisten AI yang membantu pengguna dalam bahasa Indonesia. Jawab dengan akurat dan jelas."
GRPO_SYSTEM = (
    "Jawab dalam bahasa Indonesia. Awali dengan <think> berisi ringkasan alasan singkat "
    "yang dapat diperiksa, tutup dengan </think>, lalu jawaban akhir. Jangan mengarang fakta."
)


def normalize_row(row):
    def s(key):
        v = row.get(key, "")
        return "" if v is None else str(v).strip()
    instruction, context = s("instruction"), s("input")
    prompt = instruction + ("\n\n" + context if context else "") if instruction else context
    return {"prompt_text": prompt, "answer": s("output")}


def prompt_key(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def prepare_dataset(tokenizer=None):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    cfg = settings()
    csv = hf_hub_download(cfg["dataset"], "alpaca-gpt4-indonesia.csv", repo_type="dataset", revision=cfg["dataset_revision"])
    raw = load_dataset("csv", data_files=csv, split="train")
    ds = raw.map(normalize_row, remove_columns=raw.column_names, desc="Normalize actual input/output schema")
    ds = ds.filter(lambda r: bool(r["prompt_text"] and r["answer"]))
    seen, keep = set(), []
    for i, row in enumerate(ds):
        key = prompt_key(row["prompt_text"])
        if key not in seen:
            seen.add(key)
            keep.append(i)
    ds = ds.select(keep)
    split = ds.train_test_split(test_size=cfg["validation_fraction"], seed=cfg["seed"])
    assert not ({prompt_key(x) for x in split["train"]["prompt_text"]} & {prompt_key(x) for x in split["test"]["prompt_text"]})
    report = {"source": cfg["dataset"], "revision": cfg["dataset_revision"], "raw_rows": len(raw), "unique_valid_rows": len(ds), "train": len(split["train"]), "validation": len(split["test"]), "seed": cfg["seed"], "prompt_leakage": 0, "columns": raw.column_names}
    write_json(ROOT / "artifacts/reports/dataset.json", report)
    if tokenizer is not None:
        def fmt(r):
            messages = [{"role": "system", "content": SFT_SYSTEM}, {"role": "user", "content": r["prompt_text"]}, {"role": "assistant", "content": r["answer"]}]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)}
        split = split.map(fmt, desc="Dataset.map -> native Qwen ChatML")
    return split, report


def prepare_grpo(split, tokenizer):
    def fmt(r):
        messages = [{"role": "system", "content": GRPO_SYSTEM}, {"role": "user", "content": r["prompt_text"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        # Native Qwen prefixes <think>; remove that suffix so the reward sees the
        # tag in the model's completion instead of a tag prefilled by the prompt.
        if prompt.endswith("<think>\n"):
            prompt = prompt[:-len("<think>\n")]
        return {"prompt": prompt, "ground_truth": r["answer"]}
    return split.map(fmt, remove_columns=split.column_names)
