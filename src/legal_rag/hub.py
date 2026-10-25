"""Only publish real completed runs, and verify anonymously accessible merged weights."""
import os
from .config import ROOT, settings, read_json, write_json


def publish(stage):
    from unsloth import FastLanguageModel
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_SFT_REPO" if stage == "sft" else "HF_GRPO_REPO", "")
    if not token or repo.count("/") != 1:
        raise RuntimeError("Set HF_TOKEN and HF_SFT_REPO/HF_GRPO_REPO using environment variables or Colab Secrets")
    api = HfApi(token=token)
    me = api.whoami()
    allowed = {me["name"], *(o["name"] for o in me.get("orgs", []))}
    if repo.split("/")[0] not in allowed:
        raise ValueError("Destination must be owned by this user or their organization")
    if repo == settings()["base_model"]:
        raise ValueError("Cannot overwrite the upstream model")
    name = read_json(ROOT / "artifacts/reports/selected_sft.json")["winner"] if stage == "sft" else "grpo"
    evidence = read_json(ROOT / "artifacts/reports" / f"{name}.json")
    if not evidence["complete"] or (stage == "sft" and evidence["global_step"] < 800):
        raise RuntimeError("Completed training evidence is mandatory")
    if stage == "sft":
        for exp in settings()["experiments"]:
            run = read_json(ROOT / "artifacts/reports" / f"{exp['name']}.json")
            if not run.get("complete") or run["global_step"] < 800:
                raise RuntimeError("Both SFT experiments must complete before publication")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(ROOT / evidence["adapter_path"]),
        max_seq_length=settings()["max_seq_length"], load_in_4bit=True, fast_inference=False, token=token)
    api.create_repo(repo_id=repo, private=False, exist_ok=True)
    if api.model_info(repo).private:
        api.update_repo_settings(repo_id=repo, private=False)
    # Unsloth's correct merged API is push_to_hub_merged, not plain push_to_hub.
    model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", token=token, private=False)
    parent = settings()["base_model"] if stage == "sft" else verified_repo("sft")[0]
    model_card = f"""---
base_model: {parent}
datasets:
- Ichsan2895/alpaca-gpt4-indonesian
language:
- id
tags:
- qlora
- legal-rag
- student-project
---
# LexID {stage.upper()} — Kendrick Filbert

Student-trained derivative, exported as merged 16-bit weights using Unsloth.
Original ancestor: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
revision `{settings()['base_revision']}`.

Training data: [Ichsan2895/alpaca-gpt4-indonesian](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian),
revision `{settings()['dataset_revision']}`. Dataset attribution: Ichsan2895,
CC BY-SA 4.0; review upstream model and dataset license obligations before redistribution.
No legal PDF is used as SFT/GRPO training data; legal evidence is supplied through RAG.

Completed optimizer steps: {evidence['global_step']}.
Method: QLoRA NF4 double quantization, FFN gate/up/down projection adapters.
Use the repository's native chat template. This model is not a source of current legal advice.
GRPO reward shaping and lexical correctness proxies do not prove factual accuracy.

## Actual recorded metrics
```json
{__import__('json').dumps(evidence.get('evaluation', evidence.get('metrics', {})), indent=2)}
```
See the companion project notebooks and training reports for experiment selection,
hardware, complete logs, and known limitations. No benchmark score is claimed here.
"""
    api.upload_file(path_or_fileobj=model_card.encode("utf-8"), path_in_repo="README.md", repo_id=repo)
    info = HfApi(token=False).model_info(repo)
    files = {s.rfilename for s in info.siblings}
    if info.private or not any(f.endswith(".safetensors") and not f.startswith("adapter_") for f in files):
        raise RuntimeError("Public merged weights were not verified")
    receipts_path = ROOT / "artifacts/reports/hub.json"
    receipts = read_json(receipts_path) if receipts_path.exists() else {}
    receipts[stage] = {"repo": repo, "revision": info.sha, "public": True, "merged_16bit": True, "training_run": name}
    write_json(receipts_path, receipts)
    (ROOT / "link_huggingface.txt").write_text("\n".join(f"https://huggingface.co/{v['repo']}" for v in receipts.values()) + "\n", encoding="utf-8")
    return receipts[stage]


def verified_repo(stage="grpo"):
    path = ROOT / "artifacts/reports/hub.json"
    if not path.exists():
        raise RuntimeError("No published student model. Finish SFT/GRPO and publication first.")
    receipt = read_json(path).get(stage)
    if not receipt or not receipt.get("public") or not receipt.get("merged_16bit"):
        raise RuntimeError(f"No verified public {stage} model")
    return receipt["repo"], receipt["revision"]
