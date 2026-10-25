"""GPU-only training; never substitute synthetic losses or smaller models."""
import gc
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from .config import ROOT, settings, write_json


def preflight(require_gpu=True):
    import torch
    info = {"python": platform.python_version(), "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(), "packages": {}}
    for pkg in ["torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "bitsandbytes"]:
        try:
            info["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][pkg] = "not installed"
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gib=round(p.total_memory / 2**30, 1))
    write_json(ROOT / "artifacts/reports/preflight.json", info)
    if require_gpu and not info["cuda"]:
        raise RuntimeError("Training requires a CUDA GPU. This runtime is CPU-only; no training was performed.")
    return info


def load_qlora(model_id=None, revision=None, r=16, alpha=32):
    # Unsloth must be imported before transformers / TRL in training processes.
    from unsloth import FastLanguageModel
    import torch
    from transformers import BitsAndBytesConfig
    cfg = settings()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id or cfg["base_model"], revision=revision,
        max_seq_length=cfg["max_seq_length"], load_in_4bit=True,
        quantization_config=quant, fast_inference=False, token=os.getenv("HF_TOKEN"),
    )
    # All FFN projections of the language model: a complete main computation component.
    # Qwen3.8 uses hybrid attention, so q/k/v-only LoRA would omit many layers.
    targets = ["gate_proj", "up_proj", "down_proj"]
    names = [n for n, _ in model.named_modules()]
    missing = [t for t in targets if not any(n.endswith("." + t) for n in names)]
    if missing:
        raise RuntimeError(f"Architecture adapter targets unsupported: {missing}")
    model = FastLanguageModel.get_peft_model(model, r=r, target_modules=targets, lora_alpha=alpha,
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    # Verify the live quantization configuration instead of merely declaring it.
    q = model.config.quantization_config
    q = q.to_dict() if hasattr(q, "to_dict") else q
    if not q.get("load_in_4bit") or not q.get("bnb_4bit_use_double_quant"):
        raise RuntimeError("Live model does not meet NF4/double-quantization requirements")
    write_json(ROOT / "artifacts/reports/quantization.json", q)
    model.print_trainable_parameters()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_sft(experiment):
    from unsloth import is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer
    import torch
    from .data import prepare_dataset
    cfg = settings()
    preflight()
    exp = next(x for x in cfg["experiments"] if x["name"] == experiment)
    if exp["max_steps"] < 800:
        raise ValueError("Submission SFT must run for at least 800 steps")
    out = ROOT / "artifacts/checkpoints" / experiment
    model, tokenizer = load_qlora(revision=cfg["base_revision"], r=exp["r"], alpha=exp["lora_alpha"])
    ds, report = prepare_dataset(tokenizer)
    print(ds["train"][0]["text"])
    (ROOT / "artifacts/reports" / f"{experiment}_mapped_example.txt").write_text(ds["train"][0]["text"], encoding="utf-8")
    args = SFTConfig(output_dir=str(out), max_length=cfg["max_seq_length"], dataset_text_field="text",
        packing=False, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, max_steps=exp["max_steps"], learning_rate=exp["learning_rate"],
        warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type="cosine", optim="adamw_8bit",
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        eval_strategy="steps", eval_steps=100, logging_steps=10,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        seed=cfg["seed"], data_seed=cfg["seed"], report_to="none", dataset_num_proc=1)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds["train"], eval_dataset=ds["test"], args=args)
    baseline = trainer.evaluate()
    try:
        result = trainer.train(resume_from_checkpoint=True if any(out.glob("checkpoint-*")) else None)
    except torch.cuda.OutOfMemoryError:
        write_json(ROOT / "artifacts/reports" / f"{experiment}_failure.json", {"status": "OOM", "complete": False})
        raise
    if trainer.state.global_step < 800:
        raise RuntimeError("Training did not reach required steps")
    adapter = ROOT / "artifacts/adapters" / experiment
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    evaluation = trainer.evaluate()
    write_json(ROOT / "artifacts/reports" / f"{experiment}.json", {
        "experiment": exp, "base_model": cfg["base_model"], "base_revision": cfg["base_revision"],
        "dataset": report, "complete": True, "global_step": trainer.state.global_step,
        "baseline": baseline, "evaluation": evaluation, "train_metrics": result.metrics,
        "log_history": trainer.state.log_history, "adapter_path": str(adapter.relative_to(ROOT)),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30})
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def compare_experiments():
    from .config import read_json
    records = [read_json(ROOT / "artifacts/reports" / f"{e['name']}.json") for e in settings()["experiments"]]
    if not all(r["complete"] and r["global_step"] >= 800 for r in records):
        raise RuntimeError("Both independent SFT experiments must finish")
    best = min(records, key=lambda r: r["evaluation"]["eval_loss"])
    selected = {"winner": best["experiment"]["name"], "criterion": "lowest validation loss; inspect curves and generalization gap before publishing", "eval_losses": {r["experiment"]["name"]: r["evaluation"]["eval_loss"] for r in records}}
    write_json(ROOT / "artifacts/reports/selected_sft.json", selected)
    return selected


def run_grpo():
    from unsloth import is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    from .data import prepare_dataset, prepare_grpo
    from .rewards import format_reward_func, reasoning_length_reward, correctness_reward, language_reward_func
    from .hub import verified_repo
    preflight()
    cfg, rl = settings(), settings()["grpo"]
    repo, revision = verified_repo("sft")
    model, tokenizer = load_qlora(repo, revision, r=16, alpha=32)
    ds, _ = prepare_dataset()
    train = prepare_grpo(ds["train"], tokenizer)
    valid = prepare_grpo(ds["test"].select(range(min(64, len(ds["test"])))), tokenizer)
    # TRL 1.15 removed max_prompt_length. Filter by actual token length instead.
    train = train.filter(lambda r: len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]) <= rl["max_prompt_length"])
    valid = valid.filter(lambda r: len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]) <= rl["max_prompt_length"])
    out = ROOT / "artifacts/checkpoints/grpo"
    args = GRPOConfig(output_dir=str(out), learning_rate=5e-6, max_steps=rl["max_steps"],
        per_device_train_batch_size=1, per_device_eval_batch_size=2, gradient_accumulation_steps=rl["gradient_accumulation_steps"],
        num_generations=rl["num_generations"], max_completion_length=rl["max_completion_length"],
        use_vllm=False, beta=0.04,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        optim="adamw_8bit", logging_steps=5, save_steps=50, save_total_limit=2,
        eval_strategy="steps", eval_steps=50, report_to="none", seed=cfg["seed"],
        remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=args,
        train_dataset=train, eval_dataset=valid,
        reward_funcs=[format_reward_func, reasoning_length_reward, correctness_reward, language_reward_func])
    result = trainer.train(resume_from_checkpoint=True if any(out.glob("checkpoint-*")) else None)
    if trainer.state.global_step < rl["max_steps"]:
        raise RuntimeError("GRPO incomplete")
    adapter = ROOT / "artifacts/adapters/grpo"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    write_json(ROOT / "artifacts/reports/grpo.json", {"complete": True, "source_sft": repo, "source_revision": revision,
        "global_step": trainer.state.global_step, "metrics": result.metrics, "log_history": trainer.state.log_history,
        "adapter_path": str(adapter.relative_to(ROOT))})
