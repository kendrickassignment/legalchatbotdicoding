"""Inference can only load the student's verified trained model."""
from .config import settings
from .hub import verified_repo


class StudentGenerator:
    def __init__(self, stage="grpo"):
        from unsloth import FastLanguageModel
        repo, revision = verified_repo(stage)
        self.repo, self.revision = repo, revision
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name=repo, revision=revision,
            max_seq_length=8192, load_in_4bit=True, fast_inference=False)
        FastLanguageModel.for_inference(self.model)

    def __call__(self, messages, max_new_tokens=640, temperature=0., thinking=True):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        if prompt.endswith("<think>\n"):
            prompt = prompt[:-len("<think>\n")]
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        limit = 8192 - max_new_tokens
        if batch.input_ids.shape[1] > limit:
            raise ValueError("Context exceeds token budget; reduce parent size/count instead of silently truncating evidence")
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.9)
        with torch.inference_mode():
            output = self.model.generate(**batch, **kwargs)
        return self.tokenizer.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()
