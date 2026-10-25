"""Rubric reward functions. Tags contain a concise rationale, not private deliberation."""
import re
from collections import Counter


def completion_text(c):
    return c if isinstance(c, str) else "".join(m.get("content", "") for m in c)


def final_answer(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return "" if "<think>" in text else text.strip()


def format_reward_func(completions, **kwargs):
    out = []
    for c in completions:
        s = completion_text(c)
        score = 0.2 * s.lstrip().startswith("<think>") + 0.3 * ("</think>" in s)
        if re.fullmatch(r"<think>[\s\S]*?</think>\s*\S[\s\S]*", s.strip()):
            score = 1.0
        if s.count("<think>") > 1 or s.count("</think>") > 1:
            score -= 0.5
        out.append(float(score))
    return out


def reasoning_length_reward(completions, **kwargs):
    scores = []
    for c in completions:
        s = completion_text(c)
        match = re.search(r"<think>([\s\S]*?)(?:</think>|$)", s)
        n = len(match.group(1).strip()) if match else 0
        scores.append(0.0 if n == 0 else 0.2 if n < 50 else 0.5 if n < 200 else 1.0)
    return scores


def token_f1(a, b):
    # Transparent unigram F1 (ROUGE-1-style overlap), not a semantic correctness oracle.
    ta, tb = Counter(re.findall(r"\w+", a.lower())), Counter(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    overlap = sum((ta & tb).values())
    return 2 * overlap / (sum(ta.values()) + sum(tb.values()))


def correctness_reward(completions, ground_truth, **kwargs):
    scores = []
    for c, gt in zip(completions, ground_truth, strict=True):
        answer = final_answer(completion_text(c))
        norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
        scores.append(float(bool(norm(gt)) and (norm(gt) in norm(answer) or token_f1(answer, gt) >= 0.65)))
    return scores


def language_reward_func(completions, **kwargs):
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 3407
    out = []
    for c in completions:
        s = final_answer(completion_text(c))
        if len(re.findall(r"[A-Za-z]", s)) < 20:
            out.append(0.0)
            continue
        try:
            langs = detect_langs(s)
            p_id = sum(x.prob for x in langs if x.lang == "id")
            # Classification is an imperfect proxy for Indonesian grammar.
            score = 1.0 if p_id >= 0.8 else -0.5 if any(x.lang == "en" and x.prob >= 0.5 for x in langs) else 0.0
        except Exception:
            score = 0.0
        out.append(score)
    return out
