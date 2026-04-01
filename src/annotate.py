"""Annotation utilities for labeling hallucinations."""
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
from .schema import ModelOutput, LabeledExample
from .prompts import build_judge_prompt
from .llm_clients import NIMClient

load_dotenv()


def safe_parse_json(s: str):
    """Best-effort: extract first JSON object from judge output."""
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def normalize_label(label: str) -> str:
    """Normalize judge output to one of: N, H-, H+"""
    label = (label or "").strip()
    # exact match first
    if label in {"N", "H-", "H+"}:
        return label
    # fuzzy fallback
    upper = label.upper()
    if "H+" in upper:
        return "H+"
    if "H-" in upper:
        return "H-"
    # default — no hallucination
    return "N"


def main():
    in_path  = "data/raw_outputs_PRESSURED.jsonl"
    out_path = "data/labeled_PRESSURED.jsonl"
    judge_model_name = "nim-judge"

    judge = NIMClient(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.0,        # deterministic judge
        max_tokens=256,
    )

    with open(in_path, "r", encoding="utf-8") as f:
        content = f.read()

    # entries are separated by double newline in generate.py
    json_strings = [s.strip() for s in content.split("\n\n") if s.strip()]
    rows = [ModelOutput.model_validate_json(js) for js in json_strings]

    with open(out_path, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="Annotating"):
            jprompt = build_judge_prompt(r.task_type, r.prompt, r.response)
            jtext   = judge.generate(jprompt, temperature=0.0)
            parsed  = safe_parse_json(jtext)

            if parsed is None:
                # fallback if judge returns invalid JSON
                parsed = {
                    "label": "N",
                    "rationale": "Judge output was not valid JSON; defaulted to N."
                }

            label     = normalize_label(parsed.get("label", "N"))
            rationale = (parsed.get("rationale") or "").strip()

            ex = LabeledExample(
                task_type   = r.task_type,
                prompt_id   = r.prompt_id,
                variant     = r.variant,        # carry variant through
                prompt      = r.prompt,
                model_name  = r.model_name,
                response    = r.response,
                label       = label,
                rationale   = rationale,
                judge_model = judge_model_name,
                meta        = r.meta
            )
            out.write(ex.model_dump_json() + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()