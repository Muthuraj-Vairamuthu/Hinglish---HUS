"""Generate LLM outputs for benchmarking."""
from tqdm import tqdm
from dotenv import load_dotenv

if __package__ in (None, ""):
    import os
    import sys

    # Allow running as a direct script: python src/generate.py
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.schema import ModelOutput
    from src.prompts import load_prompts, build_generation_prompt
    from src.llm_clients import NIMClient
else:
    from .schema import ModelOutput
    from .prompts import load_prompts, build_generation_prompt
    from .llm_clients import NIMClient

load_dotenv()


def extract_response(text: str) -> str:
    if not text:
        return ""
    if "RESPONSE:" in text:
        return text.split("RESPONSE:", 1)[1].strip()
    return text.strip()


def main():
    prompts = load_prompts("data/prompts.json")

    models = [
        ("nim-llama", NIMClient(
            model="meta/llama-3.1-8b-instruct",
            temperature=0.7,
            max_tokens=512,
        )),
        ("nim-mistral", NIMClient(
            model="mistralai/mistral-7b-instruct-v0.2",
            temperature=0.7,
            max_tokens=512,
        )),
        ("nim-qwen", NIMClient(
            model="qwen/qwen2.5-7b-instruct",
            temperature=0.7,
            max_tokens=512,
        )),
    ]

    out_path = "data/raw_outputs_PRESSURED.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in tqdm(prompts, desc="Generating"):
            # build_generation_prompt no longer takes mode — always PRESSURED
            gen_prompt = build_generation_prompt(p.task_type, p.prompt)

            for model_name, client in models:
                text = client.generate(gen_prompt)
                resp = extract_response(text)

                item = ModelOutput(
                    task_type=p.task_type,
                    prompt_id=p.prompt_id,
                    variant=p.variant,          # carry variant through
                    prompt=p.prompt,
                    model_name=model_name,
                    response=resp,
                    meta={"temperature": client.temperature, "mode": "PRESSURED"}
                )
                f.write(item.model_dump_json(indent=2) + "\n\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()