import json
import sys
import subprocess
from collections import defaultdict
from pathlib import Path


def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
        return True
    except Exception:
        print(f"Installing {pkg_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        try:
            __import__(import_name or pkg_name)
            return True
        except Exception as e:
            print(f"Failed to import {pkg_name} after install: {e}")
            return False


def load_jsonl(path):
    # The file contains pretty-printed JSON objects separated by blank lines
    import re
    items = []
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        try:
            obj = json.loads(b)
            items.append(obj)
        except Exception:
            # ignore blocks that aren't valid JSON
            continue
    return items


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="path to raw_outputs_PRESSURED.jsonl")
    parser.add_argument("--out-csv", default="data/metrics_PRESSURED.csv")
    parser.add_argument("--out-json", default="data/metrics_PRESSURED.json")
    args = parser.parse_args()

    data_path = Path(args.jsonl)
    if not data_path.exists():
        print("Input file not found:", data_path)
        sys.exit(1)

    # Ensure dependencies
    ok1 = ensure_package("sacrebleu")
    ok2 = ensure_package("bert-score", "bert_score")
    if not (ok1 and ok2):
        print("Required packages not available. Installed attempts finished.")

    from sacrebleu import sentence_bleu
    from bert_score import score as bert_score

    items = load_jsonl(str(data_path))

    groups = defaultdict(dict)  # key: (prompt_id, model_name) -> variant -> response

    for it in items:
        pid = it.get("prompt_id")
        model = it.get("model_name")
        variant = it.get("variant")
        resp = it.get("response")
        if not (pid and model and variant and resp):
            continue
        key = (pid, model)
        # keep first seen response for a variant
        if variant not in groups[key]:
            groups[key][variant] = resp

    pairs = [("base", "topic_fronted"), ("base", "emphasis_shift"), ("topic_fronted", "emphasis_shift")]

    rows = []
    for (pid, model), varmap in groups.items():
        if not all(v in varmap for v in ["base", "topic_fronted", "emphasis_shift"]):
            continue
        for a, b in pairs:
            ref = varmap[a]
            hyp = varmap[b]
            try:
                bleu_obj = sentence_bleu(hyp, [ref])
                bleu = float(bleu_obj.score)
            except Exception:
                bleu = None

            try:
                # use a smaller multilingual distil model to reduce downloads/time
                (P, R, F) = bert_score([hyp], [ref], lang="xx", model_type="distilbert-base-multilingual-cased", rescale_with_baseline=False)
                bert_f1 = float(F[0].item()) * 100.0
            except Exception:
                bert_f1 = None

            rows.append({
                "prompt_id": pid,
                "model": model,
                "pair": f"{a} vs {b}",
                "bleu": bleu,
                "bertscore_f1": bert_f1,
                "ref": ref,
                "hyp": hyp,
            })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "model", "pair", "bleu", "bertscore_f1"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in ["prompt_id", "model", "pair", "bleu", "bertscore_f1"]})

    out_json = Path(args.out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # print simple aggregates
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    bleu_vals = [r["bleu"] for r in rows if r["bleu"] is not None]
    bert_vals = [r["bertscore_f1"] for r in rows if r["bertscore_f1"] is not None]

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Counts:", len(rows), "pairwise comparisons")
    print("Mean BLEU:", mean(bleu_vals))
    print("Mean BERTScore F1 (%%):", mean(bert_vals))
    print("Note: COMET is not computed by this script. To add COMET, install the official COMET package and load a pretrained regressor.")


if __name__ == "__main__":
    main()
