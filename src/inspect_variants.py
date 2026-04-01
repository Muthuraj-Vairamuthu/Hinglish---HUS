import json
from collections import defaultdict, Counter
from pathlib import Path

path = Path(r"d:/Projects/Hinglish---HUS/data/raw_outputs_PRESSURED.jsonl")
text = path.read_text(encoding='utf-8')
import re
blocks = re.split(r"\n\s*\n", text)
items = []
for b in blocks:
    b = b.strip()
    if not b:
        continue
    try:
        items.append(json.loads(b))
    except Exception:
        continue

variants = Counter()
groups = defaultdict(set)
for it in items:
    v = it.get('variant')
    if v: variants[v]+=1
    key = (it.get('prompt_id'), it.get('model_name'))
    if key[0] and key[1] and v:
        groups[key].add(v)

print('Unique variants and counts:')
for v,c in variants.items():
    print(f'  {v}: {c}')

complete = 0
for k,s in groups.items():
    if all(x in s for x in ('base','topic_fronted','emphasis_shift')):
        complete+=1

print('\nGroups with all three variants (prompt_id, model):', complete)
print('\nSample groups (up to 10) and their variants:')
cnt=0
for k,s in list(groups.items())[:10]:
    print(k, sorted(s))
    cnt+=1
    if cnt>=10: break
