import os

import pandas as pd, json, re, random

random.seed(42)
DATA_PATH = "https://raw.githubusercontent.com/iSMELL2024/iSMELL/main/CodeDetection/updated_dataset.xlsx"
CACHE_FILE = os.path.dirname(os.path.abspath(__file__)) + "/raw_file_cache.json"

df = pd.read_excel(DATA_PATH)
is_github = df["Link"].astype(str).str.startswith("https://github.com/")
gh = df[is_github].copy()

with open(CACHE_FILE) as f:
    cache = json.load(f)

LINK_RE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/blob/([0-9a-f]+)/(.+?)(?:#L(\d+)(?:-L(\d+))?)?$")
def to_raw(link):
    m = LINK_RE.match(str(link).strip())
    owner, repo, commit, path, l1, l2 = m.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"

gh["raw_url"] = gh["Link"].apply(to_raw)

# Build per-file positive intervals
file_positive_intervals = {}
records = []
WIN = 60  # window size (lines) for negative sampling; positives use their own labeled span, truncated later by tokenizer

for _, row in gh.iterrows():
    raw = row["raw_url"]
    text = cache.get(raw)
    if not text:
        continue
    lines = text.split("\n")
    start = int(row["Starting Line Number"]) if pd.notna(row["Starting Line Number"]) else 1
    end = int(row["Ending Line Number"]) if pd.notna(row["Ending Line Number"]) else min(start + WIN, len(lines))
    start = max(1, start)
    end = min(len(lines), end)
    snippet = "\n".join(lines[start-1:end])
    if len(snippet.strip()) < 20:
        continue
    records.append({
        "text": snippet, "label": 1, "smell": row["Smell"],
        "raw_url": raw, "start": start, "end": end, "n_lines": len(lines),
    })
    file_positive_intervals.setdefault(raw, []).append((start, end))

print("Positive snippets extracted:", len(records))

# Negative sampling: windows outside positive intervals, same files (+ any other cached files)
neg_records = []
all_files = list(cache.keys())
random.shuffle(all_files)
target_neg = len(records)

def overlaps(a, b, intervals):
    return any(not (b < s or a > e) for s, e in intervals)

for raw in all_files:
    if len(neg_records) >= target_neg:
        break
    text = cache.get(raw)
    if not text:
        continue
    lines = text.split("\n")
    n = len(lines)
    if n < WIN + 10:
        continue
    intervals = file_positive_intervals.get(raw, [])
    tries = 0
    while tries < 5 and len(neg_records) < target_neg:
        tries += 1
        s = random.randint(1, max(1, n - WIN))
        e = min(n, s + WIN)
        if overlaps(s, e, intervals):
            continue
        snippet = "\n".join(lines[s-1:e])
        if len(snippet.strip()) < 20:
            continue
        neg_records.append({
            "text": snippet, "label": 0, "smell": "None",
            "raw_url": raw, "start": s, "end": e, "n_lines": n,
        })
        break  # one negative per file per pass to spread across files

# second pass if still short, allow multiple per file
pass_i = 0
while len(neg_records) < target_neg and pass_i < 5:
    pass_i += 1
    for raw in all_files:
        if len(neg_records) >= target_neg:
            break
        text = cache.get(raw)
        if not text:
            continue
        lines = text.split("\n")
        n = len(lines)
        if n < WIN + 10:
            continue
        intervals = file_positive_intervals.get(raw, [])
        s = random.randint(1, max(1, n - WIN))
        e = min(n, s + WIN)
        if overlaps(s, e, intervals):
            continue
        snippet = "\n".join(lines[s-1:e])
        if len(snippet.strip()) < 20:
            continue
        neg_records.append({
            "text": snippet, "label": 0, "smell": "None",
            "raw_url": raw, "start": s, "end": e, "n_lines": n,
        })

print("Negative snippets sampled:", len(neg_records))

all_records = records + neg_records
out_df = pd.DataFrame(all_records)
out_df.to_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier2_dataset.jsonl"), orient="records", lines=True)
print("Total tier-2 instances:", len(out_df))
print(out_df["label"].value_counts())
