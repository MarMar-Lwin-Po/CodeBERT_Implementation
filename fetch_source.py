import pandas as pd
import re
import requests
import time
import json
import os

DATA_PATH = "https://raw.githubusercontent.com/iSMELL2024/iSMELL/main/CodeDetection/updated_dataset.xlsx"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(OUT_DIR, "raw_file_cache.json")

df = pd.read_excel(DATA_PATH)
print("Total rows:", len(df))

LINK_RE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/blob/([0-9a-f]+)/(.+?)(?:#L(\d+)(?:-L(\d+))?)?$")

def parse_link(link):
    m = LINK_RE.match(str(link).strip())
    if not m:
        return None
    owner, repo, commit, path, l1, l2 = m.groups()
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
    return raw_url

df["raw_url"] = df["Link"].apply(parse_link)
n_unparsed = df["raw_url"].isna().sum()
print("Unparsed links:", n_unparsed, "/", len(df))
print(df[df["raw_url"].isna()][["Link"]].head(5))

unique_urls = df["raw_url"].dropna().unique()
print("Unique files to fetch:", len(unique_urls))

# Load cache if exists
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print("Loaded cache with", len(cache), "entries")

session = requests.Session()
ok, fail = 0, 0
for i, url in enumerate(unique_urls):
    if url in cache:
        continue
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            cache[url] = r.text
            ok += 1
        else:
            cache[url] = None
            fail += 1
    except Exception as e:
        cache[url] = None
        fail += 1
    if (i + 1) % 50 == 0:
        print(f"  fetched {i+1}/{len(unique_urls)} (ok={ok} fail={fail})")
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

with open(CACHE_FILE, "w") as f:
    json.dump(cache, f)
print(f"Done. ok={ok} fail={fail} cached_total={len(cache)}")
