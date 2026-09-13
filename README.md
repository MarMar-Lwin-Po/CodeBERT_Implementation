# CodeBERT Detector — Implementation Log

Built in response to supervisor guidance: "do the (full if possible) replication of the
iSMELL paper and some CodeBERT implementation first." This log is an honest record of
what ran, what it found, and one real infrastructure blocker — not a polished write-up.

## Data source

`/tmp/thesis_proposal/dataset_check/updated_dataset.xlsx`, downloaded directly from
iSMELL's public GitHub replication package (`iSMELL2024/iSMELL`). 654 rows, matching
the paper's published Table 2 totals exactly (196 God Class, 248 Refused Bequest,
210 Feature Envy; 285 positive / 369 negative).

## Tier 1 — fast pilot (COMPLETE, real results)

`tier1_pilot.py` trains directly on the file's precomputed 768-dim `codebert_vector`
column (a frozen CodeBERT embedding already included for every one of the 654 rows).
No network dependency beyond the initial file download. Stratified 80/20 split by
Smell x Label.

Results (`tier1_results.json`), held-out test set, n=131:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.740 | 0.689 | 0.737 | 0.712 |
| MLP (128,64) | 0.740 | 0.725 | 0.649 | 0.685 |

Per-smell F1 (Logistic Regression): God Class 0.889, Feature Envy 0.739, Refused
Bequest 0.370 — Refused Bequest is markedly harder for both models, worth flagging
in the proposal's expected-results / threats discussion rather than treating the
overall F1 as representative of every smell type.

## Important data-quality finding

Checking the `Link` column directly (not assuming from the paper text) shows only
119 of 654 rows carry a real, resolvable GitHub source-code link
(`https://github.com/.../blob/<commit>/<path>#L<start>-L<end>`). The other 535 rows'
`Link` field is just a Java package name (e.g. `org.webcurator.core.targets`), not a
retrievable pointer to code.

The 119 resolvable rows line up exactly with the Palomba et al. subset from Table 2
(94 God Class + 25 Feature Envy = 119) — and Fontana et al.'s 185 Feature Envy and
Khomh et al.'s 350 (102 God Class + 248 Refused Bequest) instances are the ones
without resolvable links. A second finding: **all 119 resolvable rows are
Label = 1** (smell present) — there are zero labeled negatives with retrievable
source in this file.

Practical implication: true end-to-end CodeBERT fine-tuning (tokenizing and
backpropagating through real source code, per Naz et al.'s actual protocol) is only
directly supported by this file for the Palomba subset, and needs a constructed
negative set since none exist in that subset. It is not currently supported for the
Fontana/Khomh instances (535 of 654) without separately locating their own
replication packages — worth a quick check before final scoping.

## Tier 2 — true fine-tuning: data ready, execution blocked here

`fetch_source.py` resolved and downloaded all 116 unique source files behind the 119
Palomba-subset links (cached in `raw_file_cache.json`, 10MB).

`extract_tier2.py` built a genuinely balanced dataset: the 119 real positive
snippets, plus 119 negative snippets sampled as 60-line windows from the *same*
source files, at offsets that do not overlap any labeled positive region — a
standard, defensible way to construct negatives when a corpus provides positives
only. Result: `tier2_dataset.jsonl`, 238 instances, 119/119 balanced.

`tier2_finetune.py` is a complete, ready-to-run fine-tuning script (CodeBERT
tokenizer + `RobertaForSequenceClassification` on `microsoft/codebert-base`,
3 epochs, CPU-compatible, batch size 8, max length 256).

**It has not produced results yet.** This sandboxed session's network egress policy
blocks Hugging Face Hub — `huggingface.co`, `cdn-lfs.huggingface.co`, and
`hf-mirror.com` all return HTTP 403 here (confirmed directly, not assumed), which is
where `microsoft/codebert-base`'s pretrained weights are hosted. Several other model/
data hosts (zenodo.org, storage.googleapis.com, s3.amazonaws.com) are blocked the same
way. This is specific to this cloud sandbox's allowlist, not a real research
feasibility problem.

## What to do with this

The script and data are complete and portable — run `tier2_finetune.py` as-is on any
machine with normal internet access (your own laptop, Google Colab, or university
compute) and it will download CodeBERT and fine-tune immediately. That's exactly the
"have the resource ready when you need it" your supervisor asked for: the data
extraction, negative-sampling, and training code are done now, so the only remaining
step elsewhere is running it.

## Files

- `tier1_pilot.py`, `tier1_results.json` — Tier 1, already run, real results above.
- `fetch_source.py`, `raw_file_cache.json` — source-code retrieval for the 119
  resolvable instances.
- `extract_tier2.py`, `tier2_dataset.jsonl` — balanced 238-instance real-code dataset.
- `tier2_finetune.py`, `tier2_log.txt` — fine-tuning script, ready to run; log shows
  the run up to the point HF Hub access failed.
