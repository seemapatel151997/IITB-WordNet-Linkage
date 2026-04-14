# WordNet Linkage (Indo ↔ English) — Embedding Retrieval

This is the public, portfolio-friendly version of my work on **cross-lingual WordNet linkage**: aligning Indo-language synsets to English synsets by treating the problem as **semantic retrieval**.

The original work was completed in an institutional research setting; this repository is intentionally curated so it can be published safely:
- no institute databases, credentials, or internal schemas
- no redistributable WordNet dumps
- a small, runnable demo that illustrates the method end-to-end

## Problem (in one paragraph)

Given an Indo-language synset (gloss + lemmas), retrieve the best matching English synset from a large English WordNet. I model linkage as **nearest-neighbor search in an embedding space**: encode synset texts with a multilingual model and rank English candidates by cosine similarity.

## What’s in this repo

- `demo/`: a small, self-contained demo dataset + scripts (no DB required)
- `docs/ARCHITECTURE.md`: high-level system diagram and framing
- `docs/CASE_STUDY.md`: methodology, design choices, evaluation protocol, and improvements roadmap
- `docs/DATA_SOURCES.md`: how to use real WordNets locally without publishing them
- `docs/SAFE_PUBLISHING_CHECKLIST.md`: checklist before making a repo public

## How it works (high level)

- Represent each synset using either:
  - gloss only, lemmas only, or gloss + lemmas
- Encode texts with a multilingual sentence embedding model (MuRIL)
- Retrieve top‑k English candidates using cosine similarity
- Optionally enforce part-of-speech matching (noun/verb/adjective/adverb)
- Evaluate using retrieval metrics (top‑k, MRR) when gold mappings exist

## Quickstart (toy demo, runs locally)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python demo/demo_linkage.py --top-k 5
python demo/evaluate_demo.py demo/demo_results.csv --k 5
```

You’ll get:
- `demo/demo_results.csv`: ranked English candidates per Indo synset + cosine scores
- a printed metric summary for the toy gold mappings

## What I’d do next (if you want to extend this)

- Replace brute-force cosine search with ANN indexing (FAISS/HNSW) for full-scale WordNets
- Add a cross-encoder re-ranker on the top‑N candidates
- Add better text normalization/tokenization per language and for lemma lists

If you’re reviewing this as a hiring manager: `docs/CASE_STUDY.md` is the best “walkthrough” of the engineering and the NLP decisions.



© 2026 [Your Name]. All rights reserved. View and learn from this code, but do not copy, use, or modify it without permission.
