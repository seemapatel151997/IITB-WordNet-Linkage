# Case study: cross-lingual WordNet linkage as retrieval

## What I built

A small but scalable baseline for **linking synsets across WordNets** using modern multilingual embeddings.

The key engineering decision was to treat linkage as a **retrieval problem** rather than a hard classifier:
- Query: Indo-language synset representation (gloss + lemmas)
- Corpus: English synset representations
- Ranking: cosine similarity in embedding space
- Output: top‑k candidates (usable for review or a downstream re-ranker)

This write-up focuses on the *method and design choices* and avoids any institute-specific datasets or internal database details.

## Why retrieval is a good fit

In practice, linkage is rarely “obvious” from surface forms alone:
- transliteration/orthography differences
- polysemy (same lemma, different senses)
- gloss style differences (short definition vs descriptive text)

Retrieval gives you:
- a ranked list of candidates (not just a single guess)
- a clean path to improve quality later (filters, re-ranking, ANN indexing)

## Representation choices (what gets embedded)

I structured experiments around three representations that can be swapped without changing the retrieval core:

1) **Gloss-only**
- Captures semantic meaning explicitly.
- Often more discriminative for abstract concepts.

2) **Lemma-only (synset words)**
- Works well for concrete entities and stable lexicalizations.
- Can be noisy when lemma lists are short/ambiguous.

3) **Gloss + lemmas**
- Best default: combines definitional and lexical cues.
- Simple concatenation works surprisingly well as a baseline.

## Model choice

I used MuRIL (via Sentence-Transformers) because it is designed for Indian languages + English, which makes it a strong off-the-shelf encoder for cross-lingual similarity.

The important part is not “MuRIL specifically”, but the interface:
- any sentence embedding model can be swapped in later
- the pipeline stays the same: encode → cosine → rank

## Retrieval and scoring

- English synsets are encoded once and reused (cacheable).
- Each Indo synset becomes a query vector.
- Similarity: cosine similarity between query and all English vectors.
- Candidate selection: top‑k by similarity score.

### POS-aware retrieval

When part-of-speech is available and reliable, restricting retrieval to the same POS reduces false positives (noun→noun, verb→verb, etc.). This is a simple, high-leverage constraint.

## Evaluation protocol (when gold links exist)

This project uses retrieval metrics:

- **Top‑k accuracy (recall@k)**: does the gold English synset appear in the top‑k list?
- **Top‑1 accuracy**: is the best-ranked candidate the gold link?
- **MRR** (mean reciprocal rank): rewards putting the correct answer near the top.

Why these metrics:
- linkage is operationally a ranking problem
- they’re stable and easy to compare across variants

## Engineering improvements that matter at scale

If you move from “toy” to full WordNets, two things dominate:

1) **Caching**
- Encode English once.
- Persist embeddings and only recompute when input changes.

2) **Indexing**
- Brute-force cosine scales linearly with number of English synsets.
- ANN (FAISS/HNSW) is the next step for interactive performance.

## Limitations (honest)

- Embedding similarity can miss fine-grained sense distinctions.
- Glosses are inconsistent across resources; normalization matters.
- Without a gold mapping set, you can’t claim accuracy—only generate candidates for review.

## Roadmap

If I continue this:
- Add a cross-encoder re-ranker on the top‑N candidates.
- Add language-aware normalization (punctuation, tokenization, lemma formatting).
- Add abstention thresholds (“no-link” option) based on score gaps or calibrated confidence.

