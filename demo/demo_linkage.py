from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_text(gloss: str, synset_words: str) -> str:
    gloss = (gloss or "").strip()
    words = (synset_words or "").strip()
    return (gloss + " " + words).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    english_rows = read_csv(base / "toy_english_synsets.csv")
    indo_rows = read_csv(base / "toy_indo_synsets.csv")

    # POS bucket English rows
    english_by_pos: dict[str, list[dict[str, str]]] = {}
    for r in english_rows:
        pos = (r["category"] or "").strip().lower()
        english_by_pos.setdefault(pos, []).append(r)

    model = SentenceTransformer(args.model)
    try:
        model.to(args.device)
    except Exception:
        pass

    # Precompute English embeddings per POS
    eng_embeddings: dict[str, torch.Tensor] = {}
    eng_texts: dict[str, list[str]] = {}
    eng_ids: dict[str, list[int]] = {}

    for pos, rows in english_by_pos.items():
        texts = [build_text(r["gloss"], r["synset_words"]) for r in rows]
        ids = [int(r["synset_id"]) for r in rows]
        emb = model.encode(texts, convert_to_tensor=True, device=args.device)
        eng_embeddings[pos] = emb
        eng_texts[pos] = texts
        eng_ids[pos] = ids

    out_path = base / "demo_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "indo_id",
                "indo_pos",
                "gold_english_id",
                "retrieved_rank",
                "retrieved_english_id",
                "score",
                "query_text",
                "retrieved_text",
            ],
        )
        writer.writeheader()

        for r in indo_rows:
            pos = (r["category"] or "").strip().lower()
            if pos not in eng_embeddings:
                continue
            query_text = build_text(r["gloss"], r["synset_words"])
            q = model.encode(query_text, convert_to_tensor=True, device=args.device)
            scores = util.cos_sim(q, eng_embeddings[pos])[0]
            k = min(args.top_k, scores.shape[0])
            top_scores, top_idxs = torch.topk(scores, k=k)

            for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idxs.tolist()), start=1):
                writer.writerow(
                    {
                        "indo_id": int(r["indo_id"]),
                        "indo_pos": pos,
                        "gold_english_id": int(r["gold_english_id"]),
                        "retrieved_rank": rank,
                        "retrieved_english_id": eng_ids[pos][idx],
                        "score": float(score),
                        "query_text": query_text,
                        "retrieved_text": eng_texts[pos][idx],
                    }
                )

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

