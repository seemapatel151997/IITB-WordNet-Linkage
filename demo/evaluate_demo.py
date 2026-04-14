from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    path = Path(args.csv_path)
    groups: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["indo_id"]), int(row["gold_english_id"]), (row["indo_pos"] or ""))
            groups[key].append(row)

    n = len(groups)
    top1 = 0
    topk = 0
    rr_sum = 0.0

    for (indo_id, gold_id, _pos), rows in groups.items():
        gold_rank = None
        for r in rows:
            if int(r["retrieved_english_id"]) == gold_id:
                rank = int(r["retrieved_rank"])
                gold_rank = rank if gold_rank is None else min(gold_rank, rank)

        if gold_rank == 1:
            top1 += 1
        if gold_rank is not None and gold_rank <= args.k:
            topk += 1
            rr_sum += 1.0 / gold_rank

    print(f"n={n} top1={top1/n:.4f} top{args.k}={topk/n:.4f} mrr={rr_sum/n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

