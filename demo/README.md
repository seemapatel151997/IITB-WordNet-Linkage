# Demo

This demo runs the same *idea* as the full project without requiring any external database:

- English synsets: tiny toy set
- Indo synsets: tiny toy set (Hindi-like examples)
- Gold mapping: included for evaluation

Run:
```bash
python demo_linkage.py --top-k 5
python evaluate_demo.py demo_results.csv --k 5
```

Output:
- `demo_results.csv`: per query, ranked English candidates + cosine scores.

Tip: open `demo_results.csv` in a spreadsheet to review how candidates are ranked.
