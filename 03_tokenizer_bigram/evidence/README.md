# `03_tokenizer_bigram/evidence/` — outputs that prove the module works

## What belongs here for Module 3

### `test_output.txt`
Output of `python test.py` showing all four BPE tests green:
- `test_roundtrip` — encode/decode is identity
- `test_parity_with_solution` — your merges match the reference
- `test_compression` — ratio ≥ 1.3× after 200 merges
- `test_structural_match_with_tiktoken` — both compress repeated English better than gibberish

Capture with `python test.py > evidence/test_output.txt 2>&1`.

### `bigram_metrics.json` (recommended)
Capture the bigram NLL after training:
```json
{
  "vocab_size": 65,
  "bigram_nll": 2.43,
  "uniform_baseline_nll": 4.17,
  "compression_after_200_merges": 1.96,
  "first_10_merges": ["(101, 32)→'e '", "(32, 116)→' t'", ...]
}
```

### `learned_merges.txt` (optional but illuminating)
The first 50 learned BPE merges decoded:
```
merge   0: (101, 32) -> 256 ('e ') x1834
merge   1: (32, 116) -> 257 (' t') x1502
...
```

Watching the merges land is the most direct way to *see* the algorithm learn — the first 10 are almost always tiny English bigrams, the next 20 are short words like 'the' 'and' 'to', and only after ~50 do you start seeing rarer patterns.

## Maintenance

Re-run `test.py` after any change to `bpe_starter.py` / `bpe_solution.py` and refresh these files.
