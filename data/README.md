# Benchmark data

The harness reads `$RA_DATA_DIR/<task>/test.jsonl` (default `data/`). One JSON object per line with the
fields the VaSE evaluation harness uses (`problem`/`question`, `answer` or `solution`, plus task-specific
fields); the loaders are in `kvcompress/harness/Utils/data_loader.py` and the answer extraction per task in
`kvcompress/harness/Utils/parser.py`.

| task key | benchmark | n | source |
|---|---|---|---|
| `math` | MATH-500 | 500 | `HuggingFaceH4/MATH-500` |
| `gpqa` | GPQA-Diamond | 198 | `Idavidrein/gpqa` (gated; accept the terms on the Hub) |
| `aime25` | AIME 2025 (I+II) | 30 | `{"problem","answer","id"}` per line |
| `aime26` | AIME 2026 (I+II) | 30 | same format |
| `hmmt` | HMMT 2025 (Feb + Nov) | 60 | `MathArena/hmmt_feb_2025` + `MathArena/hmmt_nov_2025` (VaSE's `convert_hmmt25.py`) |
| `livecodebench` | LiveCodeBench v6 (medium, 383 problems) | 383 | `livecodebench/code_generation_lite`, tests fetched by `data/lcb_subsets/download_tests.py` |
| `livecodebench_{easy,hard,medium}` | our difficulty subsets | 322 / 350 / 383 | `data/lcb_subsets/*.test.jsonl` (tracked); tests via `data/lcb_subsets/download_tests_subset.py` |

We used the `test.jsonl` files distributed with the VaSE evaluation harness
(https://github.com/terarachang/VaSE, `eval/reasoning_tasks/data/`, commit bfa2692), which are in exactly this
format; copying that directory here is the fastest way to reproduce. We do not redistribute them (GPQA in
particular is gated).

LiveCodeBench tests: `livecodebench_tests/` (one JSON per problem, ~2-3 GB) must sit next to each subset's
`test.jsonl`; `download_tests_subset.py` builds them with the same pipeline as VaSE's `download_tests.py`.
