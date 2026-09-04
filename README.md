# Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning

Code for the paper [*Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning*](https://arxiv.org/abs/2609.03430) (arXiv:2609.03430).

**Random Attention** (`random_pp` in the code) is a signal-free KV-cache eviction policy for reasoning models:
keep the prompt, then keep a *uniformly random* per-KV-head subset of the generated tokens up to the budget
(plus a short recency window). It reads no attention scores, no value statistics, and no calibration data, so
an eviction round costs only the compaction itself. On MATH-500, GPQA-Diamond, AIME, HMMT and LiveCodeBench, at
matched budgets, it matches or beats learned selectors (SnapKV, R-KV, VaSE, TriAttention) on Qwen3-4B/14B/32B
and Phi-4-reasoning, and it is the fastest evictor in both the Hugging Face harness and a vLLM serving stack.
The repository contains the eviction engine, the evaluation harness, the significance tests, the efficiency
benchmarks, the vLLM port, and the mechanism-study tooling that produced every number in the paper.

## Layout

```
kvcompress/engine/     KV-eviction engine: every method in the paper as an eviction mode (cache_utils.py),
                       evict-attention forwards for Qwen3 / phi3 / llama, faithful TriAttention scorer, engine tests
kvcompress/harness/    evaluation harness (adapted from VaSE, see THIRD_PARTY_NOTICES.md): eval_hf.py, the sharded
                       multi-worker launcher parallel_run_hf_mw.py, graders' utilities, TriAttention calibration + stats
kvcompress/eval/       grading (math/science, LiveCodeBench), cell-integrity audit, paired significance tests,
                       shard repair, LaTeX table generator
kvcompress/analysis/   mechanism studies: retention logs, fork replay/autopsy, carrier mass, eviction timing bench
kvcompress/synth/      the controlled synthetic-retrieval study (registered protocol, tables)
scripts/run_cell.sh    canonical launcher for one (model, task, method, budget) accuracy cell
scripts/grade_cell.sh  integrity check + grading of a cell
scripts/efficiency/    HF throughput protocols (VaSE fixed-batch, iso-memory, max-batch, eviction-round timing)
scripts/vllm_rp_bench/ Random Attention inside TriAttention's vLLM 0.19 runtime (own README + RUNBOOK)
scripts/mechanism/     retention-log panels and fork-replay launchers
figures/               figure scripts (read the graded TSVs)
data/                  benchmark layout + our LiveCodeBench difficulty subsets (data/README.md)
```

## Setup

```bash
git clone https://github.com/SalesforceAIResearch/Random-Attention && cd Random-Attention
bash setup.sh                 # Python 3.10 venv: torch 2.4.0 (cu121), flash-attn 2.7.3, transformers 5.0.0, ...
. env.sh                      # RA_ROOT / RA_ENGINE / RA_DATA_DIR / RA_MODELS_DIR / PYTHONPATH
```

* **Models**: Hugging Face checkpoints under `$RA_MODELS_DIR/<name>` (default `models/`): `Qwen3-4B`,
  `Qwen3-14B`, `Qwen3-32B`, `phi-4-reasoning`, optionally `DeepSeek-R1-Distill-Llama-8B`.
* **Data**: `$RA_DATA_DIR/<task>/test.jsonl` -- see `data/README.md` for the format and sources. Our
  LiveCodeBench subsets are included.
* **Hardware**: the paper's experiments ran on 8x H200 (141 GB).

## Run an accuracy cell

```bash
scripts/run_cell.sh Qwen3-4B math random_pp          # K defaults to the task's ~4x point (1024 here)
scripts/run_cell.sh Qwen3-4B math vase               # VaSE with the faithful n_large = K/4
scripts/run_cell.sh Qwen3-4B math triattn            # TriAttention with per-model calibration stats
scripts/run_cell.sh phi-4-reasoning gpqa snapkv 2048
scripts/run_cell.sh Qwen3-32B aime25 rkv && scripts/run_cell.sh Qwen3-32B aime26 rkv
```

Add `--dry-run` to print the exact `parallel_run_hf_mw.py` command. Cells resume when re-launched; completions
land under `results/<model>/<task>_K<K>/<method>/`. the header of `scripts/run_cell.sh` lists every method and its flags.

## Grade, test, tabulate

```bash
scripts/grade_cell.sh Qwen3-4B math_K1024 dense,random_pp,attn,attn_rkv_l05,vase_faithful,triattn_ph_memofix
python kvcompress/eval/stats_paired.py --base results/Qwen3-4B/math_K1024 --data_name math \
       --method_a random_pp --methods_b attn,attn_rkv_l05,vase_faithful,triattn_ph_memofix   # paired bootstrap + sign test
python kvcompress/eval/gen_paper_tables.py --results <graded.tsv> --lcb <lcb.tsv> --out tables/
```

`grade_cell.sh` runs `cell_integrity.py` first: grading is positional (problem = shard offset + line), so a
mixed, ragged or overlapping cell is refused rather than silently mis-graded. Accuracy is `flag_acc`
(answer correct); `acc_strict` additionally requires termination within the 32k cap and is reported as a
diagnostic.

## Efficiency

* Hugging Face harness: `scripts/efficiency/*.sh` (fixed-batch VaSE protocol, iso-memory max-batch serving,
  eviction-round timing). Never report tokens/s from the batched accuracy runs -- they share GPUs.
* vLLM serving: `scripts/vllm_rp_bench/` -- Random Attention as a selector inside TriAttention's vLLM 0.19
  runtime (their paged-KV compression machinery, our selection rule), with the accuracy-transfer check and the
  runtime shims documented in its README.

## Mechanism and synthetic studies

`KEEPLOG=1` retention logging, `FORCE_KEEP_RANGE` fork replays (`kvcompress/analysis/fork_replay.py`,
`fork_autopsy.py`), carrier-head mixing modes, and the registered synthetic-retrieval protocol
(`kvcompress/synth/`) are driven by engine environment switches documented in the `kvcompress/engine/cache_utils.py` header.

## License

This project is released under the Apache License 2.0 (see `LICENSE.txt`); it is a research release accompanying an academic
paper -- please read `AI_ETHICS.md`. Contributions: `CONTRIBUTING.md`; security reports: `SECURITY.md`.

## Acknowledgements and third-party code

The evaluation harness and the engine skeleton descend from [VaSE](https://github.com/terarachang/VaSE) (MIT);
the TriAttention baseline and the vLLM serving benchmark build on
[TriAttention](https://github.com/WeianMao/triattention) (Apache-2.0). Exact file-level provenance and license texts are in `THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@article{randomattention2026,
  title         = {Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning},
  author        = {Heng Wang and Jielin Qiu and Wenting Zhao and Cheng Qian and Liangwei Yang and Jiawei Han and Heng Ji and Silvio Savarese and Shelby Heinecke and Huan Wang},
  journal       = {arXiv preprint arXiv:2609.03430},
  year          = {2026},
  eprint        = {2609.03430},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.03430}
}
```
