# vLLM-rp: Random Attention (random_pp) inside TriAttention's vLLM runtime

Release artifact for the paper's vLLM port of **Random Attention** (`random_pp`:
prompt-protected per-head random KV eviction). The selector runs inside TriAttention's
official vLLM 0.19 runtime (their paged-KV compression machinery, our selection rule),
so efficiency AND accuracy are measured in a real serving stack.

## Validated accuracy transfer (2026-08-18)

Qwen3-4B, MATH500, free generation to 32k, matched sampling vs the HF harness
(temperature 0.6, top_p 0.95, **top_k off**), n=500, single rollout:

| budget | vLLM (this port) | HF harness | delta |
|---|---|---|---|
| K=1024 | **0.864** | 0.874 | −1.0pp |
| K=2048 | **0.922** | 0.925 | −0.3pp |

Both runs carried an armed KV-reachability assert (`TRIATTN_ASSERT_REACHABLE=1`) and a
zero-violation census. Remaining deltas are within single-run noise plus the disclosed
semantics differences (their window=128 vs our HF residual=64; their divide-length event
cadence; ≤4 tokens of budget reserved by the boundary fix).

## What is patched and why (shims/)

The upstream runtime shipped three event-boundary defects that were invisible to every
throughput bench and fatal to accuracy (initial transfer: 0.18/0.30). Each shim is a
whole-file replacement of the same-named upstream file; the `.diff` beside it is the
exact delta against the pinned checkout.

1. **Block-reclaim orphan** (`hook_runtime_context.py`): each compression event compacted
   to exactly `budget`, so the in-flight token's KV landed in a freed block — one
   permanent garbage KV vector per event, force-kept by the recency window. Fix: reserve
   `scheduled_tokens + grace` (capped at block_size−1; block counts invariant).
2. **Overstated effective-offset** (`runner_compression_actions.py`,
   `kv_allocation_sync.py`, `scheduler.py`): the offset was registered as
   `live num_computed_tokens − cache_len_after`, mixing a schedule-time-advanced count
   with a pre-step length — overstated by 1 + async-lookahead depth, making every
   block-boundary allocation permanently late (one orphaned boundary write per
   compression span). Fix: anchor the offset to the event's own pre-step count
   (`scheduler_nct_pre`), carried in the event dict.
3. **Self-trigger barrier bypass** (`runner.py`): worker self-triggered events did not
   set the batch-queue boundary flag, letting the queue look ahead across an undrained
   event. Fix: set the same `_triattention_force_boundary_sync` side-channel flag the
   scheduler-side events use.
4. **Prefill no-op guard** (`hook_runtime_context.py`, part of the same diff as 1):
   kv_usage events can arrive with budget == current length == prefill (a keep-all
   no-op); the reserve must never push the budget below the protected prefill or the
   no-op turns into a strict-mode fatal.
   Plus: `state.py` (dedup livelock), `input_patch_vllm_v1_backend.py` (GPU-resident
   override pathway + the diagnostic reachability probe), `kv_group_resolver.py`
   (vllm-0.19 API drift).

## Quickstart

```bash
TRIATTN_SRC=/path/to/triattention bash setup_venv.sh   # GPU node; builds $VLLM_WORK (default ~/triattn_vllm)
                                # pins vllm==0.19.0, copies selector + ALL shims, installs
. ${VLLM_WORK:-$HOME/triattn_vllm}/venv/bin/activate
env ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 \
    TRIATTN_RUNTIME_KV_BUDGET=1024 TRIATTN_RUNTIME_DIVIDE_LENGTH=128 \
    TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 \
    python ../vllm_math_acc.py --runs 1 --top_k -1 --lo 0 --hi 500 --out <cell>
```
Efficiency benches (latency / capacity / long-decode): see `RUNBOOK.md` (unchanged
protocol; numbers in REPORT.md Tier-2 blocks).

## Env flags

| flag | meaning |
|---|---|
| `TRIATTN_RUNTIME_SELECTOR=random_pp` | activate our selector (bypasses stats files) |
| `TRIATTN_RUNTIME_RANDOM_SEED` | selection seed (default 1234) |
| `TRIATTN_RUNTIME_KV_BUDGET` / `DIVIDE_LENGTH` / `WINDOW_SIZE` / `PROTECT_PREFILL` | their runtime's knobs, matched to the paper cells |
| `TRIATTN_RESERVE_GRACE` | extra boundary-reserve tokens (default 3 = queue depth + 1) |
| `TRIATTN_ASSERT_REACHABLE=1` | arm the KV-reachability probe (counted log) |
| `TRIATTN_ASSERT_RAISE=1` | escalate probe hits to engine-fatal (diagnostics only) |

## Hard-won caveats — read before running

- **Pin `vllm==0.19.0`.** The runtime monkeypatches that exact V1 surface.
- **Preemption-unsafe.** The integration corrupts on vLLM preemption. Cap admitted
  sequences so the KV pool can never overflow: `--max_num_seqs 96` (4B on A100 / 32B on
  H200), `224` (14B on H200).
- **Never `--runs > 1` in one process.** Engine reuse cross-contaminates runs
  (measured: conclusion behavior collapses run-to-run). One engine per run.
- **Matched sampling = top_k OFF.** Explicit `SamplingParams` bypass
  `generation_config` defaults, and the HF batch_exist sampler applies only
  temperature+top_p. `top_k=20` is a stabilizer variant, not the matched condition.
- **`enforce_eager` + prefix caching off** everywhere (both arms) for comparability.
- Killing a runner can leave orphaned `EngineCore` processes holding GPU memory —
  kill by `nvidia-smi --query-compute-apps=pid` PIDs, not just the parent.

## Provenance

- Upstream: TriAttention official repo (https://github.com/WeianMao/triattention), checkout `9c26d14`; point `TRIATTN_SRC` at a clone.
- Fix forensics + full diagnosis arc: `REPORT.md` addendum 10 / 10b;
  ledger `paired_tests_20260802.txt` 08-18 addenda.
- Raw cells: `gate2/Qwen3-4B/math_K{1024,2048}/rp_vllm_v4/`.
