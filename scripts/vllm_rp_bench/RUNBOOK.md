# Tier-2 efficiency bench: rp vs TriAttention INSIDE their vLLM stack (one H200, ~1 GPU-day)

Scoped by the a87b8a agent audit (08-07) against their checkout `9c26d14`. Everything here was
traced to file:line in that audit; re-verify flag spellings with `--help` on the node (vLLM CLI
arg names drift). These numbers are EFFICIENCY-ONLY under THEIR budget semantics (prompt counted,
window inside budget) — never mixed with our accuracy tables.

## Why this exists
Answers "your harness has no PagedAttention" directly: both methods measured inside their fused
serving stack. Prediction (registered): tri's overhead fraction GROWS here vs our harness (shared
costs shrink, scoring cost doesn't); rp's stays ~0. rp is faster at compression boundaries because
tri must gather+score every key of every layer per event while rp touches no KV.

## Steps
1. VENV (GPU node, python3.12, NEVER .venv_vase; NVMe per the cold-node fix):
   see setup_venv.sh. Pin `vllm==0.19.0` (their monkeypatch straddles that V1 surface; activation
   re-raises on mismatch, so the smoke step catches it immediately). Install their checkout COPY
   with `--no-deps`; never the `[eval]` extra (latex2sympy family — the venv-breaker).
2. PATCH: copy selector_random.py into `<copy>/triattention/vllm/runtime/`; apply hook_impl.patch
   (env-var branch `TRIATTN_RUNTIME_SELECTOR=random_pp`, which BYPASSES build_triattention_selector
   so the rp arm needs no stats file). Patch the CHECKOUT COPY — the hook runs in the EngineCore
   worker process, so driver-side monkeypatching does not propagate under V1 multiprocessing
   (fallback: VLLM_ENABLE_V1_MULTIPROCESSING=0).
3. SMOKE: expect "[TriAttention] Runtime (V2) plugin activated" + compression events with
   status=applied in logs; 3-5 fixed prompts per arm, temp 0, coherence eyeball only (ungraded).
4. TRI-ARM STATS GATE: torch.load our stats_pl/qwen3-14b.pt on the node and confirm metadata has
   inv_freq or a model_name their AutoConfig can resolve — else their converter silently falls back
   to rope_theta and degrades the TRI arm only (their own log warns "random token eviction").
5. BENCH (3 arms x 2 regimes; identical `--enforce-eager --no-enable-prefix-caching` everywhere —
   the off arm must be the EAGER baseline, never CUDA-graph):
   env (tri):  ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_DIVIDE_LENGTH=128
               TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1
               TRIATTN_RUNTIME_SPARSE_STATS_PATH=<stats_pl/qwen3-14b.pt>
   env (rp):   same minus stats, plus TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234
   env (off):  ENABLE_TRIATTENTION=0
   batch-1:    vllm bench latency  --model Qwen/Qwen3-14B --dtype bfloat16 --input-len 1024
               --output-len 8192 --batch-size 1 --num-iters 3 --num-iters-warmup 1
               --max-model-len 16384 --enforce-eager --no-enable-prefix-caching
   serving:    vllm bench throughput --model Qwen/Qwen3-14B --dtype bfloat16 --input-len 1024
               --output-len 8192 --num-prompts 64 --max-model-len 16384 --enforce-eager
               --no-enable-prefix-caching
   output-len 8192 >> budget 2048 is DELIBERATE (~47 compression events/request); shrinking it
   below budget makes all arms measure identical work and voids the bench.
6. COLLECT: output tok/s (headline), latency percentiles, peak GPU mem, post-reclaim kv_cache_usage
   from scheduler stats (KV-memory reduction is tri's real product — report it FOR tri, fairly).

## Registered gotchas
- PROTECT_PREFILL runtime default is FALSE (HF-side default is true) — must set =1 for _pp semantics.
- Their cadence is divide_length=128; per-event rp cost ~0.2-0.5ms across 40 layers => negligible;
  the precompute trick is NOT needed at this cadence (it matters in OUR harness at cadence 64 and
  as the deployment argument, not here).
- Seeding: per (request, round_start, layer) via blake2b — deterministic, no RNG-stream coupling.
- Report both arms' numbers with the off arm as the shared denominator.
