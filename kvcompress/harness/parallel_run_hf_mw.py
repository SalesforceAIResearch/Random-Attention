#!/usr/bin/env python3
# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
"""Sharded multi-worker launcher for the accuracy cells (Random Attention).

Descends from VaSE's eval/reasoning_tasks/parallel_run_hf.py (https://github.com/terarachang/VaSE, MIT) but is
substantially rewritten: NUM_SHARDS positional data sharding (fixed per cell; the graders index
problems positionally), WORKERS_PER_GPU rank reuse, SHARD_SUBSET cross-node splitting, per-task
batch/rollout overrides, and the faithfulness marker (nl=K/4) in the output directory name.
Spawns one eval_hf.py process per (rollout batch, shard); resume-safe via --use_batch_exist.
See docs/METHODS.md for the per-method flags and scripts/run_cell.sh for the canonical launcher.
"""
import subprocess
import os
import sys
import argparse
import time
from collections import deque # Use deque for efficient pop/append
from my_utils import *


def choose_task_config(model_size):
    # bs/total_run are overridable per-run via --batch_size/--total_run; config is model-size-agnostic
    # (32B included -- it just runs at WPG=1). n_examples must match the dataset.
    task_config = {
        "aime24":        {"bs": 2,   "total_run": 8,  "n_examples": 30},
        "aime25":        {"bs": 2,   "total_run": 8,  "n_examples": 29},  # FROZEN CONVENTION (decided
        # 2026-07-25): dataset has 30 problems; this original typo means EVERY aime25 cell in the
        # project ran the same 29 (problem id 29 excluded). Matched across methods => comparisons
        # valid. DO NOT change to 30 — it would shard-misalign every existing cell's resume.
        "aime26":        {"bs": 2,   "total_run": 8,  "n_examples": 30},
        "hmmt":          {"bs": 16,  "total_run": 16, "n_examples": 60},
        "math":          {"bs": 2,  "total_run": 2,  "n_examples": 500},
        "math_l5":       {"bs": 4,  "total_run": 4,  "n_examples": 134},
        "gpqa":          {"bs": 4,  "total_run": 4,  "n_examples": 198},
        "livecodebench": {"bs": 8,  "total_run": 8,  "n_examples": 383},
        "livecodebench_hard": {"bs": 4,  "total_run": 4,  "n_examples": 350},
        "livecodebench_easy": {"bs": 4,  "total_run": 4,  "n_examples": 322},
        "livecodebench_sub120": {"bs": 2,  "total_run": 2,  "n_examples": 120},
        "livecodebench_r263": {"bs": 4,  "total_run": 4,  "n_examples": 263},
    }
    return task_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks using subprocess.")
    parser.add_argument("--model_dir", type=str,
                        default="Qwen/Qwen3-14B",
                        help="Model directory path")
    parser.add_argument("--model_size", type=str, default="14B", help="model_size")
    parser.add_argument("--tasks", type=str, default="aime",
                        help="Comma-separated list of tasks (e.g., aime,math,gpqa)")
    parser.add_argument("--output_dir", type=str, default="./results/aime",
                        help="Directory to store output results")
    parser.add_argument("--attention_implementation", type=str, default="eager",
                        help="attention implementations")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit for the number of samples to process")
    parser.add_argument("--num_gpus", default="8", type=int)
    parser.add_argument("--sparsity_method", default='dense', type=str)
    parser.add_argument("--token_budget", default="2048", type=str)
    parser.add_argument("--max_tokens", default="32768", type=str)
    parser.add_argument("--run_id", default=0, type=int)
    parser.add_argument("--run_end_id", type=int)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional override of the per-task batch size (rollout samples batched "
                             "per problem). Defaults to task_config bs when unset. Must divide total_run.")
    parser.add_argument("--total_run", type=int, default=None,
                        help="Optional override of task_config total_run (rollouts/problem; e.g. 1 for a fast probe). "
                             "Default uses config. Additive: existing calls unaffected.")
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose information or not")
    args, _ = parser.parse_known_args()
    parser = expand_parser_for_methods(parser, args.sparsity_method)
    args = parser.parse_args()

    limit = args.limit
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens

    model_dir = args.model_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sparsity_method = args.sparsity_method
    token_budgets = [t.strip() for t in args.token_budget.split(",") if t.strip()]

    output_dir = args.output_dir
    attention_implementation = args.attention_implementation

    task_config = choose_task_config(args.model_size)

    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        bs = args.batch_size if args.batch_size is not None else task_config[task]["bs"]
        total_run = args.total_run if args.total_run is not None else task_config[task]["total_run"]
        if args.run_end_id is None:
            batched_run = total_run // bs
        else:
            batched_run = args.run_end_id // bs

        print(f"\n{'='*40}")
        print(f"Starting task: {task}")
        print(f"Batch size: {bs} | total_run: {total_run}")

        if sparsity_method == "eviction":
            param_combinations = [(tb,) for tb in token_budgets]
        elif sparsity_method == "dense":
            param_combinations = [()]
        else:
            raise ValueError(f"Unknown sparsity_method: {sparsity_method}")

        for params in param_combinations:
            if sparsity_method == "eviction":
                (token_budget,) = params
                param_desc = f"budget={token_budget}, {args.eviction_mode}"
                cli_params = [
                    "--token_budget", str(token_budget),
                    "--residual_length", str(args.residual_length),
                    "--eviction_mode", args.eviction_mode,
                    "--n_large", str(args.n_large),
                ]
                if (('range' in args.eviction_mode or
                     args.eviction_mode in ('mix_step_pp', 'mix_head_pp', 'union_topk_pp'))
                        and args.eviction_mode != 'evict_range_cur'):
                    param_desc += f", nl={args.n_large}"   # nl in dir name = faithfulness marker (nl=K/4)
                if args.rkv_lambda:
                    cli_params += ["--rkv_lambda", str(args.rkv_lambda)]
                    param_desc += f", lambda={args.rkv_lambda}"
                if args.smooth:
                    cli_params.append("--smooth")
                    param_desc += ", smooth"
                if args.temperature != 1.0:
                    cli_params += ["--temperature", str(args.temperature)]
                    param_desc += f", temp={args.temperature}"
                if args.verbose:
                    cli_params.append("--verbose")
            elif sparsity_method == "dense":
                param_desc = "dense"
                cli_params = []

            print(f"\n{'─'*30}")
            print(f"Processing Task:{task} | {sparsity_method}: {param_desc}")

            active_procs = {}
            import os as _os
            _wpg = int(_os.environ.get('WORKERS_PER_GPU','1'))
            available_gpus = deque(list(range(num_gpus)) * _wpg)  # node3: N workers/physical-GPU via rank reuse

            output_config_subdir = os.path.join(output_dir, f"{task}_bs{bs}_{param_desc.replace(', ', '_')}")

            # Build job queue over (rollout_id, data_shard) pairs
            num_data_parallel = int(_os.environ.get('NUM_SHARDS', num_gpus))  # node3: NUM_SHARDS>num_gpus => more jobs so multi-worker (available_gpus*WPG) has work to fill; layout fixed by NUM_SHARDS (WPG tunable/resume-safe)
            n_examples = task_config[task]["n_examples"]
            shard_size = n_examples // num_data_parallel

            job_queue = deque()
            # SHARD_SUBSET: run only these shard ids on THIS node, e.g. "0-12" or "0,3,7".
            # NUM_SHARDS still defines the boundaries, so ex_start_i -- and therefore the
            # completions_shard<ex_start>.jsonl filenames and the positional gi used by the
            # graders -- are IDENTICAL to what a single node running the whole cell would write.
            # Splitting a cell across nodes is therefore layout-neutral, UNLIKE changing
            # NUM_SHARDS, which moves every boundary and silently corrupts an existing cell.
            # The subsets MUST be disjoint: two nodes given the same shard id would race on the
            # same file and produce overlap corruption.
            _sub = _os.environ.get('SHARD_SUBSET', '').strip()
            if _sub:
                _sel = set()
                for _p in _sub.split(','):
                    _p = _p.strip()
                    if not _p:
                        continue
                    if '-' in _p:
                        _a, _b = _p.split('-')
                        _sel.update(range(int(_a), int(_b) + 1))
                    else:
                        _sel.add(int(_p))
                _shard_ids = sorted(i for i in _sel if 0 <= i < num_data_parallel)
                if not _shard_ids:
                    print(f"ABORT: SHARD_SUBSET={_sub} selects no shard in range(0,{num_data_parallel})")
                    sys.exit(5)
                print(f"SHARD_SUBSET={_sub} -> this node runs shards {_shard_ids} of {num_data_parallel}")
            else:
                _shard_ids = list(range(num_data_parallel))

            run_counter_start = args.run_id // bs
            for run_i in range(run_counter_start, batched_run):
                current_run_id = run_i * bs
                for shard_id in _shard_ids:
                    ex_start_i = shard_id * shard_size
                    ex_end_i = (shard_id + 1) * shard_size if shard_id < num_data_parallel - 1 else n_examples
                    job_queue.append({
                        "run_id": current_run_id,
                        "ex_start_i": ex_start_i,
                        "ex_end_i": ex_end_i,
                    })

            while job_queue or active_procs:
                for proc, info in list(active_procs.items()):
                    if proc.poll() is not None:
                        print(f"Run {info['run_id']} shard ex{info['ex_start_i']} on GPU {info['gpu_id']} finished.")
                        available_gpus.append(info['gpu_id'])
                        del active_procs[proc]

                while job_queue and available_gpus:
                    gpu_id = available_gpus.popleft()
                    job = job_queue.popleft()
                    current_run_id = job["run_id"]
                    ex_start_i = job["ex_start_i"]
                    ex_end_i = job["ex_end_i"]

                    print(f"Launching run {current_run_id} shard ex{ex_start_i}:{ex_end_i} on GPU {gpu_id}...")

                    env = os.environ.copy()
                    _here = os.path.dirname(os.path.abspath(__file__))
                    _root = os.path.dirname(os.path.dirname(_here))
                    env["PYTHONPATH"] = _root + os.pathsep + _here + os.pathsep + env.get("PYTHONPATH", "")
                    cmd = [
                        sys.executable, os.path.join(_here, "eval_hf.py"),
                        "--model_name_or_path", model_dir,
                        "--data_name", task,
                        "--batch_size", str(bs),
                        "--output_dir", output_config_subdir,
                        "--attention_implementation", attention_implementation,
                        "--use_batch_exist",
                        "--surround_with_messages",
                        "--rank", str(gpu_id),
                        "--sparsity_method", sparsity_method,
                        "--run_id", str(current_run_id),
                        "--max_tokens", str(max_tokens),
                    ] + cli_params
                    if num_gpus > 1:
                        cmd += ["--ex_start_i", str(ex_start_i), "--ex_end_i", str(ex_end_i)]

                    proc = subprocess.Popen(cmd, env=env)
                    active_procs[proc] = {"gpu_id": gpu_id, "run_id": current_run_id, "ex_start_i": ex_start_i}

                if (job_queue and not available_gpus) or (not job_queue and active_procs):
                    time.sleep(5)

            # Grading is a separate, positional step: kvcompress/eval/grader.py (math/science) or
            # kvcompress/eval/grader_lcb.py (code). See scripts/grade_cell.sh.
        print(f"\nCompleted: {task}")

    print("\n All tasks and configurations completed!")
