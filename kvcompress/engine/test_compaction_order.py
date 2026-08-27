# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test: keep-set semantics across repeated eviction rounds (the compaction-order bug class).

The engine compacts kept tokens in topk(score) order, not chronological order. Any mode whose score
is a function of the raw cache INDEX (recency_pp, the recency tail of random_pp_rec) therefore needs
a chronological sort of ids_to_keep before the gather, or from the 2nd eviction round onward "recent
by index" is not recent in time (verified failure: recency_pp kept the OLDEST of the previously-kept
block and dropped the newest).

This test drives the real EvictLayer._run_eviction with keys/values that ENCODE their absolute token
position, runs 3 eviction rounds, and asserts:
  1. recency_pp keeps exactly {prompt} U {true most-recent window} at every round (the 2026-07-08 fix);
  2. random_pp never loses a prompt token (protection survives compaction);
  3. every *_pp mode (incl. attn_pp/mixes/strat/cluster, 2026-07-09) keeps the prompt at every round;
  4. UNIVERSAL chronological compaction: every mode's cache is stored in strictly increasing time order;
  5. random_strat_pp covers one token per chunk (round 1, exact); random_cluster_pp keeps one contiguous
     window (round 1, exact);
  6. the KEEPLOG keep-set logger writes valid rows (entropy/jaccard/redundancy in range).

Run:  <venv>/bin/python kvcompress/engine/test_compaction_order.py
"""
import os
import sys
import tempfile
import torch

# CACHE_UTILS_DIR overrides which engine is under test (default: the vendored copy next to this file;
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kvcompress.engine.cache_utils import EvictLayer

B, HKV, HQ, D = 1, 2, 4, 4          # GQA group of 2 (attn-scored modes assert num_kv_groups > 1)
TOKEN_BUDGET, RESIDUAL = 12, 4
K = TOKEN_BUDGET - RESIDUAL          # 8 scored keep slots
PL = 3                               # prompt length


def make_layer(mode):
    layer = EvictLayer(0, TOKEN_BUDGET, RESIDUAL, mode,
                       rkv_lambda=0.5, smooth=False, verbose=False)
    k0 = torch.zeros(B, 16, HKV, D)
    layer.lazy_initialization(k0)
    layer.prompt_len = PL
    layer.queries = torch.randn(B, HQ, RESIDUAL, D)    # attn/caote/mix modes read the query buffer
    return layer


def fill(layer, abs_positions, start_slot):
    """Write tokens whose K/V VALUES are their absolute position, at cache slots start_slot..."""
    for j, p in enumerate(abs_positions):
        layer.k_cache[:, start_slot + j] = float(p)
        layer.v_cache[:, start_slot + j] = float(p)
        if getattr(layer, '_track_positions', False):
            layer.positions[:, :, start_slot + j] = int(p)
    layer.cache_seqlens.fill_(start_slot + len(abs_positions))
    layer.cumulative_length = int(abs_positions[-1]) + 1


def kept_positions(layer, n):
    """Absolute positions currently stored in cache slots [0:n) (per head 0; batch 0)."""
    return sorted(int(x) for x in layer.k_cache[0, :n, 0, 0].tolist())


def raw_slot_values(layer, n):
    """UNSORTED slot values — order as stored (chronology check)."""
    return [int(x) for x in layer.k_cache[0, :n, 0, 0].tolist()]


def run_rounds(mode, raw=False):
    layer = make_layer(mode)
    fill(layer, range(0, 16), 0)                       # abs 0..15 fills 16 slots
    kept_per_round = []
    next_pos = 16
    for _ in range(3):
        layer._run_eviction(16)                        # candidates = [0:12), residual = [12:16)
        kept_per_round.append(raw_slot_values(layer, TOKEN_BUDGET) if raw
                              else kept_positions(layer, TOKEN_BUDGET))
        fill(layer, range(next_pos, next_pos + RESIDUAL), TOKEN_BUDGET)   # decode 4 more
        next_pos += RESIDUAL
    return kept_per_round


PP_MODES = ['random_pp', 'recency_pp', 'caote_pp', 'attn_pp', 'attn_rkv_pp', 'range_sink_sample_attn_pp',
            'mix_step_pp', 'mix_head_pp', 'union_topk_pp', 'random_strat_pp', 'random_cluster_pp']
ALL_MODES = PP_MODES + ['random', 'attn', 'caote', 'random_pp_rec', 'random_block_pp']


def test_recency_pp_true_window():
    for rnd, kept in enumerate(run_rounds('recency_pp'), 1):
        # cache after round r holds prompt + the most recent (TOKEN_BUDGET - PL) tokens
        newest = 16 + (rnd - 1) * RESIDUAL - 1
        want = sorted(set(range(PL)) | set(range(newest - (TOKEN_BUDGET - PL) + 1, newest + 1)))
        assert kept == want, f"recency_pp round {rnd}: kept {kept} != true window {want}"
    print("PASS recency_pp: exact prompt+recency window at every round")


def test_random_pp_prompt_survives():
    torch.manual_seed(0)
    for trial in range(20):                            # 20 random draws: protection must never fail
        for rnd, kept in enumerate(run_rounds('random_pp'), 1):
            missing = set(range(PL)) - set(kept)
            assert not missing, f"random_pp trial {trial} round {rnd}: prompt tokens {missing} evicted"
    print("PASS random_pp: prompt tokens survive every compaction (20 random trials x 3 rounds)")


def test_all_pp_modes_prompt_survives():
    torch.manual_seed(1)
    for mode in PP_MODES:
        for trial in range(5):
            for rnd, kept in enumerate(run_rounds(mode), 1):
                missing = set(range(PL)) - set(kept)
                assert not missing, f"{mode} trial {trial} round {rnd}: prompt tokens {missing} evicted"
    print(f"PASS all *_pp modes ({len(PP_MODES)}): prompt survives every compaction (5 trials x 3 rounds)")


def test_universal_chronological_compaction():
    torch.manual_seed(2)
    for mode in ALL_MODES:
        for rnd, raw in enumerate(run_rounds(mode, raw=True), 1):
            assert raw == sorted(raw), f"{mode} round {rnd}: cache not chronological: {raw}"
            assert len(set(raw)) == TOKEN_BUDGET, f"{mode} round {rnd}: duplicate slots: {raw}"
    print(f"PASS universal chronological compaction: {len(ALL_MODES)} modes, strictly increasing, no dups")


def test_strat_one_per_chunk_round1():
    # Round 1: candidate index == absolute position. n_hist = 12 - PL = 9, m = K - PL = 5 chunks;
    # segmented argmax keeps EXACTLY one token per chunk (chunk = (idx-PL)*m // n_hist).
    torch.manual_seed(3)
    for trial in range(10):
        kept = run_rounds('random_strat_pp')[0]
        hist = [p for p in kept if PL <= p < 12]       # non-prompt candidates kept in round 1
        assert len(hist) == K - PL, f"strat trial {trial}: kept {len(hist)} history tokens != {K - PL}"
        chunks = {(p - PL) * (K - PL) // (12 - PL) for p in hist}
        assert len(chunks) == K - PL, f"strat trial {trial}: chunks {sorted(chunks)} not one-per-chunk ({hist})"
    print("PASS random_strat_pp: exactly one kept token per chunk (round 1, 10 trials)")


def test_cluster_contiguous_round1():
    torch.manual_seed(4)
    for trial in range(10):
        kept = run_rounds('random_cluster_pp')[0]
        hist = sorted(p for p in kept if PL <= p < 12)
        assert len(hist) == K - PL and hist[-1] - hist[0] == K - PL - 1, \
            f"cluster trial {trial}: history keep {hist} is not one contiguous window"
    print("PASS random_cluster_pp: one contiguous window (round 1, 10 trials)")


def test_keeplog_writes_valid_rows():
    torch.manual_seed(5)
    with tempfile.TemporaryDirectory() as td:
        os.environ['KEEPLOG'] = td
        try:
            for mode in ('random_pp', 'union_topk_pp', 'random_cluster_pp'):
                run_rounds(mode)
            files = [f for f in os.listdir(td) if f.startswith('keeplog_')]
            assert files, "KEEPLOG produced no CSV"
            rows = open(os.path.join(td, files[0])).read().strip().split('\n')
            hdr, body = rows[0].split(','), rows[1:]
            assert len(body) == 9, f"expected 9 rows (3 modes x 3 rounds), got {len(body)}"
            for line in body:
                rec = dict(zip(hdr, line.split(',')))
                for m in ('cov_entropy', 'jaccard', 'v_redundancy', 'union_frac_slots'):
                    v = float(rec[m])
                    assert 0.0 <= v <= 1.0, f"{rec['mode']}: {m}={v} out of [0,1]"
                assert int(rec['K']) == K and int(rec['prompt_len']) == PL
        finally:
            del os.environ['KEEPLOG']
    print("PASS KEEPLOG: valid rows, metrics in range")


def test_keeplog_entropy_ordering_bigscale():
    # At realistic scale (504 history tokens, 64 bins) the coverage-entropy ordering must hold:
    # stratified (max coverage) >= plain random >= clustered window (min coverage), with real margins.
    torch.manual_seed(6)
    TB2, RES2, PL2, N2 = 192, 64, 8, 512
    K2 = TB2 - RES2
    with tempfile.TemporaryDirectory() as td:
        os.environ['KEEPLOG'] = td
        try:
            ents = {}
            for mode in ('random_strat_pp', 'random_pp', 'random_cluster_pp'):
                layer = EvictLayer(0, TB2, RES2, mode, rkv_lambda=0.5, smooth=False, verbose=False)
                layer.lazy_initialization(torch.zeros(B, N2, HKV, D))
                layer.prompt_len = PL2
                fill(layer, range(N2), 0)
                layer._run_eviction(N2)
                fp = [f for f in os.listdir(td) if f.startswith('keeplog_')][0]
                rows = open(os.path.join(td, fp)).read().strip().split('\n')
                hdr = rows[0].split(',')
                rec = dict(zip(hdr, rows[-1].split(',')))
                assert rec['mode'] == mode
                ents[mode] = float(rec['cov_entropy'])
        finally:
            del os.environ['KEEPLOG']
    assert ents['random_strat_pp'] >= ents['random_pp'] - 0.02, f"strat < random: {ents}"
    assert ents['random_pp'] >= ents['random_cluster_pp'] + 0.05, f"random <= cluster: {ents}"
    print(f"PASS KEEPLOG entropy ordering at scale: strat {ents['random_strat_pp']:.3f} >= "
          f"random {ents['random_pp']:.3f} >= cluster {ents['random_cluster_pp']:.3f} (+margin)")


if __name__ == '__main__':
    test_recency_pp_true_window()
    test_random_pp_prompt_survives()
    test_all_pp_modes_prompt_survives()
    test_universal_chronological_compaction()
    test_strat_one_per_chunk_round1()
    test_cluster_contiguous_round1()
    test_keeplog_writes_valid_rows()
    test_keeplog_entropy_ordering_bigscale()
