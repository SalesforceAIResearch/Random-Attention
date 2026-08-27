# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
"""KV-eviction engine: EvictCache / EvictLayer with every eviction policy compared in the paper.

Provenance: derived from VaSE's eval/reasoning_tasks/modified/transformers/cache_utils.py (https://github.com/terarachang/VaSE,
MIT License, Copyright (c) 2026 Ting-Yun Chang) at commit bfa2692, which is itself a modified copy of
Hugging Face transformers' cache_utils.py (Apache-2.0). VaSE's modes (attn/SnapKV, attn_rkv/R-KV,
range_sink_sample_attn/VaSE, caote, recency, ...) are theirs; Random Attention adds (~850 lines):
  * random_pp (Random Attention: prompt-protected, per-KV-head uniform random retention) and the
    matched-protection variants *_pp (attn_pp, attn_rkv_pp, range_sink_sample_attn_pp, recency_pp);
  * the faithful in-engine TriAttention scorer (triattn_ph / triattn_pl; math in triattn_official.py,
    ported from github.com/WeianMao/triattention, Apache-2.0) with the per-generation memo reset;
  * rkv_official (R-KV with the authors' lambda), the budget-parity guard, chronological compaction,
    the KEEPLOG / KEEPLOG_POS retention loggers, EVICT_TIMING hooks, and the mechanism-study modes
    (carrier_mix / softmix, random_pcap, synth_pin, FORCE_KEEP_RANGE replay).
See docs/PROVENANCE.md for the exact diff against upstream.
"""
from typing import Any, Optional
import math
import os
import sys
import torch
import torch.nn.functional as F

from transformers.configuration_utils import PreTrainedConfig
from transformers.cache_utils import Cache, DynamicLayer

# Modes with no 'attn' in the name that still consume the query buffer / attention scores
# (the T1.0 mixes need all three heuristic signals: value-range, attention, CAOTE output-error).
_QUERY_MODES = ('caote', 'caote_pp', 'cusa', 'mix_step_pp', 'mix_head_pp', 'union_topk_pp', 'rkv_official')


class EvictCache(Cache):
    def __init__(
        self,
        config: PreTrainedConfig,
        token_budget: int,
        residual_length: int,
        eviction_mode: str,
        **kwargs,
    ):

        config = config.get_text_config(decoder=True)
        layers = [
            EvictLayer(l, token_budget, residual_length, eviction_mode, **kwargs)
            for l in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        for _lyr in layers:            # back-ref so triattn_ph can aggregate scores across ALL layers (layer-shared keep-set)
            _lyr._sibling_layers = layers
        # GENERATION-BOUNDARY MEMO RESET (08-04 audit, must-fix #1). The triattn_ph cross-layer
        # keep-set memo is a single module-global slot. Without this reset, if problem A ends exactly
        # one eviction round after its first compaction and problem B ties on the whole key
        # (cache_seqlens, batch, n_cand, round_start, clamped prompt_len), B's ENTIRE first round
        # re-applies A's keep-set indices across all layers -- silently and shard-order-dependently.
        # A new EvictCache is constructed per generation, so this is exactly the official reference's
        # per-generation state reset (methods/triattention.py:784-796). One line closes the class.
        import sys as _sys
        _sys.modules[__name__]._triattn_ph_shared = None


def _tri_env_on(name):
    """VALUE-parse a triattn debug flag (08-04 audit, must-fix #2). The old presence test meant
    TRIATTN_NO_NORM=0 silently turned the faithful normalisation OFF -- one mis-templated export
    would reproduce the pre-08-04 defect across a whole re-run with no log evidence."""
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


def _load_triattn_pl(device):
    """Load + cache the faithful TriAttention (`triattn_pl`) stats (official format + baked omega/freq_scale)."""
    import sys as _sys
    _mod = _sys.modules[__name__]
    if not hasattr(_mod, '_triattn_pl_cache'):
        _mod._triattn_pl_cache = {}
    _path = os.environ['TRIATTN_STATS']
    if _path in _mod._triattn_pl_cache:
        return _mod._triattn_pl_cache[_path]
    # PROVENANCE BANNER, once per process: the audit's point is that scorer provenance must be in the
    # log, not inferred from the launcher. Grep for 'triattn provenance' when auditing a run.
    print(f"[triattn provenance] stats={_path} norm={not _tri_env_on('TRIATTN_NO_NORM')} "
          f"frozen_memo={_tri_env_on('TRIATTN_FROZEN_MEMO')}", flush=True)
    from kvcompress.engine.triattn_official import load_head_frequency_stats, build_geometric_offsets
    meta, stats = load_head_frequency_stats(_path, device)
    Hq, Hkv, Ln = int(meta['Hq']), int(meta['Hkv']), int(meta['n_layers'])
    F = int(meta['head_dim']) // 2
    qmc = torch.zeros(Ln, Hq, F, dtype=torch.cfloat, device=device)
    qam = torch.zeros(Ln, Hq, F, device=device)
    for (l, h), hs in stats.items():
        qmc[l, h] = hs.q_mean_complex.to(device); qam[l, h] = hs.q_abs_mean.to(device)
    out = dict(qmc=qmc, qam=qam,
               omega=meta['omega'].to(device).float(),
               fss=(meta['freq_scale'].to(device).float() ** 2),
               offsets=build_geometric_offsets(65536, device),
               style=meta.get('rope_style', 'half'), Hq=Hq, Hkv=Hkv, grp=Hq // Hkv)
    _mod._triattn_pl_cache[_path] = out
    return out


class EvictLayer(DynamicLayer):
    """
    KV cache layer backed by pre-allocated [B, max_len, Hkv, D] storage for `flash_attn_with_kvcache`
    """

    _parity_logged = False  # one-time budget-parity log/assert across all layers in a process

    def __init__(
        self,
        layer_idx: int,
        token_budget: int,
        residual_length: int,
        eviction_mode: str,
        rkv_lambda: float,
        smooth: bool,
        verbose: bool,
        n_large: int = 200,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.token_budget = token_budget
        self.residual_length = residual_length
        self.rkv_lambda = rkv_lambda
        self.smooth = smooth
        self.verbose = verbose
        self.MODE = eviction_mode
        self.n_large = n_large
        self.temperature = temperature
        self.n_sink = 4
        self.cumulative_length = 0
        self._q_write_pos = 0  # circular buffer head for self.queries (attn modes only)
        # Absolute-position tracking: required by triattn (delta/phi) and by the T1.0 keep-set logger
        # (KEEPLOG env), which needs the true trace position of every kept slot for coverage entropy.
        self._track_positions = eviction_mode in ('triattn_pl', 'triattn_ph', 'random_pcap', 'synth_pin') or bool(os.environ.get('KEEPLOG')) or bool(os.environ.get('FORCE_KEEP_RANGE'))

        if verbose and layer_idx == 0:
            print(
                f'Eviction Strategy: {self.MODE}, Budget: {self.token_budget}, Buffer: {self.residual_length}, '
                f'RKV_lambda: {self.rkv_lambda}, Smooth: {self.smooth}'
            )

    def lazy_initialization(self, key_states: torch.Tensor, value_states: Optional[torch.Tensor] = None) -> None:
        batch_size, prompt_len, num_kv_heads, head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        max_cache_len = max(prompt_len, self.token_budget) + self.residual_length
        self.k_cache = torch.zeros(
            batch_size, max_cache_len, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.v_cache = torch.zeros(
            batch_size, max_cache_len, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        if self._track_positions:   # per-KV-head absolute token position of each cache slot (triattn delta/phi; KEEPLOG coverage)
            self.positions = torch.zeros(batch_size, num_kv_heads, max_cache_len, dtype=torch.long, device=self.device)
        self.is_initialized = True
        if self.verbose and self.layer_idx == 0:
            print(f'self.k_cache shape: {self.k_cache.shape}')

    def prepare(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache, cache_seqlens) for flash_attn_with_kvcache."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        return self.k_cache, self.v_cache, self.cache_seqlens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "EvictLayer is driven by prepare() + post_attention_update() in the "
            "patched attention forward; the standard Cache.update() path is bypassed."
        )

    def update_queries(self, query_states: torch.Tensor) -> None:
        """
        Cache query_states into the circular [B, Hq, residual_length, D] buffer.
        Called *after* RoPE and *before* the flash-attn transpose, so query_states is in the post-RoPE
        """
        if 'attn' not in self.MODE and self.MODE not in _QUERY_MODES:
            return

        q_len = query_states.shape[2]
        if q_len == 1: # decode: round-robin single-slot copy into the existing buffer.
            self.queries[:, :, self._q_write_pos:self._q_write_pos + 1, :].copy_(query_states)
            self._q_write_pos = (self._q_write_pos + 1) % self.residual_length
        else:          # prefill (first call): allocate the buffer and fill from last `n` queries.
            B, Hq, _, D = query_states.shape
            self.queries = torch.zeros(B, Hq, self.residual_length, D,
                                       dtype=query_states.dtype, device=query_states.device)
            n = min(q_len, self.residual_length)
            self.queries[:, :, :n, :].copy_(query_states[:, :, -n:, :])
            self._q_write_pos = n % self.residual_length

    def post_attention_update(self, n_new: int) -> None:
        """
        Called after `flash_attn_with_kvcache` has written n_new K/V into the cache.
        Advances cache_seqlens and periodically runs eviction.
        """
        self.cache_seqlens += n_new
        self.cumulative_length += n_new
        if n_new > 1:                       # prefill: remember the prompt length (for *_pp prompt protection)
            self.prompt_len = int(n_new)
        if self._track_positions:                           # stamp the absolute position of the just-appended slots
            cur = int(self.cache_seqlens[0].item()); C = self.cumulative_length
            self.positions[:, :, cur - n_new:cur] = torch.arange(
                C - n_new, C, device=self.device, dtype=torch.long).view(1, 1, -1)

        is_decode = (n_new == 1)
        cur_len = int(self.cache_seqlens[0].item())  # same length across the batch — GUARANTEED by the
        # harness (eval_hf.py batches ONE prompt replicated bs times; batch dim = sampling runs). If you ever
        # batch DIFFERENT prompts here, this shared trigger + the scalar prompt_len become wrong (pads would
        # enter the cache and the budget); audited 2026-07-14 — see REPORT "Batching audit".
        if is_decode and cur_len > self.token_budget and cur_len % self.residual_length == 0:
            self._run_eviction(cur_len)

    def _run_eviction(self, cur_len: int) -> None:
        def select_remaining_based_on_scores(ids_to_keep, scores, n_remain):
            scores.scatter_(dim=-1, index=ids_to_keep, value=float('-inf'))
            ids_remain = scores.topk(n_remain, dim=-1, largest=True).indices
            return torch.cat([ids_to_keep, ids_remain], dim=-1)

        def compute_attention_scores(query_cache, keys):
            # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L8
            batch_size, q_heads, q_len, head_dim = query_cache.shape
            kv_heads = keys.shape[1]
            num_kv_groups = q_heads // kv_heads
            assert num_kv_groups > 1, "not GQA"

            query_cache = query_cache.view(batch_size, kv_heads, num_kv_groups, q_len, head_dim)
            keys = keys.unsqueeze(2)

            # shape: [batch_size, kv_heads, num_kv_groups, q_len, kv_len]
            attn_weights = torch.matmul(
                query_cache, keys.transpose(3, 4)
            ) / (math.sqrt(head_dim) * self.temperature)

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_cache.dtype)
            scores = attn_weights.mean(-2)  # avg over q_len (window_size)
            if self.smooth:
                # https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/snapkv_press.py#L96
                orig_shape = scores.shape
                scores = scores.view(batch_size, q_heads, -1)
                kernel_size = 5
                scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
                scores = scores.view(orig_shape)
            scores = scores.mean(2)  # avg heads in the same q_group
            return scores

        def cal_redundancy(keys):
            # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L42
            k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
            cos_similarity = torch.matmul(k_norm, k_norm.transpose(-1, -2))
            # zero diagonal (self-similarity)
            diag = torch.eye(keys.size(2), device=self.device, dtype=torch.bool)
            cos_similarity.masked_fill_(diag.unsqueeze(0).unsqueeze(0), 0.0)
            return cos_similarity.mean(dim=2).softmax(dim=-1)

        def multinomial_sample(scores, K):
            orig_shape = scores.shape
            flat_scores = scores.view(-1, orig_shape[-1])
            ids_to_keep = torch.multinomial(flat_scores, K, replacement=False)
            return ids_to_keep.view(*orig_shape[:-1], K)

        residual = self.residual_length
        K = self.token_budget - residual

        k_candidates = self.k_cache[:, :cur_len - residual].transpose(1, 2)  # [B, Hkv, cur_len - residual, D]
        v_candidates = self.v_cache[:, :cur_len - residual].transpose(1, 2)
        batch_size, n_heads, _, head_dim = v_candidates.shape

        # Several methods use v_magnitude as the importance scores for eviction
        v_magnitude = None
        if 'range' in self.MODE:
            v_magnitude = (v_candidates.amax(-1) - v_candidates.amin(-1))
        elif 'var' in self.MODE:
            v_magnitude = v_candidates.var(dim=-1)
        elif 'l2' in self.MODE:
            v_magnitude = v_candidates.norm(dim=-1)
        if v_magnitude is not None and 'sink' in self.MODE:
            v_magnitude[..., :self.n_sink] = float('inf')
        
        # Several methods use attention as the importance scores for eviction
        if 'attn' in self.MODE or self.MODE in _QUERY_MODES:
            attn_scores = compute_attention_scores(self.queries, k_candidates)
            if 'rkv' in self.MODE and self.MODE != 'rkv_official':
                similarities = cal_redundancy(k_candidates)
                attn_scores = attn_scores * self.rkv_lambda - similarities * (1 - self.rkv_lambda)

        if self.MODE == 'rkv_official':
            # FAITHFUL R-KV port (audited 07-21 vs github.com/Zefan-Cai/R-KV @ HuggingFace/rkv):
            # (1) importance: attention of the LAST 8 queries over candidates, MAX-pooled over the GQA query
            #     group (their compute_attention_scores default pooling="max"), softmax over keys, mean over
            #     the 8 queries, then max_pool1d(kernel=7) (their R1KV.update_kv);
            # (2) redundancy: their cal_similarity VERBATIM — cosine matrix, diag zeroed, THRESHOLD 0.5 mask,
            #     per-row LAST similar index spared (retain_direction='last': the newest copy of each redundant
            #     cluster keeps no penalty), mean over rows, softmax;
            # (3) score = lambda*importance - (1-lambda)*redundancy, lambda via --rkv_lambda (README: 0.1).
            # Divergences from the VaSE-port attn_rkv it replaces: group MAX (not mean), last-8 queries (not
            # 64-buffer mean), max-pool k=7 (not optional avg k=5), thresholded spare-the-newest redundancy
            # (not plain mean). Engine-common cadence/residual/compaction as for every baseline (disclosed).
            _W8 = 8
            _qb = self.queries  # [B, Hq, residual, D] circular buffer; take the newest 8 in written order
            _rl = self.residual_length
            _idx = torch.arange(self._q_write_pos - _W8, self._q_write_pos, device=self.device) % _rl
            _q8 = _qb.index_select(2, _idx)                                    # [B, Hq, 8, D]
            _B, _Hq, _, _D = _q8.shape
            _Hkv = k_candidates.shape[1]; _g = _Hq // _Hkv
            _qg = _q8.view(_B, _Hkv, _g, _W8, _D)
            _aw = torch.matmul(_qg, k_candidates.unsqueeze(2).transpose(3, 4)) / math.sqrt(_D)
            _aw = _aw.max(dim=2).values                                        # MAX over query group -> [B,Hkv,8,n]
            _imp = F.softmax(_aw, dim=-1, dtype=torch.float32).mean(dim=-2).to(k_candidates.dtype)
            _imp = F.max_pool1d(_imp, kernel_size=7, padding=3, stride=1)      # their kernel_size=7
            _kn = k_candidates / (k_candidates.norm(dim=-1, keepdim=True) + 1e-8)
            _cs = torch.matmul(_kn, _kn.transpose(-1, -2))
            _n = k_candidates.shape[2]
            _cs.masked_fill_(torch.eye(_n, dtype=torch.bool, device=self.device).view(1, 1, _n, _n), 0.0)
            _mask = _cs > 0.5                                                  # their threshold=0.5
            _ind = torch.where(_mask, torch.arange(_n, device=self.device).view(1, 1, 1, _n),
                               torch.zeros_like(_mask, dtype=torch.long))
            _spare = torch.max(_ind, dim=-1)[0]                                # retain_direction='last'
            _cs.scatter_(-1, _spare.unsqueeze(-1), 0)                          # newest copy keeps no penalty
            _red = _cs.mean(dim=-2).softmax(dim=-1)
            attn_scores = _imp * self.rkv_lambda - _red * (1 - self.rkv_lambda)

        if self.MODE in ['attn', 'attn_rkv', 'rkv_official']:
            ids_to_keep = attn_scores.topk(K, dim=-1, largest=True).indices

        elif self.MODE in ['range_attn', 'range_sample_attn', 'range_sink_sample_attn',
            'var_sink_sample_attn', 'l2_sink_sample_attn']:
            ids_large = v_magnitude.topk(self.n_large, dim=-1, largest=True).indices
            if 'sample' in self.MODE:
                attn_scores.scatter_(dim=-1, index=ids_large, value=0.0)
                ids_attn = multinomial_sample(attn_scores, K - self.n_large)
                ids_to_keep = torch.cat([ids_large, ids_attn], dim=-1)
            else:
                ids_to_keep = select_remaining_based_on_scores(ids_large, attn_scores, K - self.n_large)

        elif 'cur' in self.MODE:
            # https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/cur_press.py
            if self.MODE == 'cur_fixed_gauss':
                if not hasattr(self, 'G'):
                    r = 20
                    # per-row G to avoid biased sampling across the batch (1-2% acc gain empirically)
                    self.G = (torch.randn(batch_size, 1, head_dim, r, device=self.device) / math.sqrt(r)).to(self.dtype)
                G = self.G
            else:  # cur_resample_gauss
                r = 20
                G = (torch.randn(batch_size, 1, head_dim, r, device=self.device) / math.sqrt(r)).to(self.dtype)
            keys = k_candidates @ G
            values = v_candidates @ G
            k2 = (keys ** 2).sum(dim=-1)
            v2 = (values ** 2).sum(dim=-1)
            scores = k2 * v2
            scores /= scores.sum(dim=-1, keepdim=True)
            scores[:, :, :self.n_sink] = 1.0
            ids_to_keep = scores.topk(K, dim=-1, largest=True).indices

        # == Ablation ==
        elif self.MODE == 'evict_large_range':
            # Note: sink tokens have small range (value-state drain)
            ids_to_keep = v_magnitude.topk(K, dim=-1, largest=False).indices

        elif self.MODE == 'attn_sample':
            ids_to_keep = multinomial_sample(attn_scores, K)

        elif self.MODE == 'attn_random':
            random_scores = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            ids_large = random_scores.topk(self.n_large, dim=-1, largest=True).indices
            ids_to_keep = select_remaining_based_on_scores(ids_large, attn_scores, K - self.n_large)

        # == Baselines added for the selection-vs-random study (per-KV-head, faithful; no prompt protection) ==
        elif self.MODE == 'random':            # pure random per head
            random_scores = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            ids_to_keep = random_scores.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'recency':           # keep most-recent K candidates (StreamingLLM-like; prompt evicted)
            n_cand = v_candidates.size(2)
            rec = torch.arange(n_cand - K, n_cand, device=self.device, dtype=torch.long)
            ids_to_keep = rec.view(1, 1, K).expand(batch_size, n_heads, K).contiguous()

        elif self.MODE == 'random_pp':         # FAIR baseline: protect prompt [0:pl) + random rest
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            rs = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            rs[..., :pl] = float('inf')        # force-keep the prompt (top-K always includes it)
            fkr = os.environ.get('FORCE_KEEP_RANGE')      # fork-autopsy replay: pin span [lo,hi) by
            if fkr:                                       # ABSOLUTE position in EVERY head (matched budget:
                lo, hi = (int(x) for x in fkr.split(':')) # the pinned L tokens displace L random keeps)
                pos_t = self.positions[:, :, :v_candidates.size(2)]
                rs = torch.where((pos_t >= lo) & (pos_t < hi), torch.full_like(rs, float('inf')), rs)
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'random_pmix':       # PER-HEAD PROBABILISTIC prompt protection (user idea 07-18):
            # each KV head is drawn ONCE (first eviction, persisted) as ARCHIVIST (prob PMIX_P, keeps the FULL
            # prompt) or SCATTERER (keeps only the first PMIX_SMALL prompt tokens + max random scatter). The
            # cross-head/layer UNION retains ~the whole prompt (P(no archivist in a layer)=(1-p)^Hkv) while the
            # AVERAGE per-head prompt cost drops ~(p*pl+(1-p)*small)/pl — freeing scatter budget on long-prompt
            # problems. Role stability across evictions is essential (a scatterer's dropped prompt tokens are
            # gone for good; per-head front-compaction keeps each head's retained prompt block contiguous).
            # Envs: PMIX_P (dflt 0.25), PMIX_SMALL (dflt 256). Zero learned signal.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            small = min(int(os.environ.get('PMIX_SMALL', '256')), pl)
            if not hasattr(self, '_pmix_arch'):
                p = float(os.environ.get('PMIX_P', '0.25'))
                self._pmix_arch = (torch.rand(n_heads, device=self.device) < p)
                if not bool(self._pmix_arch.any()):          # guarantee >=1 archivist per layer
                    self._pmix_arch[torch.randint(n_heads, (1,), device=self.device)] = True
                self._pmix_thresh = torch.where(self._pmix_arch,
                                                torch.full((n_heads,), pl, device=self.device),
                                                torch.full((n_heads,), small, device=self.device))
            rs = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            pos = torch.arange(v_candidates.size(2), device=self.device).view(1, 1, -1)
            rs = torch.where(pos < self._pmix_thresh.view(1, n_heads, 1), torch.full_like(rs, float('inf')), rs)
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'random_pcap':       # random_pp with CAPPED prompt protection (32B-LCB long-prompt
            # squeeze, 07-18: rp term drops 0.746->0.647 short->long prompts while vase stays flat — rp's hard
            # prompt reservation strangles the scatter on long-prompt problems). Designated set = first PCAP
            # prompt tokens + every-2nd token of the overflow [PCAP, pl); everything else scatters uniformly.
            # Membership is computed from TRACKED ABSOLUTE POSITIONS every round (requires _track_positions),
            # so the designated set is stable by construction — the earlier slot-range version decayed because
            # a non-contiguous designated set interleaves with undesignated survivors under chronological
            # compaction (bug found by the 07-18 CPU unit tests). Stateless. PROMPT_CAP env (dflt 512).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            pcap = min(int(os.environ.get('PROMPT_CAP', '512')), pl)
            pos_t = self.positions[:, :, :v_candidates.size(2)]
            desig = (pos_t < pcap) | ((pos_t >= pcap) & (pos_t < pl) & (((pos_t - pcap) % 2) == 0))
            rs = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            rs = torch.where(desig, torch.full_like(rs, float('inf')), rs)
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'synth_pin':         # SYNTHETIC MECHANISM PROBE (S1/S1b/S4 controls): uniform random
            # background with NO protection rule anywhere; needle spans are PINNED into an exact (head, layer)
            # set and BLACKLISTED everywhere else, so the number of surviving copies h is exact by construction.
            # Spec: SYNTH_SPEC = JSON list of {"lo": int, "hi": int, "heads": [kv-head idx...], "layers":
            # "all" | [layer idx...]} in ABSOLUTE token positions (half-open [lo,hi)). Membership is computed
            # from tracked absolute positions every round (same stability argument as random_pcap: a
            # non-contiguous designated set decays under chronological compaction if done by slot). Default for
            # any span position: BLACKLISTED (-inf, evicted at first eligibility); pinned (+inf) only where
            # layer-in-scope AND head-in-set. h=0 = pure blacklist. Runtime audit: SYNTH_AUDIT=1 dumps per-event
            # per-head span-retention counts so every cell can assert its h from the log (silent-invalidation guard).
            spec_env = os.environ.get('SYNTH_SPEC', '[]')
            if getattr(self, '_synth_env', None) != spec_env:
                import json as _json
                self._synth_env = spec_env
                self._synth_spec = _json.loads(spec_env)
            pos_t = self.positions[:, :, :v_candidates.size(2)]              # [B, Hkv, n_cand] absolute pos
            rs = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            for _si, _sp in enumerate(self._synth_spec):
                span = (pos_t >= int(_sp['lo'])) & (pos_t < int(_sp['hi']))  # [B, Hkv, n_cand]
                rs = torch.where(span, torch.full_like(rs, float('-inf')), rs)   # default: blacklist
                if 'rho' in _sp:
                    # Item 3 (correlation dose): survival of the span at each site head is drawn
                    # ONCE (first eviction event, persisted) with marginal s and pairwise
                    # cross-head correlation rho via a common-factor mixture: with prob rho all
                    # site heads share one Bernoulli(s) draw, else independent Bernoulli(s).
                    # Matched expected mass across rho by construction. Draw is batch-shared
                    # (engine evicts batch-jointly) — use small bs for draw diversity.
                    key = f'_synth_rho_{_si}'
                    if not hasattr(self, key):
                        site = [h for h in _sp.get('site_heads', [2, 3, 5]) if h < n_heads]
                        svv = float(_sp.get('s', 0.4)); rho = float(_sp['rho'])
                        # one coin per LAYER decides shared-vs-independent -> pairwise
                        # cross-head correlation is exactly rho (per-site mixing gives rho^2)
                        common = torch.rand(1).item() < svv
                        shared_layer = torch.rand(1).item() < rho
                        indep = (torch.rand(len(site)) < svv).tolist()
                        keep = [common] * len(site) if shared_layer else indep
                        hs_draw = [h for h, k in zip(site, keep) if k]
                        setattr(self, key, hs_draw)
                    hs = getattr(self, key)
                else:
                    _ly = _sp.get('layers', 'all')
                    hs = ([h for h in _sp.get('heads', []) if h < n_heads]
                          if (_ly == 'all' or self.layer_idx in _ly) else [])
                if hs:
                    hm = torch.zeros(n_heads, dtype=torch.bool, device=self.device)
                    hm[hs] = True
                    rs = torch.where(span & hm.view(1, n_heads, 1), torch.full_like(rs, float('inf')), rs)
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices
            if os.environ.get('SYNTH_AUDIT'):
                kept_pos = torch.gather(pos_t, -1, ids_to_keep)              # [B, Hkv, K]
                for _si, _sp in enumerate(self._synth_spec):
                    inset = ((kept_pos >= int(_sp['lo'])) & (kept_pos < int(_sp['hi']))).sum(-1)  # [B, Hkv]
                    _dir = os.environ.get('SYNTH_AUDIT_DIR', '/tmp')
                    with open(os.path.join(_dir, f'synth_audit_p{os.getpid()}.tsv'), 'a') as _f:
                        _f.write(f"{self.layer_idx}\t{_si}\t{self.cumulative_length}\t"
                                 f"{inset[0].tolist()}\n")

        elif self.MODE == 'random_ppp':        # random_pp + PLAN-PREFIX: protect prompt [0:pl) AND the first
            # P generated tokens (the model's own plan/derivation opening). Code-failure forensics 07-18:
            # rp's code losses are cap-rambles that endlessly re-derive a lost plan (P(pass|term) ties vase on
            # matched problems; capped runs never pass; both LOST-SPEC cases contradict the model's OWN mid-CoT
            # derivation, never the protected prompt). Chronological compaction keeps the earliest-generated
            # survivors at slots [pl:pl+P), so slot protection == exact plan-prefix protection. Zero signal,
            # structural only. P via PLAN_PREFIX env (default 768); always leaves >=64 slots for scatter.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            ppfx = min(pl + int(os.environ.get('PLAN_PREFIX', '768')), K - 64)
            rs = torch.rand(batch_size, n_heads, v_candidates.size(2), device=self.device)
            rs[..., :ppfx] = float('inf')
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices

        elif self.MODE in ('carrier_mix', 'anti_mix'):   # F2 CAUSAL TRANSFER TEST (07-23): matched-budget
            # age-reallocation by carrier map. carrier_mix: carrier heads (CARRIER_HEADS env, default
            # 2,3,5 = the synthetically-identified live storage carriers of Qwen3-4B) keep rp's uniform
            # random scatter over history; all other heads spend their K on the YOUNGEST history
            # (pure recency). anti_mix: identical policy with the head sets SWAPPED — the control that
            # makes the comparison causal (same budget, same structure, only the head-assignment
            # permuted). Both protect the prompt in every head (matched to random_pp).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            ch = [int(x) for x in os.environ.get('CARRIER_HEADS', '2,3,5').split(',')]
            hm = torch.zeros(n_heads, dtype=torch.bool, device=self.device)
            hm[[h for h in ch if h < n_heads]] = True
            if self.MODE == 'anti_mix':
                hm = ~hm
            rs = torch.rand(batch_size, n_heads, n_cand, device=self.device)
            pos = torch.arange(n_cand, device=self.device, dtype=torch.float).view(1, 1, n_cand)
            rec = (pos / max(n_cand, 1)).expand(batch_size, n_heads, n_cand)   # monotone recency score
            rs = torch.where(hm.view(1, n_heads, 1), rs, rec)
            rs[..., :pl] = float('inf')
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices

        elif self.MODE in ('carrier_softmix', 'anti_softmix'):   # IDLE-BURST SOFT TILT (08-07, Heng):
            # the INTERIOR point of the carrier_mix family. carrier_mix's null was the q=1 EXTREME
            # (non-carriers 100% recency); rp is q=0. Here non-carrier heads keep a guaranteed
            # recency slice of m = floor(SOFTMIX_Q * (K - pl)) slots and spend the rest on rp's
            # random scatter; carrier heads (CARRIER_HEADS, default 2,3,5) keep full scatter.
            # anti_softmix = identical policy, head sets swapped (the causal control). Uniform K per
            # head throughout -- no rectangularity issues; prompt protected in every head.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            q = float(os.environ.get('SOFTMIX_Q', '0.5'))
            ch = [int(x) for x in os.environ.get('CARRIER_HEADS', '2,3,5').split(',')]
            hm = torch.zeros(n_heads, dtype=torch.bool, device=self.device)
            hm[[h for h in ch if h < n_heads]] = True
            if self.MODE == 'anti_softmix':
                hm = ~hm
            m = max(int(q * (K - pl)), 0)
            rs = torch.rand(batch_size, n_heads, n_cand, device=self.device)
            if m > 0:
                # newest m candidates guaranteed for NON-carrier heads (recency slice), rest random
                rec_slice = torch.zeros_like(rs, dtype=torch.bool)
                rec_slice[..., max(n_cand - m, 0):] = True
                rs = torch.where(rec_slice & (~hm).view(1, n_heads, 1), torch.full_like(rs, float('inf')), rs)
            rs[..., :pl] = float('inf')
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'random_unified':    # CAUSAL CONTROL for random_pp (union hypothesis): SAME random
            # signal + SAME prompt protection, but ONE keep-set SHARED across all KV-heads (minimal union).
            # random_pp draws independently per head (max union); random_unified shares one draw across heads
            # (union = a single head's K). The ONLY difference is cross-head correlation -> isolates union/
            # diversity from the signal. Same K budget. (Mirrors cusa's head-broadcast, but with random scores.)
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            rs = torch.rand(batch_size, n_cand, device=self.device)   # ONE score per (batch, position): no head dim
            rs[..., :pl] = float('inf')                               # force-keep the prompt (matched to random_pp)
            keep = rs.topk(K, dim=-1, largest=True).indices           # [B, K]
            ids_to_keep = keep.unsqueeze(1).expand(batch_size, n_heads, K).contiguous()  # SAME positions all heads

        elif self.MODE == 'segment_pp':        # STRUCTURED-COVERAGE variant of random_pp: protect prompt [0:pl) +
            # each head keeps a CONTIGUOUS block of K-pl history tokens; block centers are tiled evenly across the
            # history so the n_kv_heads together cover the whole trace (maximizes union + keeps reasoning spans
            # intact), vs random_pp's per-head random scatter. Matched K (+64 residual = token_budget).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            n_hist = max(n_cand - pl, 1)
            pos = torch.arange(n_cand, device=self.device, dtype=torch.float).view(1, 1, n_cand)      # [1,1,n_cand]
            h_idx = torch.arange(n_heads, device=self.device, dtype=torch.float).view(1, n_heads, 1)  # [1,Hkv,1]
            centers = pl + (h_idx + 0.5) * (n_hist / n_heads)     # head h -> middle of its 1/Hkv slice of history
            scores = -(pos - centers).abs().expand(batch_size, n_heads, n_cand).contiguous()  # closeness to center
            scores[..., :pl] = float('inf')                       # force-keep prompt (matched to random_pp)
            ids_to_keep = scores.topk(K, dim=-1, largest=True).indices  # pl prompt + K-pl nearest-to-center = a block

        elif self.MODE == 'segment_pp_rot':    # segment_pp + CROSS-LAYER diversity: rotate the head->segment map by
            # layer_idx so each history region is read by a DIFFERENT query-group across depth. (Fixed segment_pp
            # traps each region to one head at EVERY layer; random_pp gets per-layer+per-call diversity for free via
            # fresh torch.rand each eviction.) Stable WITHIN a layer -> coherent under repeated decode-eviction;
            # rotated ACROSS layers -> over Hkv layers every query-group cycles through all segments. Same coverage
            # (a rotation still tiles the history), matched K.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            n_hist = max(n_cand - pl, 1)
            pos = torch.arange(n_cand, device=self.device, dtype=torch.float).view(1, 1, n_cand)
            h_long = torch.arange(n_heads, device=self.device)
            seg = ((h_long + int(self.layer_idx)) % n_heads).to(torch.float).view(1, n_heads, 1)  # head h @ layer L -> seg (h+L)%Hkv
            centers = pl + (seg + 0.5) * (n_hist / n_heads)
            scores = -(pos - centers).abs().expand(batch_size, n_heads, n_cand).contiguous()
            scores[..., :pl] = float('inf')
            ids_to_keep = scores.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'recency_pp':        # FAIR baseline: protect prompt [0:pl) + most-recent rest
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            score = torch.arange(n_cand, device=self.device, dtype=torch.float)  # position = recency
            score[:pl] = float('inf')          # force-keep the prompt
            ids_to_keep = score.view(1, 1, n_cand).expand(batch_size, n_heads, n_cand).topk(K, dim=-1, largest=True).indices
            ids_to_keep, _ = torch.sort(ids_to_keep, dim=-1)   # CHRONOLOGICAL COMPACTION (recency-fix, 2026-07-08):
            # topk(largest) returns the non-prompt indices in DESCENDING order, so without this sort the compacted
            # cache stores the recency window reversed; on the 2nd+ eviction "score = index" then keeps the OLDEST
            # of the previously-kept block and drops the newest (verified by simulation: keeps {7,12..15} where a
            # true window keeps {11..15}). All recency_pp cells run before this fix (incl. the §4 2x2 concentrate
            # arm, 0.512) were NOT a faithful prompt+recency window and must be re-run.

        elif self.MODE == 'random_pp_rec':     # random_pp + a LARGER recency window: protect prompt [0:pl) +
            # force-keep the most-recent n_rec candidates + uniform-random scatter the middle. The recency
            # fraction = rkv_lambda (of K), repurposed here (default 0.5); n_rec is ON TOP of the always-kept
            # residual buffer. frac=0 -> random_pp; frac=1 -> ~recency_pp. All within the same K budget.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            frac = self.rkv_lambda if self.rkv_lambda is not None else 0.5
            n_rec = min(int(round(frac * K)), K - pl)            # recent slots, leaving >=0 for random
            rs = torch.rand(batch_size, n_heads, n_cand, device=self.device)
            rs[..., :pl] = float('inf')                          # force-keep the prompt
            if n_rec > 0:
                rs[..., n_cand - n_rec:] = float('inf')          # force-keep the most-recent n_rec candidates
            ids_to_keep = rs.topk(K, dim=-1, largest=True).indices  # pl prompt + n_rec recent + (K-pl-n_rec) random
            ids_to_keep, _ = torch.sort(ids_to_keep, dim=-1)        # CHRONOLOGICAL COMPACTION (recency-fix): the gather
            # below stores kept tokens in topk(score) order; without this, on the 2nd+ eviction the "most-recent n_rec"
            # positions are scrambled-random past the 64-tok residual, so the recency window never grows. Sorting keeps
            # positions chronological so n_rec is genuinely recent. No-op for attention (post-RoPE keys are permutation-invariant).

        elif self.MODE == 'random_block_pp':   # BLOCK-granularity random_pp: protect prompt [0:pl) + each head keeps
            # random CONTIGUOUS blocks of size B (~Khist/B whole blocks/head). B=1 -> random_pp (token scatter);
            # large B -> few big blocks (segment-like). Keeps LOCAL span coherence (intact reasoning chunks) while
            # restoring per-head coverage diversity (independent block draws -> a token lands in several heads). The
            # middle ground that segment_pp (one big block/head = max specialization, rambles) is the failed extreme of.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            B = max(int(os.environ.get('BLOCK_SIZE', 32)), 1)
            n_blocks = (n_cand + B - 1) // B
            blk = torch.rand(batch_size, n_heads, n_blocks, device=self.device)         # one random score per block per head
            tok_blk = (torch.arange(n_cand, device=self.device) // B)                   # token -> its block id
            scores = blk.gather(-1, tok_blk.view(1, 1, n_cand).expand(batch_size, n_heads, n_cand)).contiguous()
            scores[..., :pl] = float('inf')                                             # protect prompt
            ids_to_keep = scores.topk(K, dim=-1, largest=True).indices                  # keeps the highest-scored blocks' tokens
            ids_to_keep, _ = torch.sort(ids_to_keep, dim=-1)                            # chronological compaction

        elif self.MODE in ('attn_pp', 'attn_rkv_pp'):   # snapkv_pp / rkv_pp: attention (or R-KV) scoring + PROMPT PROTECTION
            # attn_rkv_pp added 08-03 to close the matched-protection table: R-KV was the ONE headline method
            # with no protected form. No new scoring code is needed -- line ~254 already fills attn_scores for any
            # mode containing 'attn', and line ~256 (`'rkv' in MODE and MODE != 'rkv_official'`) already applies
            # the redundancy mix attn*lambda - sim*(1-lambda). The body below is signal-agnostic, so it protects
            # the prompt and top-K's whatever score was computed (R-KV's score may be negative; topk is unaffected).
            # LAUNCH REQUIREMENT: --rkv_lambda 0.5 is MANDATORY for attn_rkv_pp. my_utils.py declares --rkv_lambda
            # with NO argparse default, and parallel_run_hf forwards it only under `if args.rkv_lambda:`, so
            # omitting it leaves rkv_lambda=None and line ~258 raises TypeError on `attn_scores * None`.
            # within K. Mirrors caote_pp/vase_pp: force-keep the prompt [0:pl), fill the remaining K-pl by the
            # attention score. Completes the 4-signal matched-protection contrast (random/vase/caote/snapkv all _pp).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            attn_scores[..., :pl] = float('inf')          # force-keep the prompt (top-K always includes it)
            ids_to_keep = attn_scores.topk(K, dim=-1, largest=True).indices

        elif self.MODE in ('mix_step_pp', 'mix_head_pp', 'union_topk_pp'):
            # T1.0 DIVERSITY LADDER (pure-heuristic mixes, no random injected into the *selection*): does
            # combining BIASED samplers alone recover the coverage that makes random_pp work? All three modes
            # build the keep-set from the three heuristic signals {vase, caote, snapkv} under matched prompt
            # protection (+inf at [0:pl)) and matched budget:
            #   mix_step_pp  — each eviction round (per layer) uses ONE heuristic, drawn uniformly at random;
            #   mix_head_pp  — KV head h permanently uses heuristic h % 3 (strongest simultaneous diversity);
            #   union_topk_pp— exact interleaved-union: h1[0],h2[0],h3[0],h1[1],... dedup'd, first K kept.
            # The vase component reproduces range_sink_sample_attn as a full RANKING: top-n_large by value-range
            # (sinks pinned) first, then Gumbel-top-k on log-attention = the same sampling-without-replacement
            # distribution as multinomial_sample, but expressed as an ordering (run with --n_large K/4, faithful).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            s_snap = attn_scores.float()
            a = attn_scores.clamp(1e-6, 1 - 1e-6)
            o = (a.unsqueeze(-1) * v_candidates).sum(dim=2, keepdim=True)
            s_caote = ((a / (1 - a)) * (v_candidates - o).norm(dim=-1)).float()
            vmag = (v_candidates.amax(-1) - v_candidates.amin(-1)).float()
            vmag[..., :self.n_sink] = float('inf')
            n_large_eff = min(self.n_large, max(K - pl, 1))
            ids_large = vmag.topk(n_large_eff, dim=-1, largest=True).indices          # desc by value-range
            u = torch.rand_like(s_snap).clamp(1e-12, 1.0 - 1e-7)   # strictly inside (0,1): no NaN/inf from the logs
            gumbel = -torch.log(-torch.log(u))                     # (NaN here would silently outrank +inf in topk!)
            s_vase = torch.log(s_snap.clamp_min(1e-12)) + gumbel
            lift = 1e6 - torch.arange(n_large_eff, device=self.device, dtype=torch.float)  # preserve range order
            s_vase.scatter_(-1, ids_large, lift.view(1, 1, -1).expand_as(ids_large).contiguous())
            heuristics = (s_vase, s_caote, s_snap)
            if self.MODE == 'mix_step_pp':
                j = int(torch.randint(3, (1,), device=self.device).item())    # fresh draw per (layer, round)
                score = heuristics[j].clone()
                score[..., :pl] = float('inf')
                ids_to_keep = score.topk(K, dim=-1, largest=True).indices
            elif self.MODE == 'mix_head_pp':
                score = torch.empty_like(s_snap)
                for j, s in enumerate(heuristics):
                    score[:, j::3] = s[:, j::3]                               # head h -> heuristic h % 3
                score[..., :pl] = float('inf')
                ids_to_keep = score.topk(K, dim=-1, largest=True).indices
            else:  # union_topk_pp: priority(pos) = min_j (3*rank_j(pos) + j); keep the K best priorities.
                arange_n = torch.arange(n_cand, device=self.device).view(1, 1, n_cand).expand_as(s_snap)
                prio = None
                for j, s in enumerate(heuristics):
                    order = s.argsort(dim=-1, descending=True)                # [B,Hkv,n_cand]
                    rk = torch.empty_like(order)
                    rk.scatter_(-1, order, arange_n)
                    pj = 3 * rk + j
                    prio = pj if prio is None else torch.minimum(prio, pj)
                prio[..., :pl] = -1                                           # prompt outranks everything
                ids_to_keep = (-prio.float()).topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'random_strat_pp':   # T1.0 MAX-COVERAGE control: protect prompt [0:pl) + STRATIFIED
            # random scatter — history is split into (K-pl) equal chunks per head and exactly one random token
            # is kept per chunk (segmented argmax of iid uniforms). Same budget; upper end of the coverage
            # interpolation (random_cluster_pp = lower end, random_pp = middle).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            n_hist = n_cand - pl
            m = max(min(K - pl, n_hist), 1)                                   # number of chunks
            r = torch.rand(batch_size, n_heads, n_hist, device=self.device)
            chunk = (torch.arange(n_hist, device=self.device) * m // n_hist).clamp_(max=m - 1)
            chunk_e = chunk.view(1, 1, n_hist).expand_as(r)
            # segmented argmax via sort (portable; no reduce kernels — scatter_reduce_(amax) wedges in the
            # real generation context on H200/sm_90, hanging strat before it emits any token; isolated it
            # runs fine, so the hang is stream/kernel-state dependent). chunk ids are monotone => segments
            # contiguous. float64 key: at prod chunk counts (m~1948) chunk*4~7788 whose float32 ULP (~5e-4)
            # would swallow r's low bits; double preserves r. GPU-verified bit-identical to the amax path (60 trials, 07-19).
            key = chunk_e.double() * 4.0 - r.double()                        # ascending sort => chunk-ordered, best-r first in chunk
            order = key.argsort(dim=-1)
            sc = chunk_e.gather(-1, order)
            first = torch.ones_like(sc, dtype=torch.bool)
            first[..., 1:] = sc[..., 1:] != sc[..., :-1]                      # first of each contiguous chunk run = winner
            is_max = torch.zeros_like(r)
            is_max.scatter_(-1, order, first.float())                        # ~1 winner per chunk
            score = torch.zeros(batch_size, n_heads, n_cand, device=self.device)
            score[..., pl:] = r + 2.0 * is_max                                # winners first, then best losers
            score[..., :pl] = float('inf')
            ids_to_keep = score.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'random_cluster_pp': # T1.0 MIN-COVERAGE (anti-diverse) control: protect prompt [0:pl) +
            # one CONTIGUOUS window of K-pl tokens at a random center per head per eviction. The picks are still
            # "random" (random placement) but maximally clustered -> minimal per-head coverage of the trace.
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            n_cand = v_candidates.size(2)
            W = K - pl
            pos = torch.arange(n_cand, device=self.device, dtype=torch.float).view(1, 1, n_cand)
            lo, hi = pl + W / 2.0, max(n_cand - W / 2.0, pl + W / 2.0)
            center = lo + torch.rand(batch_size, n_heads, 1, device=self.device) * (hi - lo)
            score = -(pos - center).abs().expand(batch_size, n_heads, n_cand).contiguous()
            score[..., :pl] = float('inf')
            ids_to_keep = score.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'range_sink_sample_attn_pp':   # vase_pp: VaSE selection (value-range + attn-sample) +
            # PROMPT PROTECTION within K. Isolates the selection signal vs random_pp: both force-keep the prompt
            # [0:pl), then vase_pp fills the remaining K-pl by VaSE's value-magnitude+attention rule while random_pp
            # fills it uniformly at random. ('range'+'sink'+'attn' in the name -> v_magnitude & attn_scores already
            # computed above with sink protection.) Same budget (K + 64 residual = token_budget).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            v_magnitude[..., :pl] = float('-inf')            # drop prompt from the value-range pool (kept separately)
            attn_scores[..., :pl] = 0.0                      # drop prompt from the attention-sampling pool
            Kv = K - pl
            n_large_eff = min(self.n_large, Kv)
            ids_large = v_magnitude.topk(n_large_eff, dim=-1, largest=True).indices
            attn_scores.scatter_(dim=-1, index=ids_large, value=0.0)
            n_samp = Kv - n_large_eff
            prompt_ids = torch.arange(pl, device=self.device, dtype=torch.long).view(1, 1, pl).expand(batch_size, n_heads, pl)
            if n_samp > 0:
                ids_attn = multinomial_sample(attn_scores, n_samp)
                ids_to_keep = torch.cat([prompt_ids, ids_large, ids_attn], dim=-1)  # pl + n_large_eff + n_samp = K
            else:
                ids_to_keep = torch.cat([prompt_ids, ids_large], dim=-1)

        elif self.MODE == 'caote':             # CAOTE: a/(1-a) * ||v - o|| per head (o = attention output)
            a = attn_scores.clamp(1e-6, 1 - 1e-6)                          # [B, Hkv, n_cand]
            o = (a.unsqueeze(-1) * v_candidates).sum(dim=2, keepdim=True)   # [B, Hkv, 1, D]
            err = (a / (1 - a)) * (v_candidates - o).norm(dim=-1)          # [B, Hkv, n_cand]
            ids_to_keep = err.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'caote_pp':          # FAIR: protect prompt [0:pl) + CAOTE output-error for the rest.
            # Mirrors random_pp/vase_pp protection -> isolates CAOTE's (output-error) selection signal vs random
            # scatter at matched budget. CAOTE is the strongest cheap signal, so this is the hardest test of the
            # 'random scatter >= selection under protection' claim. Same K (+64 residual = token_budget).
            pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            a = attn_scores.clamp(1e-6, 1 - 1e-6)
            o = (a.unsqueeze(-1) * v_candidates).sum(dim=2, keepdim=True)
            err = (a / (1 - a)) * (v_candidates - o).norm(dim=-1)          # [B, Hkv, n_cand]
            err[..., :pl] = float('inf')                                  # force-keep the prompt (top-K includes it)
            ids_to_keep = err.topk(K, dim=-1, largest=True).indices       # pl prompt + CAOTE top-(K-pl)

        elif self.MODE == 'cusa':              # head-unified attn (mean over heads) + r=0.25 recency reserve
            # NOTE (index-vs-time drift): the recency reserve below identifies "recent" by raw cache index; after a
            # compaction the [0:K) block is stored in keep-order, so the reserve is only the CURRENT tail each round
            # (previously reserved slots must re-earn their place via the attention score). This matches a
            # sliding-window reading of the reserve but is stated here explicitly; cusa is a labeled proxy, not
            # LessIsMore's published CUSA.
            n_cand = v_candidates.size(2)
            n_rec = max(1, int(round(0.25 * K)))
            uni = attn_scores.mean(dim=1).clone()                          # [B, n_cand] cross-head-unified
            uni[:, n_cand - n_rec:] = float('-inf')                        # reserve the recency tail separately
            top_ids = uni.topk(K - n_rec, dim=-1, largest=True).indices    # [B, K-n_rec]
            rec_ids = torch.arange(n_cand - n_rec, n_cand, device=self.device, dtype=torch.long)
            rec_ids = rec_ids.view(1, -1).expand(batch_size, -1)           # [B, n_rec]
            keep = torch.cat([top_ids, rec_ids], dim=-1)                   # [B, K]
            ids_to_keep = keep.unsqueeze(1).expand(batch_size, n_heads, K).contiguous()

        elif self.MODE in ('triattn', 'triattn_pp'):   # TriAttention (arXiv 2604.04921): query-INDEPENDENT eviction.
            # Score each key by expected FUTURE attention using calibrated per-(head,band) query centers:
            #   S_trig = mean_{d in D} sum_f ||E[q_f]|| ||k_f|| cos(w_f*(Δ+d) + phi_f)
            #   S_mlr  = sum_f (1-R_f) E[||q_f||] ||k_f||
            # ||k_f|| = band-f magnitude of the post-RoPE key (rotation-invariant); Δ = recency rank (like recency mode).
            # Calibrated stats file via env TRIATTN_STATS (per model); aggregated Hq->Hkv (mean over query-group).
            import sys as _sys
            _mod = _sys.modules[__name__]
            if not hasattr(_mod, '_triattn_cache'):
                _mod._triattn_cache = {}
            _path = os.environ['TRIATTN_STATS']
            if _path not in _mod._triattn_cache:
                _d = torch.load(_path, map_location='cpu')
                _Hq, _Hkv, _hd, _Ln = _d['Hq'], _d['Hkv'], _d['head_dim'], _d['n_layers']
                _grp, _hf = _Hq // _Hkv, _hd // 2
                def _redm(t):                                  # real [Hq or Hkv, half] -> [Hkv, half]: plain mean ok
                    return t.reshape(_Hkv, _grp, _hf).mean(1) if t.shape[0] == _Hq else t
                # FIX: reduce q_center as a COMPLEX vector over the GQA group (circular-correct), then derive
                # |E[q_f]| and Q-K phase from the reduced centers. Old code averaged phi (an ANGLE) arithmetically
                # -> ~0.59 rad (19% of pi) corruption of s_trig; measured 15pp acc drop vs caote/random_unified.
                qc = torch.stack([_d['stats'][l]['q_center'].reshape(_Hkv, _grp, _hf).mean(1) for l in range(_Ln)])  # [L,Hkv,half] cfloat
                kc = torch.stack([_d['stats'][l]['k_center'] for l in range(_Ln)])                                    # [L,Hkv,half] cfloat
                qmag = qc.abs()
                phi = qc.angle() - kc.angle()
                qnorm = torch.stack([_redm(_d['stats'][l]['q_absmean']) for l in range(_Ln)])
                R = torch.stack([_redm(_d['stats'][l]['R']) for l in range(_Ln)])
                omega = float(_d['rope_theta']) ** (-(torch.arange(_hf).float()) * 2.0 / _hd)
                _mod._triattn_cache[_path] = dict(
                    qmag=qmag.to(self.device).float(), qnorm=qnorm.to(self.device).float(),
                    R=R.to(self.device).float(), phi=phi.to(self.device).float(),
                    omega=omega.to(self.device).float(), D=[2 ** i for i in range(0, 15)])
            _st = _mod._triattn_cache[_path]
            _L = int(self.layer_idx); _hf = head_dim // 2
            qmag = _st['qmag'][_L]; qnorm = _st['qnorm'][_L]; R = _st['R'][_L]; phi = _st['phi'][_L]  # [Hkv, half]
            omega = _st['omega']; Doff = _st['D']
            n_cand = v_candidates.size(2)
            kf = torch.sqrt(k_candidates[..., :_hf] ** 2 + k_candidates[..., _hf:] ** 2)      # [B,Hkv,n_cand,half]
            delta = torch.arange(n_cand - 1, -1, -1, device=self.device, dtype=torch.float)   # newest=0 .. oldest=n_cand-1
            s_mlr = (((1 - R) * qnorm).view(1, n_heads, 1, _hf) * kf).sum(-1)                  # [B,Hkv,n_cand]
            s_trig = torch.zeros_like(s_mlr)
            for _dd in Doff:
                ang = omega.view(1, 1, _hf) * (delta.view(1, n_cand, 1) + _dd) + phi.view(n_heads, 1, _hf)  # [Hkv,n_cand,half]
                s_trig = s_trig + (qmag.view(1, n_heads, 1, _hf) * kf * torch.cos(ang).unsqueeze(0)).sum(-1)
            score = s_trig / len(Doff) + s_mlr                                                # [B,Hkv,n_cand]
            if self.MODE == 'triattn_pp':
                pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
                score[..., :pl] = float('inf')                                               # force-keep the prompt
            ids_to_keep = score.topk(K, dim=-1, largest=True).indices

        elif self.MODE == 'triattn_pl':   # FAITHFUL TriAttention (per_layer_perhead), verbatim scoring
            from kvcompress.engine.triattn_official import (
                unrotate_key_complex, compute_frequency_statistics_from_means, score_keys_for_round)
            _st = _load_triattn_pl(self.device)
            # Same flag as triattn_ph: faithful = normalised; TRIATTN_NO_NORM=1 reproduces the
            # pre-08-04 unnormalised aggregation for auditing old cells only.
            _TRI_NORM = not _tri_env_on('TRIATTN_NO_NORM')
            _L = int(self.layer_idx); _grp = _st['grp']
            n_cand = v_candidates.size(2)
            pos_cand = self.positions[:, :, :n_cand]                      # [B, Hkv, n_cand] absolute positions
            round_start = int(self.cumulative_length)
            _pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)        # FIX: pin the prompt (official pins prefill; also budget-fair vs *_pp)
            keep_bh = []
            for b in range(batch_size):
                kv_keep = []
                for kv in range(n_heads):
                    posk = pos_cand[b, kv]                                # [n_cand]
                    k_unrot = unrotate_key_complex(                       # post-RoPE -> pre-RoPE (for phi)
                        k_candidates[b, kv].float(), posk, _st['omega'], style=_st['style'])
                    grp_scores = []
                    for hq in range(kv * _grp, (kv + 1) * _grp):         # this KV head's query heads
                        amp, phi, extra = compute_frequency_statistics_from_means(
                            _st['qmc'][_L, hq], _st['qam'][_L, hq], k_unrot, style=_st['style'])
                        grp_scores.append(score_keys_for_round(
                            posk, round_start, amp, phi, _st['omega'], extra,
                            _st['offsets'], 'mean', _st['fss']))
                    # PER-HEAD Z-NORMALISATION, same as the triattn_ph branch and for the same reason.
                    # The reference normalises this layer's per-head score matrix over the candidate
                    # tokens BEFORE the group max: methods/triattention.py:502-506, then the max at
                    # :531-535. `amax` is scale-sensitive, so unnormalised the query head with the
                    # largest raw scale wins every token in the group and the other grp-1 heads never
                    # contribute at all.
                    # (b9818c3 fixed only triattn_ph and left this branch raw -- caught by the 08-04
                    # adversarial re-audit, 3/3 refuters failed to refute.)
                    _gs = torch.stack(grp_scores, 0)                     # [grp, n_cand]
                    if _TRI_NORM:
                        _d = _gs[:, _pl:] if _pl > 0 else _gs
                        if _d.size(-1) > 1:
                            _gs = (_gs - _d.mean(-1, keepdim=True)) / _d.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)
                    agg = _gs.amax(0)                                    # max over group (layer_perhead_aggregation="max")
                    if _pl > 0:
                        agg[:_pl] = float('inf')                         # prefill/prompt always kept (official pins prefill)
                    kv_keep.append(agg.topk(K, largest=True).indices)   # [K]
                keep_bh.append(torch.stack(kv_keep, 0))                  # [Hkv, K]
            ids_to_keep = torch.stack(keep_bh, 0)                        # [B, Hkv, K]

        elif self.MODE == 'triattn_ph':   # FAITHFUL TriAttention (per_head_pruning): ONE per-KV-head keep-set
            # SHARED across ALL layers. Verified as the scheme behind the paper's reported MATH-500 numbers
            # (per_head_pruning=True, per_layer_perhead_pruning=False). Score for KV head h at position p =
            #   mean over layers L of ( max over that head's query-group of score_keys(layer L, hq, k_L[p], p) ).
            # Computed ONCE per eviction round (all layers hit the same cur_len; the candidate region
            # [0:cur_len-residual] is already written in every layer since residual>=1), then reused by every layer.
            from kvcompress.engine.triattn_official import (
                unrotate_key_complex, compute_frequency_statistics_from_means, score_keys_for_round)
            import sys as _sys
            _mod = _sys.modules[__name__]
            _st = _load_triattn_pl(self.device)
            # Faithful = normalised (see the block below). TRIATTN_NO_NORM=1 reproduces the
            # unnormalised aggregation for auditing pre-08-04 cells; it is not TriAttention.
            _TRI_NORM = not _tri_env_on('TRIATTN_NO_NORM')
            _grp = _st['grp']; n_cand = v_candidates.size(2)
            round_start = int(self.cumulative_length)
            _pl = min(int(getattr(self, 'prompt_len', 0)), K - 1)
            # MEMO KEY (08-04, DEFAULT FLIPPED TO FAITHFUL).
            # The memo itself is necessary and correct in intent: triattn_ph is the engine's only
            # CROSS-LAYER selector (score = mean over layers of max over query-group), but the eviction
            # hook fires once per layer -- 40x per round on 14B -- so the keep-set must be computed once
            # and shared across that round's layers. Every other mode is per-layer, scores only its own
            # K/V, and therefore has no memo and no cache-invalidation surface at all. That is why this
            # was the only mode that could go stale.
            # The defect was purely the key: the cached keep-set DEPENDS on round_start (it is the
            # token-age term at `bd = (round_start - pos_b) + offsets` below), but the old key omitted it
            # while every other component pins to a constant after the first compaction (cache_seqlens[0]
            # -> token_budget+residual_length, n_cand -> token_budget). So "same round" never registered a
            # miss: the keep-set was computed ONCE PER PROCESS, and since `_triattn_ph_shared` is a module
            # global that is never reset, it persisted ACROSS PROBLEMS too. triattn_ph then re-applied one
            # frozen set of slot INDICES forever -- an index-scored policy wearing a content-scored name.
            # Verified against the official implementation (refs/triattention/.../core/{compressor,scoring}.py):
            # compress() takes round_start as an explicit per-call argument and rescores every round; its
            # only cache is trig_cache, a content-independent sin/cos table. There is NO keep-set memo
            # upstream, so the memo was ours and per-round rescoring is the published algorithm.
            # round_start strictly increases per round, so including it gives a HIT across this round's
            # layers (optimisation preserved) and a MISS at the next round (correctness restored).
            # prompt_len is included as a cheap cross-problem guard: the memo holds a single slot, so two
            # problems would otherwise need only a coincident round_start to collide.
            # TRIATTN_FROZEN_MEMO=1 reproduces the pre-08-04 frozen-keep-set cells and is for auditing
            # those cells ONLY -- it is not TriAttention.
            # (TRIATTN_FIX_MEMO=1 is still accepted and is now a redundant no-op, so launchers written
            # against TRIATTN_FIX_MEMO -- and their `grep -c TRIATTN_FIX_MEMO` engine guards -- keep working.)
            _key = (int(self.cache_seqlens[0].item()), int(batch_size), int(n_cand), round_start, _pl)
            if _tri_env_on('TRIATTN_FROZEN_MEMO'):
                _key = (int(self.cache_seqlens[0].item()), int(batch_size), int(n_cand))
            _shared = getattr(_mod, '_triattn_ph_shared', None)
            if _shared is None or _shared.get('key') != _key:       # compute the shared keep-set once per round
                # VECTORIZED score (bit-identical to the reference nested loop; validated diff 0.0, keep-sets equal).
                # Loop only over (layer, batch); batch the (KV-head × query-group) scoring into tensor ops -> ~100x
                # fewer kernel launches than the per-(L,b,kv,hq) loop (which cost ~18s/eviction-round).
                sib = self._sibling_layers
                _F = head_dim // 2; _om = _st['omega']; _fss = _st['fss']; _off = _st['offsets']; _O = _off.numel()
                _qmc = _st['qmc']; _qam = _st['qam']
                acc = torch.zeros(batch_size, n_heads, n_cand, device=self.device)
                for _L, lyr in enumerate(sib):
                    kperm = lyr.k_cache[:, :n_cand].transpose(1, 2).float()   # [B, Hkv, n_cand, D]
                    pos_L = lyr.positions[:, :, :n_cand].float()             # [B, Hkv, n_cand]
                    kc_all = torch.complex(kperm[..., :_F], kperm[..., _F:])  # [B, Hkv, n_cand, F] (half-split on last dim)
                    qc = _qmc[_L].view(n_heads, _grp, _F); qa = _qam[_L].view(n_heads, _grp, _F); qabs = qc.abs()
                    for b in range(batch_size):
                        pos_b = pos_L[b]                                     # [Hkv, n_cand]
                        ang = pos_b.unsqueeze(-1) * _om.view(1, 1, _F)
                        kpre = kc_all[b] * torch.polar(torch.ones_like(ang), -ang)   # [Hkv, n_cand, F] un-rotated
                        kabs = kpre.abs()
                        amp = qabs.view(n_heads, _grp, 1, _F) * kabs.view(n_heads, 1, n_cand, _F)
                        rel = qc.view(n_heads, _grp, 1, _F) * kpre.conj().view(n_heads, 1, n_cand, _F)
                        phi = torch.atan2(rel.imag, rel.real)
                        extra = (qa - qabs).view(n_heads, _grp, 1, _F) * kabs.view(n_heads, 1, n_cand, _F)
                        bd = (round_start - pos_b).unsqueeze(-1) + _off.view(1, 1, _O)      # [Hkv, n_cand, O]
                        phase = bd.view(n_heads, 1, n_cand, _O, 1) * _om.view(1, 1, 1, 1, _F) + phi.unsqueeze(-2)
                        base = (amp.unsqueeze(-2) * _fss.view(1, 1, 1, 1, _F) * torch.cos(phase)).sum(-1)
                        add = (extra * _fss.view(1, 1, 1, _F)).sum(-1, keepdim=True)
                        _s = (base + add).mean(-1)                          # [Hkv, grp, n_cand] per QUERY-head score
                        # PER-HEAD Z-NORMALISATION (08-04). The reference normalises each sampled head's
                        # score vector over the candidate tokens BEFORE the group-max and the cross-layer
                        # mean: methods/triattention.py:259-262, and it is ON in every official runner and
                        # runtime -- scripts/worker.py:341 (default=True), scripts/cli.py:307
                        # (setdefault True), vllm/core/config.py:71 and vllm/runtime/config.py:51
                        # (sparse_normalize_scores=True), longlive configs (kv_normalize_scores: true).
                        # Only the bare dataclass default is False, which is why this was missed.
                        # It is NOT cosmetic: a per-head affine rescale is rank-preserving for one head,
                        # but this score is mean_layers(max_over_query_group(.)) and BOTH reductions are
                        # scale-sensitive -- unnormalised, the (layer, head) pairs with the largest raw
                        # scale dominate the keep-set.
                        # Statistics are taken over DECODE tokens only, matching the reference, whose score
                        # matrix never contains prefill at all; the +inf prompt pin is applied afterwards.
                        if _TRI_NORM:
                            _d = _s[..., _pl:] if _pl > 0 else _s
                            if _d.size(-1) > 1:
                                _s = (_s - _d.mean(-1, keepdim=True)) / _d.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)
                        acc[b] += _s.amax(1)                                # max over query-group -> [Hkv, n_cand]
                acc /= len(sib)                                         # mean over layers
                if _pl > 0:
                    acc[:, :, :_pl] = float('inf')                     # prefill/prompt always kept
                ids = acc.topk(K, dim=-1, largest=True).indices        # [B, Hkv, K]
                _mod._triattn_ph_shared = {'key': _key, 'ids': ids}
            ids_to_keep = _mod._triattn_ph_shared['ids']              # [B, Hkv, K] shared across layers

        else:
            raise NotImplementedError(f"{self.MODE} not matched in cache_utils.py!")

        assert ids_to_keep.size(-1) == K

        # UNIVERSAL CHRONOLOGICAL COMPACTION (2026-07-09): sort every mode's keep-set before the gather so the
        # compacted cache is always stored in time order. For content-scored modes this is a pure no-op on the
        # attention output (post-RoPE keys are permutation-invariant) and on the keep-set itself; for index-scored
        # modes (recency_pp, random_pp_rec's tail, random_strat_pp's chunks, random_cluster_pp's window, cusa's
        # recency reserve) it is REQUIRED — "index = time" only holds if compaction preserves chronology. This
        # subsumes the per-mode recency-fix sorts above and removes the whole bug class for future modes.
        # (NOTE: pre-2026-07-09 `cusa` cells ran with index-vs-time drift in their recency reserve — see the
        # cusa branch comment; from this change onward the reserve is genuinely the most-recent tail.)
        ids_to_keep, _ = torch.sort(ids_to_keep, dim=-1)

        # T1.0 KEEP-SET LOGGER (OFF unless KEEPLOG=<dir> -> zero cost for normal runs). Per eviction round,
        # batch 0, layers with layer_idx % KEEPLOG_STRIDE == 0, appends one CSV row per layer to
        # <KEEPLOG>/keeplog_<pid>.csv with the coverage/diversity metrics behind the accuracy-vs-coverage
        # money figure: (a) cross-head union of kept cache slots; (b) positional-coverage entropy of kept
        # NON-PROMPT tokens over the full trace so far (needs self.positions -> _track_positions is forced on
        # when KEEPLOG is set); (c) mean pairwise cross-head Jaccard of keep-sets; (d) kept-value redundancy
        # (mean |cos| of pairwise value vectors, strided subsample <=512/head).
        if os.environ.get('KEEPLOG') and self.layer_idx % int(os.environ.get('KEEPLOG_STRIDE', '1')) == 0:
            with torch.no_grad():
                n_cand_i = v_candidates.size(2)
                ids0 = ids_to_keep[0]                                          # [Hkv, K] (chronological)
                Hkv, Ksz = ids0.shape
                union = int(torch.unique(ids0).numel())
                pl_abs = int(getattr(self, 'prompt_len', 0))
                C = max(int(self.cumulative_length), 1)
                pos0 = torch.gather(self.positions[0, :, :n_cand_i], -1, ids0) # [Hkv, K] absolute positions
                hist_pos = pos0.reshape(-1).float()
                hist_pos = hist_pos[hist_pos >= pl_abs]                        # scatter differs on history only
                nbins = 64
                if hist_pos.numel() > 0 and C > pl_abs:
                    b_ = ((hist_pos - pl_abs) * nbins / max(C - pl_abs, 1)).long().clamp_(0, nbins - 1)
                    p = torch.bincount(b_, minlength=nbins).float()
                    p = p / p.sum()
                    p = p[p > 0]
                    cov_ent = float(-(p * p.log()).sum().item() / math.log(nbins))
                else:
                    cov_ent = 0.0
                memb = torch.zeros(Hkv, n_cand_i, device=self.device)
                memb.scatter_(1, ids0, 1.0)
                inter = memb @ memb.T                                          # [Hkv, Hkv]
                iu = torch.triu_indices(Hkv, Hkv, 1, device=self.device)
                jac = float((inter[iu[0], iu[1]] / (2 * Ksz - inter[iu[0], iu[1]])).mean().item()) if Hkv > 1 else 1.0
                stride = max(1, Ksz // 512)
                sub = ids0[:, ::stride]                                        # [Hkv, S]
                vk = torch.gather(v_candidates[0].float(), 1,
                                  sub[..., None].expand(-1, -1, head_dim))     # [Hkv, S, D]
                vn = vk / (vk.norm(dim=-1, keepdim=True) + 1e-8)
                Gm = torch.matmul(vn, vn.transpose(1, 2)).abs()                # [Hkv, S, S]
                S = sub.size(1)
                vred = float(((Gm.sum() - Gm.diagonal(dim1=1, dim2=2).sum()) /
                              max(Hkv * S * (S - 1), 1)).item()) if S > 1 else 0.0
                # prompt retention (E1): fraction of prompt positions [0, pl_abs) still in the cache —
                # union across heads + per-head mean. rp/_pp modes are 1.0 by construction; the informative
                # case is unprotected selectors (vase) on long prompts.
                if pl_abs > 0:
                    pk_u = float(torch.unique(pos0[pos0 < pl_abs]).numel() / pl_abs)
                    pk_h = float(((pos0 < pl_abs).sum(dim=-1).float() / pl_abs).mean().item())
                else:
                    pk_u, pk_h = 1.0, 1.0
                _dir = os.environ['KEEPLOG']
                os.makedirs(_dir, exist_ok=True)
                _fp = os.path.join(_dir, f"keeplog_{os.getpid()}.csv")
                _new = not os.path.exists(_fp)
                with open(_fp, 'a') as _f:
                    if _new:
                        _f.write("mode,layer,cum_len,n_cand,K,prompt_len,union,union_frac_slots,"
                                 "union_frac_headsxK,cov_entropy,jaccard,v_redundancy,"
                                 "prompt_kept_union,prompt_kept_perhead\n")
                    _f.write(f"{self.MODE},{self.layer_idx},{C},{n_cand_i},{Ksz},{pl_abs},{union},"
                             f"{union / max(n_cand_i, 1):.6f},{union / max(Hkv * Ksz, 1):.6f},"
                             f"{cov_ent:.6f},{jac:.6f},{vred:.6f},{pk_u:.6f},{pk_h:.6f}\n")
                # WHOLE-EVIDENCE position dumps (KEEPLOG_POS=1; survival-calculus program 07-22): every
                # KEEPLOG_POS_STRIDE-th logged event, dump per-head ABSOLUTE kept positions [Hkv, K] so
                # position-level survival curves / cross-head exclusion correlation / all-copies-lost are
                # computable offline for ANY method. ~32KB compressed/dump; stride 16 -> ~200MB/cell.
                if os.environ.get('KEEPLOG_POS'):
                    self._kp_ev = getattr(self, '_kp_ev', -1) + 1
                    if self._kp_ev % int(os.environ.get('KEEPLOG_POS_STRIDE', '16')) == 0:
                        import numpy as _np
                        _pd = os.path.join(_dir, 'pos')
                        os.makedirs(_pd, exist_ok=True)
                        _np.savez_compressed(
                            os.path.join(_pd, f"p{os.getpid()}_L{self.layer_idx}_e{self._kp_ev}.npz"),
                            pos=pos0.cpu().numpy().astype('int32'), cum_len=C, prompt_len=pl_abs,
                            mode=self.MODE)

        # E1 union-coverage probe (OFF unless LOG_UNION set -> zero cost for normal runs). ids_to_keep is
        # [B, Hkv, K] here. union = distinct positions kept across all KV-heads (batch 0); union/(Hkv*K) in
        # [1/Hkv, 1]: 1 = per-head disjoint (max breadth), 1/Hkv = all heads identical (random_unified / cusa).
        if os.environ.get('LOG_UNION'):
            with torch.no_grad():
                _union = int(torch.unique(ids_to_keep[0]).numel())
                _ncand = int(v_candidates.size(2))
                with open(os.environ.get('UNION_LOG', '/tmp/union_log.csv'), 'a') as _f:
                    _f.write(f"{self.MODE},{self.layer_idx},{n_heads},{K},{_ncand},{_union},"
                             f"{_union / max(_ncand, 1):.6f},{_union / max(n_heads * K, 1):.6f}\n")

        _pos_keep = (torch.gather(self.positions[:, :, :v_candidates.size(2)], 2, ids_to_keep)
                     if self._track_positions else None)

        ids_to_keep = ids_to_keep[..., None].expand(-1, -1, -1, head_dim)

        k_compress = torch.gather(k_candidates, dim=2, index=ids_to_keep)
        v_compress = torch.gather(v_candidates, dim=2, index=ids_to_keep)
        self.k_cache[:, :K].copy_(k_compress.transpose(1, 2))
        self.v_cache[:, :K].copy_(v_compress.transpose(1, 2))

        # Move recency buffer
        self.k_cache[:, K:self.token_budget].copy_(self.k_cache[:, cur_len - residual:cur_len])
        self.v_cache[:, K:self.token_budget].copy_(self.v_cache[:, cur_len - residual:cur_len])
        if self._track_positions:   # compact per-head positions identically (kept candidates + residual)
            self.positions[:, :, K:self.token_budget] = self.positions[:, :, cur_len - residual:cur_len]
            self.positions[:, :, :K] = _pos_keep

        self.cache_seqlens.fill_(self.token_budget)

        # Budget-parity guard: EVERY mode (faithful or *_pp) must compact to exactly token_budget.
        # _pp protects the prompt WITHIN K (pl capped at K-1), so it never inflates the budget.
        if not EvictLayer._parity_logged:
            pl_dbg = min(int(getattr(self, 'prompt_len', 0)), K - 1) if self.MODE.endswith('_pp') else None
            print(f"[budget-parity] mode={self.MODE} layer={self.layer_idx} K_sel={K} residual={residual} "
                  f"token_budget={self.token_budget} final_cache_seqlen={int(self.cache_seqlens[0].item())} "
                  f"prompt_len={getattr(self, 'prompt_len', None)} pp_pl={pl_dbg}",
                  file=sys.stderr, flush=True)
            assert int(self.cache_seqlens[0].item()) == self.token_budget, \
                f"budget parity violated: {int(self.cache_seqlens[0].item())} != {self.token_budget}"
            assert pl_dbg is None or pl_dbg <= K - 1, "prompt protection exceeds K budget"
            EvictLayer._parity_logged = True

        if self.verbose and self.layer_idx == 0:
            print(f'Buffer: {cur_len - residual}:{cur_len} -> {K}:{self.token_budget}', end=' | ')
            print(f'Cache size: {cur_len} -> {self.token_budget}')

    def get_seq_length(self) -> int:
        # use cumulative_length (not cur_len) for the position_embeddings/rope
        return self.cumulative_length

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized:
            return
        self.k_cache = self.k_cache[indices].contiguous()
        self.v_cache = self.v_cache[indices].contiguous()
        self.cache_seqlens = self.cache_seqlens[indices].contiguous()
        if hasattr(self, 'queries'):
            self.queries = self.queries[indices].contiguous()
        if hasattr(self, 'G'):
            self.G = self.G[indices].contiguous()
        if hasattr(self, 'positions'):   # triattn_pl per-batch position buffer must shrink with the batch too
            self.positions = self.positions[indices].contiguous()

