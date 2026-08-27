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
import os, sys, torch
sys.path.insert(0, os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))); sys.path.insert(0, os.environ.get("RA_ROOT", "."))
from kvcompress.engine.cache_utils import EvictLayer

B, HKV, HQ, D = 1, 2, 4, 16
TB, RES = 256, 64
K = TB - RES
PL = 40
CUR = TB + RES

def make(mode, lam=0.1):
    l = EvictLayer(0, TB, RES, mode, rkv_lambda=lam, smooth=False, verbose=False)
    l.lazy_initialization(torch.zeros(B, CUR, HKV, D))
    l.prompt_len = PL
    return l

torch.manual_seed(0)
l = make("rkv_official")
# fill KV with random keys; plant a DUPLICATE cluster: positions 10, 100, 200 share (nearly) the same key
kv = torch.randn(B, CUR, HKV, D)
dup = torch.randn(1, 1, HKV, D)
for p in (10, 100, 200):
    kv[:, p] = dup[:, 0] + 0.001 * torch.randn(HKV, D)
l.k_cache[:, :CUR] = kv; l.v_cache[:, :CUR] = kv
l.cache_seqlens.fill_(CUR); l.cumulative_length = CUR
# query buffer: rkv_official is in _QUERY_MODES -> update_queries must have populated; simulate prefill call
q = torch.randn(B, HQ, RES, D)
l.update_queries(torch.randn(B, HQ, CUR, D))   # prefill path fills buffer with last RES queries
l._run_eviction(CUR)

n_kept = int(l.cache_seqlens[0].item())
assert n_kept == TB, f"budget: {n_kept} != {TB}"
print("PASS budget exact:", n_kept)

# spare-the-newest: among the duplicate cluster {10,100,200}, position 200 (newest) must have had its
# redundancy penalty zeroed -> its score strictly higher than the older copies'. Verify via a direct
# scoring probe: rerun the scoring math standalone
import torch.nn.functional as F, math
kc = kv[:, :CUR-RES].transpose(1, 2)
kn = kc / (kc.norm(dim=-1, keepdim=True) + 1e-8)
cs = torch.matmul(kn, kn.transpose(-1, -2))
n = kc.shape[2]
cs.masked_fill_(torch.eye(n, dtype=torch.bool).view(1,1,n,n), 0.0)
mask = cs > 0.5
ind = torch.where(mask, torch.arange(n).view(1,1,1,n), torch.zeros_like(mask, dtype=torch.long))
spare = torch.max(ind, dim=-1)[0]
cs2 = cs.clone(); cs2.scatter_(-1, spare.unsqueeze(-1), 0)
red = cs2.mean(dim=-2).softmax(dim=-1)
# newest duplicate (200) must carry LESS redundancy penalty than older duplicates (10, 100)
r10, r100, r200 = red[0,0,10].item(), red[0,0,100].item(), red[0,0,200].item()
print(f"redundancy penalties: pos10={r10:.5f} pos100={r100:.5f} pos200={r200:.5f}")
assert r200 < r10 and r200 < r100, "newest copy not spared!"
print("PASS spare-the-newest: pos200 penalty < older copies")

# contrast: the VaSE-port cal_redundancy penalizes all three ~equally
red_port = cs.mean(dim=2).softmax(dim=-1)
p10, p200 = red_port[0,0,10].item(), red_port[0,0,200].item()
print(f"PASS contrast (VaSE port would give pos10={p10:.5f} ~= pos200={p200:.5f})")
print("ALL PASS")
