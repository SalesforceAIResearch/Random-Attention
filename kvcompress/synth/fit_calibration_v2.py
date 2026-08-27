#!/usr/bin/env python
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
"""Pin the R2-P5 calibration fit bit-reproducibly (review hygiene item).
Exact procedure that produced the frozen 10.735/-8.638: Nelder-Mead on n-weighted binomial
NLL, x0=[11.6,-9.0], default scipy tolerances, the 7 pinned (R, acc) points."""
import numpy as np
from scipy.optimize import minimize
PTS = [(0.838, 0.597), (0.780, 0.386), (0.641, 0.161), (0.754, 0.331),
       (0.702, 0.250), (0.840, 0.620), (0.751, 0.396)]
X = np.array([p[0] for p in PTS]); Y = np.array([p[1] for p in PTS]); N = 500
def nll(th):
    a, b = th
    p = np.clip(1 / (1 + np.exp(-(a * X + b))), 1e-6, 1 - 1e-6)
    return -N * (Y * np.log(p) + (1 - Y) * np.log(1 - p)).sum()
r = minimize(nll, [11.6, -9.0], method="Nelder-Mead")
print(f"a={r.x[0]:.3f} b={r.x[1]:.3f}  (frozen: 10.735 -8.638)")
