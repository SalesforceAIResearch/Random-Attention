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
"""Apply the rp-selector dispatch to triattention's vllm hook_impl.py.

Replaces the malformed hook_impl.patch (headerless @@ hunks -> patch(1) rejects it).
Structural edit instead of a diff so it survives upstream line drift; loud failure
so run_all.sh's SETUP FAILED gate fires rather than benching an unpatched plugin.
Idempotent: safe to re-run on an already-edited file.
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()

if "build_random_pp_selector" in src:
    print(f"[apply_hook_edit] {path}: already applied")
    sys.exit(0)

ANCHOR = ("from .selector_hf import build_triattention_selector"
          " as _build_triattention_selector_impl")
if ANCHOR not in src:
    sys.exit(f"[apply_hook_edit] FATAL: import anchor not found in {path}")

repl = ANCHOR + "\nfrom .selector_random import build_random_pp_selector"
if not re.search(r"^import os\b", src, re.M):
    repl = "import os\n" + repl
src = src.replace(ANCHOR, repl, 1)

lines = src.splitlines(keepends=True)
call = next((i for i, l in enumerate(lines)
             if "= _build_triattention_selector(config, base_runner=base_runner)" in l), None)
if call is None:
    sys.exit(f"[apply_hook_edit] FATAL: selector call site not found in {path}")
top = next((i for i in range(call, -1, -1) if lines[i].strip() == "try:"), None)
if top is None:
    sys.exit("[apply_hook_edit] FATAL: no enclosing try: above the call site")
ind = lines[top][: len(lines[top]) - len(lines[top].lstrip())]
exc = next((i for i in range(call + 1, len(lines))
            if lines[i].strip() == "except TypeError:"
            and lines[i].startswith(ind + "except")), None)
if exc is None:
    sys.exit("[apply_hook_edit] FATAL: no except TypeError: at try-level indent")
end = next((i for i in range(exc + 1, len(lines))
            if "= _build_triattention_selector(config)" in lines[i]), None)
if end is None:
    sys.exit("[apply_hook_edit] FATAL: fallback call not found inside except block")

new = [
    ind + 'selector_kind = os.environ.get("TRIATTN_RUNTIME_SELECTOR",'
          ' "triattention").strip().lower()\n',
    ind + 'if selector_kind in {"random", "random_pp"}:\n',
    ind + "    (\n",
    ind + "        select_keep_indices,\n",
    ind + "        select_keep_indices_for_group,\n",
    ind + "        selector_status,\n",
    ind + "    ) = build_random_pp_selector(config, base_runner=base_runner)\n",
    ind + "else:\n",
]
new += ["    " + l if l.strip() else l for l in lines[top:end + 1]]
out = "".join(lines[:top] + new + lines[end + 1:])

compile(out, path, "exec")
open(path, "w").write(out)
print(f"[apply_hook_edit] {path}: applied (dispatch inserted at line {top + 1})")
