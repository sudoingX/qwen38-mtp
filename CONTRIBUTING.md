# Adding your row

The table's contract: a **paired A/B on the same machine, same GGUF, same serving config, only the spec flags changing**. Spec-off is the baseline, spec-on is the flag arm. Rows that compare two caches, two builds, or arrive without a baseline are welcome as sections, not main-table rows.

A mergeable PR has:

1. **One table row:** card, baseline tok/s, with-flag tok/s, n-max used, acceptance, contributor link.
2. **One footnote** (`\* ` prefix, matching the block): quant + repo, context size, KV cache type, llama.cpp build, OS/backend, VRAM before and after, and your method (`probe.py` unchanged at a stated commit is the default; if your method differs, say exactly how).
3. **Both arms at `--parallel 1`** (rule 5: a parallel-2 baseline reads ~20% low and inflates your gain).
4. Medians of at least 3 runs. Single-run screens belong in prose, labeled as screens (rule from the RX 9070 section: n=1 crowned the wrong arm).
5. Sweeps, findings, and analysis go in a `###` section under the table. Honest caveats raise the odds of a merge, they never lower them.

README.md only. No scripts, no binaries. Edits to other contributors' rows don't merge.
