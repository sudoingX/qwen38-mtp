# RTX 4060 Ti 16GB — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP on the RTX 4060 Ti 16GB (Ada, SM 89, 288 GB/s).

### RTX 4060 Ti 16GB: the slowest single card in the table — n-max 3 pays, n-max 4 OOMs
*by [@CeIest2](https://github.com/CeIest2)*

First single-RTX-4060-Ti row, and the lowest-bandwidth single card in the table (288 GB/s —
the 5060 Ti row sits at 448 GB/s). Runs **quimmedes/Qwen3.8-27B-XYZ Q4-XYZ-v2** (15,064,569,440 B
= 14.03 GiB, sha256 ab58f29fa81dd604…) — the same file as the 5060 Ti row, making this a
controlled cross-card pair — at **32K context** with a **q4_0 KV cache**.

Host: Ubuntu 24.04.4, driver 580.173.02 (CUDA 13.0), 31 GB RAM. llama.cpp master `9a286ac`
(2026-08-21) built from source with CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=89`. Desktop session
during the bench but nearly idle (240-290 MiB GPU footprint, no compositor spill — rule 7).

| n-max | p-min | tok/s (median) | tok/s (mean) | acceptance | VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| off | — | 17.6 | 17.5 | — | 14,778 |
| 2 | 0.00 | 35.9 | 34.9 | 0.54–0.96 | 15,678 |
| 3 | 0.00 | **40.0** | 40.2 | 0.41–0.94 | 15,830 |
| 4 | 0.00 | OOM — did not load | — | — | — |
| 3 | 0.70 | 39.8 | 37.5 | 0.72–0.98 | 15,833 |

Per-prompt medians (code / prose / bash): baseline 17.6 / 17.6 / 17.5 — n2 40.6 / 29.4 / 35.9 —
n3 49.0 / 32.3 / 40.0 — n3+pmin 47.3 / 26.8 / 39.8.

**Findings:**

1. **Depth still pays at n-max 3, then VRAM runs out.** n2 → n3 gains +11% overall
   (35.9 → 40.0), carried by every prompt. n-max 4 OOMs at load, reproducibly: the same 130 MiB
   `cudaMalloc failed: out of memory` request the 5060 Ti hit at n-max 6, and reducing batch to
   `-b 512 -ub 512` does not save it. n-max 3 sits at 15,830/16,380 MiB (96.6%) with a ~280 MiB
   desktop — n-max 4 is the hard ceiling here, one step earlier than the 5060 Ti (which held
   n-max 4 at 99.1% headless).
2. **Controlled pair against the 5060 Ti row (same file, same 32K/q4_0 KV).** Baseline scales
   with bandwidth: 17.6 vs 26.3, a 0.67 ratio against a 0.64 bandwidth ratio — fully
   bandwidth-bound spec-off. With the flag: 40.0 vs 59.5 at each card's own ceiling (n3 vs n4).
   The +127% gain on this card is larger than the 5060 Ti's +126% at its peak — the slower the
   card, the more the flag is worth.
3. **p-min gating is a wash at the ceiling, not a cost.** The 0.70 gate at n-max 3 costs 0.5%
   (40.0 → 39.8, within noise) while lifting acceptance to 0.72–0.98 — against the 5060 Ti's
   −11.5% at n-max 4 and the v1 file's −3.5% on this same card (section below). Rule 2's
   inversion point really does sit near this bandwidth class; at 288 GB/s the gate is roughly
   break-even, and the higher acceptance makes it the safer daily driver.

**Method:** unchanged `probe.py` at commit `c7bc415`, three runs x three prompts (python merge,
mmap-vs-read, bash watcher), 400 tokens, thinking off, warmup discarded. Both arms `--parallel 1`;
only the spec flags differ (same ngl 999, `-fa 1`, 32K ctx, q4_0 K/V). VRAM via `nvidia-smi`
during serve. Acceptance from llama-server `draft acceptance` log lines (range across the nine
tasks, warmup excluded).

**Honest caveats:**
- **32K context**, not the repo-standard 131K — a Q4-tier file cannot hold 131K on one 16 GB card.
- Benched in a live desktop session, not headless. The 5060 Ti row ran compositor-free; the
  ~280 MiB desktop footprint here is part of why n-max 4 OOMs on this card. Baseline probe runs
  were flat to ±0.5% (17.5-17.7), so no spill is suspected during measurement.
- Single 3-run pass per arm; the n3 vs n3+pmin delta (0.5%) is inside run-to-run noise.

### Q4-XYZ v1 vs v2 on the same card: the quant version changes the n-max curve
*by [@CeIest2](https://github.com/CeIest2)*

The first pass on this card ran the older **Q4-XYZ v1** file (15,194,248,512 B = 14.15 GiB,
sha256 58f6975a5b5707ee…) — same host, same llama.cpp build `9a286ac`, same 32K/q4_0 KV config,
same method, one day earlier. Only the GGUF differs.

| n-max | p-min | v1 tok/s (median) | v2 tok/s (median) |
|---|---:|---:|---:|
| off | — | 17.8 | 17.6 |
| 2 | 0.00 | 34.8 | 35.9 |
| 3 | 0.00 | 34.9 | 40.0 |
| 4 | 0.00 | 34.4 | OOM |
| 4 / 3 | 0.70 | 33.2 (n4) | 39.8 (n3) |

**Findings:**

1. **Baselines are identical (17.8 vs 17.6, within noise)** — the ~130 MiB file-size difference
   buys nothing spec-off, as expected for a bandwidth-bound decode.
2. **The n-max curve shape changes with the quant version.** On v1 the overall median is flat
   from n-max 2 (34.8 / 34.9 / 34.4 — code gains, prose loses symmetrically); on v2, n-max 3
   clearly pays (+11% over n2) and n-max 4 OOMs. Same silicon, same build, same flags — the
   draft-quality difference between the two quant versions moves the depth optimum by a full
   step. Rule 1's "re-sweep after any config change" extends to the quant file itself.
3. **v1 fits n-max 4 (15,471 MiB, 94.5%) where v2 OOMs** — the smaller file needs *more* room
   for the MTP context, not less. Counter-intuitive, reproducible (two load attempts, one with
   reduced batch), worth knowing before assuming file size predicts headroom.

Caveat: the two passes ran a day apart in two desktop sessions (v1 at 246 MiB desktop footprint,
v2 at ~280 MiB). Within-file A/Bs are same-session; cross-file deltas share host and build but
not session, and part of the v2 n-max-4 OOM may be the slightly larger desktop footprint.
