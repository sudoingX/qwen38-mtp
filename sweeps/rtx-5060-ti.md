# RTX 5060 Ti 16GB — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP on the RTX 5060 Ti 16GB (Blackwell SM 120).

### RTX 5060 Ti 16GB: the first single-card row — a Q4 file fits, and draft depth keeps paying
*by [@jaisusx](https://github.com/jaisusx), PR #52*

The dual-5060-Ti rows in the main table established that unsloth's 16.68 GiB UD-Q4_K_XL does not
fit one 16 GB card. This is the first single-RTX-5060-Ti row: it runs **quimmedes/Qwen3.8-27B-XYZ
Q4-XYZ-v2** (15.06 GB) — the largest Q4-tier file that fits a single card — at **32K context** with
a **q4_0 KV cache** (a Q4-tier file cannot hold the repo-standard 131K on one 16 GB card).

Host: Ryzen 5 3600, 31 GB RAM, Ubuntu 24.04, driver 595.84. Model drive SATA SSD. No desktop
compositor running during the bench. Main row: llama.cpp **b10472** (commit `60eeeb6`, 2026-08-17)
built from source with CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=120`. The n-max sweep below ran on
the previous build (`6ee0f65`, 2026-06-22); the build A/B at the bottom reconciles them.

| n-max | p-min | build | tok/s (median) | tok/s (mean) | acceptance | VRAM (MiB) |
|---|---:|---|---:|---:|---:|---:|
| off | — | b10472 | 26.3 | 26.3 | — | 14,514 |
| off | — | 6ee0f65 | 25.9 | 25.9 | — | 14,830 |
| 2 | 0.00 | 6ee0f65 | 50.0 | 49.6 | 0.53–0.86 | 15,404 |
| 3 | 0.00 | 6ee0f65 | 53.6 | 53.9 | 0.47–0.77 | 15,554 |
| 4 | 0.00 | 6ee0f65 | 59.3 | 57.0 | 0.36–0.69 | 15,704 |
| 4 | 0.00 | b10472 | **59.5** | 59.6 | 0.34–0.69 | 15,704 |
| 6 | 0.00 | 6ee0f65 | OOM — did not load | — | — | — |
| 4 | 0.65 | 6ee0f65 | 52.5 | 52.8 | ~0.77 | 15,704 |

**Findings:**

1. **Draft depth keeps paying where the 24 GB cards plateau.** The table's 24 GB rows peak at
   n-max 2; this card gains at every step (2→3→4: 50.0 → 53.6 → 59.3, +18.6% from n2 to n4).
   The 48/64 Gated DeltaNet layers make the verify pass cheap, so deeper drafts stay profitable
   even as acceptance per position falls (first-position acceptance stays ~0.85 at n-max 4).
2. **p-min gating inverts here** (rule 2's Blackwell inversion). The 0.65 confidence gate at
   n-max 4 raised acceptance (0.36–0.69 → ~0.77) but **cost 11.5% throughput** (59.3 → 52.5).
   The 16 GB card behaves like the fast desktop Blackwells, not the bandwidth-starved APUs —
   despite only 448 GB/s. Sweep it, don't adopt it.
3. **n-max 6 does not fit.** The MTP context creation OOM'd at load (`cudaMalloc failed: out of
   memory`, 130 MiB request with 136 MiB free): n-max 4 sits at 15,704/15,840 MiB (99.1%), and
   the deeper draft's KV + compute buffers exceed the card. n-max 4 is the hard ceiling here.
4. **The baseline is the real story for 16 GB owners.** 26.3 tok/s spec-off on a Q4-tier 27B
   (vs 7.7–9.2 tok/s measured for the previous Qwen3.6-27B Q4_K_M setup, which spilled ~3 GB of
   weights to host RAM on the same card). A file that fits VRAM is worth more than the MTP flag.

**Build A/B (rule 6 check, same method, only the binary changed):** b10472 vs `6ee0f65` on the
same file, same 32K/q4_0 KV config: baseline 25.9 → 26.3 (+1.5%), n-max 4 59.3 → 59.5 (+0.3%).
Rule 6's "+10-15% on every quant" did not reproduce on this card/model pair — the June-22 build
was already on the current qwen35 kernel path. Deltas within each build are the durable numbers.

**Method:** unchanged `probe.py` at commit `70a699e`, three runs x three prompts (python merge,
mmap-vs-read, bash watcher), 400 tokens, thinking off, warmup discarded. Both arms `--parallel 1`;
only the spec flags differ (same ngl). VRAM via `nvidia-smi` during serve. Acceptance from
llama-server `draft acceptance` log lines (range across the nine tasks).

**Honest caveats:**
- **32K context**, not the repo-standard 131K — the fit constraint above.
- Acceptances and per-prompt medians swing between passes (the bash prompt runs fastest, the
  prose prompt lowest acceptance); every number is the median of three complete passes.
- The full depth sweep (n2/n3/n6/p-min) ran on the June-22 build; only baseline + n-max 4 were
  re-run on b10472. Per the build A/B above, the pattern transfers.

### Unsloth IQ4-family files on one 16 GB card: none of them serve MTP at speed
*by [@jaisusx](https://github.com/jaisusx), PR #52*

The dual-card rows run unsloth files that cannot fit a single 5060 Ti. This section records what
actually happens when you try to force unsloth's Q4-family onto one card. Per CONTRIBUTING this
belongs here, not in the main table: the IQ4_XS flag arm changes two variables (spec flags and
`-ngl`), so it is not a clean table-row A/B.

| file | size | attempt | result |
|---|---|---|---|
| IQ4_NL | 16.34 GB (15.22 GiB) | `-ngl 999` | OOM at load (16.3 GiB need vs 15.47 GiB usable) |
| IQ4_NL | same | `--fit on --fit-target 64`, card free | loads; **16.8 t/s** baseline (June-22 build) — usable but 35% under the fitting Q4 file on the same build |
| IQ4_XS | 15.71 GB (14.63 GiB) | `-ngl 999` baseline | fits; **25.2 t/s** (b10472), VRAM 15,188 MiB |
| IQ4_XS | same | `-ngl 999` + n-max 4 | OOM at load (240 MiB short) |
| IQ4_XS | same | `-ngl 60` (4 layers CPU) + n-max 4 | loads; **20.3 t/s median** — *net loss* vs its own 25.2 baseline (acceptance 0.33–0.71) |

Read together with the main section: on a single 16 GB card, the only Q4-tier file that runs MTP
at full GPU speed is one small enough to fit with headroom — quimmedes Q4-XYZ-v2 (14.03 GiB) at
n-max 4 = +126%. Unsloth's Q4-family files either do not fit (IQ4_NL, Q4_K_M 17.11 GB, UD-Q4_K_XL
17.92 GB) or fit but have no room for the MTP context (IQ4_XS: OOM, or spill that eats the gain).

Caveats: the IQ4_NL `--fit` baseline ran on the June-22 build (same build as the quimmedes
sweep); the earlier `--fit` + n-max 4 IQ4_NL OOM was measured with the production Qwen3.6 server
resident on the card (confounded, not re-tested with a free card — the IQ4_XS OOM above is the
clean datapoint).
