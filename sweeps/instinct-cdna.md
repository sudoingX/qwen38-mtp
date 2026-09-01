# Instinct (CDNA) — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP on AMD datacenter parts. Rows and footnotes
go in the main [community table](../README.md#community-numbers).

### AMD Instinct MI210 64GB (CDNA2/gfx90a, ROCm 6.2): n-max sweep, prompt dependence, and a second drafting path
*by [@pestopoppa](https://github.com/pestopoppa)*

One MI210 64 GB HBM2e (`gfx90a`, CDNA2), ROCm 6.2.0-66, headless server with no compositor or
browser — rule 7's desktop tax does not apply. Model is unsloth `Qwen3.8-27B-Q8_0.gguf`
(29,047,086,048 B), 32768 context, **f16 K/V cache** (64 GB has the headroom, so there is no
quantized-KV confound anywhere in this study), `-ngl 99`, `-fa on`, `-b/-ub 2048`, `-t 8 -tb 8`,
`--parallel 1` on every arm. VRAM 28.65 GiB baseline, 30.90 GiB at MTP n-max 8, 33.92 GiB on the
DFlash arm (resident draft model). **The build is not stock upstream**: a private llama.cpp fork
("frozen-v9 + champion" lineage) at champion tip `9e18beb0036860f87cde32a77350f12fda8c1793`. Every
arm below is the same binary and the same serving config, so the deltas are clean, but the absolute
numbers will not reproduce on master (rule 6).

Method: unchanged `probe.py` at repo HEAD `431bf8a821`, one warmup discarded, then three runs x
three prompts at 400 max tokens, thinking off. Figures are `probe.py`'s own overall median of the
nine measured requests; per-prompt columns are that prompt's median of three. Acceptance is the
aggregate from the server's `draft acceptance` lines at end of run.

| n-max | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Overall median | Mean | Acceptance | Mean draft len |
|---|---|---|---|---|---|---|---|
| off | 30.4 | 30.5 | 30.3 | 30.4 | 30.4 | — | — |
| 2 | 42.0 | 32.0 | 40.3 | 40.3 | 37.5 | 0.880 | 2.76 |
| **8** | **74.6** | 30.5 | 46.8 | **46.8** | **52.0** | 0.375 | 3.99 |

**n-max 8 is the optimum on this card (+54.0% over baseline), and it beats n-max 2 by 16%.** That
is worth stating plainly because it cuts against the rest of the table: n-max 8 is recorded as a
loser nearly everywhere it has been swept here — "confirmed worst spec setting" on the NVFP4 5090,
"turns down" on the b10680 5090, 30.0–36.7 against 42.9 at n-max 4 on the 2x9070 — and the A5000
pair measured n-max 8 faster but declined to use it because the spread was unusable. Rule 1 says
24 GB cards peak at 2 and bigger/faster cards at 3–4; a 64 GB HBM2e datacenter part peaking at 8
extends that tiering by a step rather than contradicting it.

**Acceptance is again a vanity metric (rule 2), and this is a clean example.** n-max 2 accepts
0.880 of its drafts and n-max 8 only 0.375 — but the mean accepted draft *length* rises from 2.76
to 3.99, and throughput rises with it. Deep drafting on this card wins by getting more tokens per
verify, not by being more often right.

**Prompt dependence is extreme, and it grows with depth.** At n-max 2 the three prompts span
32.0–42.0 tok/s (1.3x). At n-max 8 they span 30.5–74.6 (2.4x): the python prompt more than doubles
(+145% over baseline) while **the prose prompt gets nothing at all — 30.5 vs a 30.5 baseline**. If
your workload is prose rather than code, the flag's value on this card rounds to zero at any depth
tested, and a single-prompt headline would have been badly misleading in either direction.

**Run-to-run stability, stated because n-max 8 is the row.** Per-run figures at n-max 8 were P1
`[74.6, 79.5, 72.0]`, P2 `[31.1, 30.5, 29.7]`, P3 `[58.2, 46.8, 45.9]`. P1 and P2 are tight; P3
carries one 58.2 outlier against a 45.9/46.8 pair, a ~25% excursion. The median is robust to it and
the baseline arm was extremely tight across all nine runs (30.0–30.6), but readers weighing this
against the A5000's decision to decline an n-max 8 row on spread grounds deserve the raw numbers.

#### A second drafting path on the same hardware

The same fork ships a block drafter ("DFlash") that is *not* the built-in MTP head and is not the
flag this repo is about. Measured on the identical config and instrument, with a separate draft
model (`-md`, 2,056,414,752 B) and `--spec-type draft-dflash --spec-draft-n-max 8`:

| path | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Overall median | Mean | Acceptance | Mean draft len |
|---|---|---|---|---|---|---|---|
| MTP n-max 8 | 74.6 | 30.5 | 46.8 | 46.8 | 52.0 | 0.375 | 3.99 |
| DFlash n-max 8 | 93.4 | 39.6 | 61.1 | **61.1** | 64.6 | 0.537 | 4.75 |

DFlash reaches 2.0x baseline against MTP's 1.54x, and — unlike MTP — it moves the prose prompt too
(39.6 vs 30.5 baseline, +30%). It is reported here only as context for what the MTP number is being
compared against; it needs a non-stock build and a separate draft model, so it is not something a
reader of this table can turn on.

**Caveat carried deliberately**: on this platform a greedy-vs-baseline divergence exists in the
*shared speculative verify path* — it affects all speculation modes including plain n-gram, it is
not attributable to the DFlash drafter, and it is under investigation. The throughput figures above
are therefore honest speed measurements, not identical-output speed claims. The same caution the
7900 XTX section states applies here.

**Instrument note.** These are client-side streaming numbers from `probe.py`. The same DFlash arm
measured server-side on our own harness reads roughly 13% higher, which is the expected direction
for an end-to-end client instrument that counts per-token delivery. Cross-instrument comparisons of
speculative decoding should not be made without stating which instrument produced them.
