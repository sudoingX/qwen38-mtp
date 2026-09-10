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

#### Concurrency: how the second drafting path scales past one slot
*added 2026-09-09 by [@pestopoppa](https://github.com/pestopoppa)*

Everything above is `--parallel 1`, which is what the community table measures. This section
answers a different question that the table has no column for: **what happens to speculative
decoding on this card when several users share the slot pool.** It is context, not a table row —
different instrument, different workload, different drafter, different kernel tip from the section
above. The method notes below are load-bearing; please read them before quoting any of it.

Same MI210 64 GB (gfx90a, ROCm 6.2), same unsloth `Qwen3.8-27B-Q8_0.gguf`, DFlash drafter
(`--spec-type draft-dflash --spec-draft-n-max 8`, same resident draft model as above,
2,056,414,752 B), f16 K/V, `-ngl 99`, `-fa on`, `-b/-ub 2048`, `-c 16384`, `-t 8` pinned to 8
host cores, greedy sampling (`temp 0`, `top-k 1`), `n_predict 384`, `cache_prompt: false`.
**Only `-np` is varied**; every other flag is byte-identical across the four points. Three
separate server launches per point (not three requests against one server), one np-wide warmup
round discarded per launch, median of the three launches reported. GPU residency verified by
sampling during the request phase; sclk pinned at 1700 MHz.

| slots (`-np`) | aggregate tok/s | per slot | p95 dev over 3 launches | per-launch | peak VRAM |
|---:|---:|---:|---:|---|---:|
| 1 | **79.2** | 79.2 | 0.44% | 79.2, 79.2, 79.6 | 33.1 GiB |
| 2 | 109.4 | 54.7 | 1.60% | 109.4, 108.9, 111.2 | 34.5 GiB |
| 4 | 167.8 | 41.9 | 3.33% | 167.8, 162.2, 169.7 | 37.7 GiB |
| 8 | **179.1** | 22.4 | 1.82% | 182.4, 178.2, 179.1 | 43.1 GiB |

**The curve turns over hard between 4 and 8 slots.** Going from 4 to 8 buys +6.8% aggregate
throughput and costs each user nearly half their rate (41.9 → 22.4 tok/s). `-np 4` is the
operating point on this card: 93.7% of peak aggregate while every user still sees ~42 tok/s. If
you are sizing a shared server around speculative decoding, the useful number is not peak
aggregate — it is the last slot count before the per-user rate collapses, and here that is 4.

**Read this before you compare a number here to a number anywhere else.**

1. **Each `-np` point runs a DIFFERENT SET OF PROMPTS.** The harness fires one request per slot
   from a fixed list, so `-np 1` measures prompt #1 alone and `-np 8` measures all eight. Given
   what the section above establishes about prompt dependence on this card — a 2.4x span at
   n-max 8 — **the concurrency scaling and the prompt mix are confounded, and this sweep cannot
   separate them.** The turnover between 4 and 8 is large and monotone across every launch, so we
   do not think it is a prompt artifact, but the honest statement is that it was not controlled
   for. A prompt-matched sweep is the experiment that would settle it, and it has not been run.
   For the same reason the `-np 1` figure here is **not** comparable to the per-prompt columns
   above: different prompts (reasoning/math, not the py/prose/bash trio), different instrument.
2. **"Aggregate" here is the sum of the concurrent slots' own decode rates**, taken from each
   response's `timings.predicted_per_second` — not a wall-clock tokens/second for the batch. That
   choice deliberately excludes scheduling-tail jitter (which ran 5–10% on wall-clock even at
   greedy), so it flatters the aggregate relative to what a client would time end-to-end. The
   per-slot column is a decode rate, not a delivered rate.
3. **Different instrument from the rest of this page.** Server-side timings, not client-side
   `probe.py` streaming. The instrument note at the end of the previous section measured that gap
   at roughly 13% in this direction on the same arm.
4. **Different kernel tip.** The section above is champion `9e18beb0`; this sweep is
   `ef81196d5bdd4190b46dff4ae7eecc333a46c8ce`, a later tip with more folded in. Unlike the earlier
   tip this one is public and rebuildable —
   [`pestopoppa/llama.cpp`](https://github.com/pestopoppa/llama.cpp), branch
   `ak/champion/llama-cpp-0db32c06e3e5` — so rule 6 still applies (not stock upstream), but the
   build is no longer a black box.
5. **Different drafter.** This is the DFlash block drafter with its own draft model, not the MTP
   head this repo is about. **No MTP concurrency sweep exists on this hardware**, so nothing here
   should be read as a statement about what `-np` does to MTP self-drafting.

**On the spread.** p95 deviation across launches is 0.44% at `-np 1` and 3.33% at `-np 4` — but
it falls back to 1.82% at `-np 8`, so it is **not** monotone in slot count and we will not claim a
trend from four points at n=3. What the column does establish is that `-np 4` is the least stable
point measured, roughly 7x the `-np 1` spread: a single reading there is much weaker evidence than
a single reading at `-np 1`, which is why the per-launch figures are printed. Anyone A/B-ing a
change at concurrency should state their launch count. (At `-np 8` the clock left its pin briefly,
1695–1700 MHz, on one launch.)

**The greedy-divergence caveat from the previous section applies unchanged**: a greedy-vs-baseline
divergence exists in the shared speculative verify path on this platform, affects all speculation
modes, and is under investigation. These are honest speed measurements, not identical-output
speed claims.
