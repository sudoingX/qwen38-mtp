# APUs and iGPUs — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP, moved verbatim from the main README. Each section carries its author and original PR. Add yours via a PR to this file, row and footnote go in the main [community table](../README.md#community-numbers).

### Ryzen AI Max+ 395 / Radeon 8060S: n-max 2 versus 4
*by [@shiwuxiu](https://github.com/shiwuxiu), PR #9*

Same 64GB Strix Halo system, same unsloth UD-Q4_K_XL model and serving config as the row above: Windows Vulkan, 32K context, q8_0 KV, Flash Attention, batch 2048/512, and `--parallel 1`. Upstream `probe.py` was unchanged; all values are medians of three runs per prompt with thinking off. Acceptance is from the server log, excluding warmup.

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| **2** | **23.7** | 25.2 | **19.1** | **23.7** | 0.52-0.94 (78.2% aggregate) |
| 4 | 23.5 | **29.9** | 16.5 | 23.5 | 0.35-0.91 (65.8% aggregate) |

MTP n-max 2 doubles the overall median versus the 11.5 tok/s spec-off control (+106%). N-max 4 is effectively flat overall (-0.8% versus n-max 2), but the workload split is sharp: Python rises 18.7% while prose falls 13.6%. This system keeps n-max 2 for mixed use and treats n-max 4 as a code-specialized option.


### Radeon 890M iGPU (Strix Point): n-max sweep — biggest gain here, and an unstable peak
*by [@davidglogan](https://github.com/davidglogan), PR #10*

Same 890M, same config as the row above, `--spec-draft-n-max` swept with a no-flag baseline.
**Two independent runs, both with unchanged `probe.py`, identical flags, nothing else on the GPU.**
Overall probe medians (tok/s):

| n-max | Run A (11:10) | Run B (14:04/15:14) | Acceptance (Run B) |
|---|---|---|---|
| baseline (no flag) | 2.6 | 2.7 | - |
| 2 | 5.9 | 5.7 | 0.59-0.91 |
| 3 | 7.5 | 7.0 | 0.47-0.73 |
| 4 | 7.9 | **7.4** | 0.35-0.69 |
| 5 | **8.5** | 6.7 | 0.27-0.61 |
| 6 | 7.2 | not run | - |

🛑 **The two runs disagree on where the peak is — Run A says n-max 5, Run B says n-max 4 — so I am
not claiming a peak for this card.** Baseline and n-max 2 reproduce to within 3% (2.6/2.7 and
5.9/5.7), and the divergence widens with n, reaching **27% at n-max 5**. That is worth knowing on
its own: on a bandwidth-shared iGPU whose clocks and memory contention move between runs, three
runs x three prompts is not enough to resolve a peak, even though it is plenty for the n-max 2
headline. Anyone sweeping on an APU should repeat the sweep before trusting an optimum.

**The table row above uses n-max 2, where both runs agree.** At that setting this is **+111%**;
at whichever of 4/5 is the true optimum it is **+174% to +227%** — the largest gain posted here
either way, on the slowest hardware in the table.

Why an iGPU tops the gain column while sitting last on absolute throughput:
**MTP pays in proportion to how bandwidth-starved decode already is.** A dense 27B on a 128-bit
LPDDR5 bus is the most starved configuration in this table, so amortising the weight read across
accepted drafts buys the most. A large multiple of a small number is still a small number - this
does not make the 890M a good card for a dense 27B, it makes MTP the difference between unusable
and marginal on one.

Daily mixed use: 4 is the safe pick (top-2 in both runs). n-max 6 fell away in the run that tested it.


### Mac Studio (M3 Ultra, 96GB UMA, Metal): shallow and ungated wins
*by [@adityavsingh](https://x.com/adityavsingh)*

Mac Studio with Apple M3 Ultra and 96 GB unified memory, macOS/Metal, unsloth Q6_K (21 GB file), 131K context, Flash Attention, q4_0 K/V main and draft KV caches, and `--parallel 1`. LM Studio's bundled llama.cpp Metal backend 2.29.0 reports `version: 1 (dd1ea52)`. The upstream `probe.py` was unchanged: warmup discarded, three runs per prompt, thinking off, with Qwen3.8 as the only resident model. Overall is the probe's overall median; acceptance is taken from server logs excluding warmup.

| Config | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---:|---:|---:|---:|---|
| spec off | 22.8 | 23.0 | **22.9** | 22.6 | — |
| **MTP n-max 2, ungated** | **24.2** | 26.4 | 20.2 | **24.2** | 0.53-0.95 (79.8% aggregate) |
| MTP n-max 2, p-min 0.60 | 19.9 | **27.0** | 16.1 | 19.9 | 0.66-0.97 (86.9% aggregate) |
| MTP n-max 3, ungated | 20.7 | 23.6 | 14.3 | 20.7 | 0.41-0.94 (71.9% aggregate) |
| MTP n-max 4, ungated | 18.1 | 21.8 | 11.5 | 18.1 | 0.33-0.93 (65.1% aggregate) |

N-max 2 is the mixed-workload setting here: 22.8 → 24.2 tok/s (+6.1%). It improves both code prompts while the prose prompt regresses 11.8%, so the modest overall gain should not be read as a universal speedup. Deeper drafts lose rapidly, and the 0.60 gate is a useful counterexample to treating acceptance as the goal: it raises aggregate acceptance from 79.8% to 86.9% but lowers overall throughput from 24.2 to 19.9 tok/s. On this Q6/Metal configuration, keep the shallow, ungated setting for throughput; do not copy the deep or gated stacks from bandwidth-starved APUs.
