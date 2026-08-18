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


### Ryzen AI Max+ 395 on Linux Vulkan/RADV: the n-max curve, and why 12 collapses

Same 64GB Strix Halo silicon as the two 395 rows in the table, on a third stack: Linux with
Vulkan/RADV (Mesa 26.3.0-devel) rather than Windows Vulkan or Linux ROCm/HIP. unsloth UD-Q4_K_XL,
131K context, q4_0 KV, `--parallel 1`, thinking off, headless. Build is the gfx1151 fork
[Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp) branch `strix-halo-vulkan` at
`baf6360be`, not upstream. Method: unchanged `probe.py`, three full passes per arm, median of pass
medians, warmup discarded. Acceptance excludes the per-pass warmup request.

| n-max | overall | code | prose | bash | acceptance | mean accepted draft len |
|---|---|---|---|---|---|---|
| off (baseline) | 11.9 | 11.8 | 11.9 | 11.8 | n/a | n/a |
| 2 | 24.6 | 26.9 | 19.8 | 24.6 | 0.797 | 2.6 |
| 3 | 27.7 | 32.1 | 19.6 | 27.7 | 0.732 | 3.2 |
| 4 | **28.7** | 34.9 | 18.9 | 28.7 | 0.642 | 3.6 |
| 12 | 13.9 | 26.3 | 8.3 | 13.9 | 0.274 | 3.9 |

Three things worth taking away.

**The two prompt classes move in opposite directions as draft depth rises.** From n-max 2 to 4,
code gains 30% (26.9 to 34.9) while prose *loses* 5% (19.8 to 18.9). The overall figure is a
weighted average of a lever that pays on predictable continuations and taxes unpredictable ones,
so the headline number is sensitive to prompt mix. This is the same code-up/prose-down shape the
A6000, 3x3090 and RTX PRO 6000 sweeps report, which suggests it is a property of the MTP head
rather than of any one backend.

**n-max 12 collapses, and the mechanism is measurable rather than inferred.** Going from n-max 4
to 12 triples the tokens drafted per step, but accepted draft length barely responds: median 3.6
to 3.9, mean 3.5 to 4.6. Acceptance meanwhile halves, 0.642 to 0.274. The head runs dry after
roughly four tokens no matter what n-max allows, so the extra drafts are verified for almost
nothing while costing full verify bandwidth. Prose falls to 8.3, i.e. **below the 11.9 spec-off
baseline**: on unpredictable content, deep drafting is a net loss rather than a smaller win.

That is not a contradiction of the GMK EVO-X2 row's n-max 12. That row's footnote reports
0.95-1.0 acceptance on its bench prompt but records novel-traffic acceptance of 0.345, which is
close to the 0.274 measured here on probe.py's novel prompts. The honest reading is that n-max 12
wins on highly predictable text and loses on novel text, so the optimum is a property of the
workload rather than of ROCm versus Vulkan. Anyone copying an n-max from another row should
re-sweep against their own traffic, per rule 1.

**A launch-to-launch variance floor, measured by accident.** A planned arm for a fork lever turned
out to be a no-op (the flag was already default-on and only `=0` opts out, so setting `=1` changed
nothing), which means it ran the identical configuration to the n-max 2 arm a second time, in a
separate server launch:

| launch | pass 1 | pass 2 | pass 3 | median |
|---|---|---|---|---|
| A | 24.6 | 24.6 | 25.5 | 24.6 |
| B | 23.8 | 24.1 | 23.9 | 23.9 |

2.8% apart, with every pass of B below every pass of A, while within each launch the three passes
agree to about 1%. So on this box three passes inside one server launch understate the real error
bar, and a clean-looking 3% difference between two separately launched arms is not evidence of
anything. That is why the n-max peak above is quoted as 3 to 4 rather than a single winner, and it
is worth keeping in mind when reading any sweep in this repo, including this one.
