# Radeon (RDNA) — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP, moved verbatim from the main README. Each section carries its author and original PR. Add yours via a PR to this file, row and footnote go in the main [community table](../README.md#community-numbers).

### 2x RX 9070 16GB: n-max sweep, and the second knob nobody is turning
*by [@tomertec](https://github.com/tomertec), PR #7*

Same pair of cards, same config as the RX 9070 row above, at the full 262K window. Decode medians (tok/s) at three prompt depths, draft acceptance from the server log:

| config | ~0 ctx | 16K | 65K | Acceptance |
|---|---|---|---|---|
| spec off | 22.1 | 21.0 | — | — |
| n-max 2, no gating | 41.6 | 37.2 | 28.0 | 0.73 |
| **n-max 4, `--spec-draft-p-min 0.60`** | **42.9** | **38.5** | 27.4 | 0.86 |
| n-max 4, `--spec-draft-p-min 0.75` | 38.3 | 36.6 | **28.9** | 0.91 |
| n-max 4, `--spec-draft-p-min 0.80` | 40.7 | 35.5 | — | 0.90 |
| n-max 8, any p-min | 30.0-36.7 | 31.8-33.9 | — | 0.74-0.84 |

**`--spec-draft-n-max` alone is only half the tuning.** `--spec-draft-p-min` is a confidence gate: the head stops drafting once its own probability falls below the threshold, so a deeper n-max costs nothing on the rounds where the draft was going to be rejected anyway. That is what makes n-max 4 usable here — ungated n-max 4 was worse than n-max 2, and gated at 0.60 it drafts ~3.3 tokens per round at 0.86 acceptance instead of exactly 2.0 at 0.73. On a copy-heavy prompt (rename-and-echo over a 4K-token file) the gated arm is **+21%** over the deployed n-max 2, well above the +3% it shows on mixed work. n-max 8 loses at every gate value.

Two things this rig says that the NVIDIA rows do not:

- **The delta is much larger and the ceiling is the same.** +88% at n-max 2, +94% at the tuned setting, against +33-39% on the 24GB NVIDIA cards — because the *baseline* is unusually bad here, not because the flag is better. Vulkan on RDNA4 at batch-1 decodes across two cards serially, so 22.1 tok/s unassisted; with the head engaged it lands at 41.6-42.9, the same absolute band as everyone else. If your card is bandwidth-poor at batch 1, this flag is worth more to you than to a 3090 owner.
- **N=1 crowned the wrong arm.** A single-run screen put p-min 0.75 on top; medians of 3 flipped the winner to 0.60. The noise band here is around +/-2 tok/s, which is wide enough to invert a ranking. Do reps before you deploy a config.

Method note, since it is not the repo's: numbers come from a local greedy streaming harness (max 1200 tokens) rather than `probe.py`, all in one session against one server instance with only the spec flags varying. The n-max 2 and p-min 0.60 arms are medians of 3; the spec-off, p-min 0.75/0.80 and n-max 8 arms are single screening runs. Treat the sweep as a shape, and the two headline arms as measurements.


### AMD Radeon AI PRO R9700 32GB: n-max 2 versus 4
*by [@ajnytebot](https://github.com/ajnytebot), PR #8*

Same R9700, same unsloth UD-Q4_K_XL model and serving config as the row above, at 262K context with q4_0 KV. Upstream `probe.py` unchanged: medians of three runs per prompt, thinking off. Draft acceptance is from the server log.

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 43.3 | 45.3 | **36.3** | 43.3 | 0.60-0.94 |
| **4** | **47.4** | **75.5** | 33.3 | **47.4** | 0.28-0.90 |

N-max 4 lifts the overall median another 9.5% over n-max 2, driven by a 66.7% jump on the Python prompt. Prose falls 8.3% as aggregate acceptance drops from 82.3% to 60.5%; prose-only acceptance is 29.4% at n-max 4. The speed shape supports n-max 4 as a code-specialized arm, while n-max 2 remains the mixed-workload default. Loaded VRAM: 22.53 GB spec-off, 24.55 GB at n-max 2, and 24.87 GB at n-max 4.


### AMD Radeon 9060 XT 16 GB
*by [@kdrapel](https://github.com/kdrapel), PR #17*
Single card. LLama compiled with ROCM 7.1 support (Vulkan slower with dense models). Model is AtomicChat IQ3_XXS (https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF). Upstream `probe.py` unchanged.

| n-max | Overall | Acceptance |
|---|---|---|
| 2 | 28.7 | 0.62-0.93 |
| 3 | 30.7 |  0.41-0.91 |
| 4 | 30.9 |  0.29-0.93 |

nb. made another sweeping test on MTP n-max with another benchmark and more complex task and n-max = 2 was the clear winner, performance was impacted with larger n-max, 2 is the sweet general spot


### RX 7900 GRE 16GB: packed-16GB live-traffic study, spec-off baseline added 2026-08-16
*by [@lsunay](https://x.com/lsunay1), PR #19*

Data: [@lsunay](https://x.com/lsunay1), Hermes agent on PC-12.

Same host, same model, same serve, only the spec flags varying. The card runs the custom AtomicChat IQ3_XXS quant at 90K context with turbo3/turboquant KV — VRAM sits at 15.35 GB of 16 GB (~96%), so this is the tightest 16 GB rig in the table, and the config was tuned for a live Hermes agent session, not a clean probe run.

Live serve, one slot (`-np 1`), `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`. Decode speeds and draft acceptance from the server log over eight consecutive real agent turns (context growing 29.2K → 37.0K tokens between turns):

| Task | Output | Decode | Acceptance | Mean draft len |
|---|---|---|---|---|
| 1436 | 700 tok | 51.8 tok/s | 0.930 | 3.46 |
| 1671 | 310 tok | 47.7 tok/s | 0.932 | 3.13 |
| 1795 | 109 tok | 44.8 tok/s | 0.903 | 3.32 |
| 1844 | 778 tok | 53.8 tok/s | 0.964 | 3.71 |
| 2090 | 482 tok | 50.2 tok/s | 0.940 | 3.46 |
| 2260 | 150 tok | 36.6 tok/s | 0.872 | 2.55 |
| 2347 | 751 tok | 52.6 tok/s | 0.962 | 3.70 |
| 2577 | 472 tok | 44.5 tok/s | 0.905 | 3.25 |

Average decode **47.8 tok/s** (best 53.8, worst 36.6 — the 150-token turn, where the spec overhead dominates exactly as the length rule predicts). Acceptance band **0.87–0.96, average ~0.93**; mean draft length 2.55–3.71, i.e. n-max 3 was almost fully consumed. Prompt processing held 239–314 tok/s (3.3–4.2 ms/token) across the 29K–37K context growth. Vulkan compute-graph reuse climbed 757 → 1420 over the session.

**Spec-off baseline, measured 2026-08-16** (the card was deployed spec-on from the start, so this was back-filled): same host, same model, same turbo3 KV, same 90K context — the only variable removed is MTP (no `--spec-type/--spec-draft-*` flags), run in a fresh `--rm` container on port 8095 under the same live Hermes agent session (context grew to ~45K). Server-log `print_timing` over the final turns:

| Task | Output | Decode | Prompt eval |
|---|---|---|---|
| 8 | 251 tok | 34.08 tok/s | 300 tok @ 246.8 tok/s |
| 10 | 150 tok | 30.94 tok/s | 20,265 tok @ 460.7 tok/s (cold) |
| 425 | 455 tok | 30.58 tok/s | 3,753 tok @ 370.8 tok/s |
| 885 | 84 tok | 30.94 tok/s | 356 tok @ 261.7 tok/s |
| 2059 | 116 tok | 29.42 tok/s | 132 tok @ 205.1 tok/s |
| 2177 | 1,290 tok | 28.96 tok/s | 1,852 tok @ 301.9 tok/s |
| 3470 | 93 tok | 29.40 tok/s | 946 tok @ 265.1 tok/s |
| 4379 | 610 tok | 28.71 tok/s | 829 tok @ 251.0 tok/s |
| 4993 | 186 tok | 28.80 tok/s | 303 tok @ 223.4 tok/s |
| 5181 | 305 tok | 28.70 tok/s | 328 tok @ 232.7 tok/s |
| 5488 | 351 tok | 28.51 tok/s | 272 tok @ 208.3 tok/s |
| 5841 | 200 tok | 28.53 tok/s | 292 tok @ 222.1 tok/s |
| 6043 | 152 tok | 33.13 tok/s | 2,565 tok @ 497.1 tok/s |
| 6199 | 232 tok | 28.49 tok/s | 320 tok @ 235.2 tok/s |

Decode settled at **28.5–28.8 tok/s (avg ~28.7)** by 35–45K context; the early 34 tok/s figures are small-context turns. VRAM: **13.77 GB of 16 GB** (15.35 GB with MTP — the draft context costs ~1.58 GB on top of the packed weights + KV).

**Honest read of the A/B:** the baseline band (28.5–28.8 at 35–45K) sits ~1.5K tokens *above* the MTP band's top (37K), where this card's spec-on decode is already at its floor (44.5 tok/s). MTP-on therefore gained at least **+54%** over the same card's unassisted decode at adjacent context — the true figure at equal context is probably a few points higher, but we do not claim more than the measured numbers show. Two structural notes: (1) without MTP the 90K slot fits with ~2.2 GB to spare, so the spec-on deployment was *not* forced by VRAM — MTP bought speed, not context; (2) the baseline's cold prefill (460–497 tok/s) is faster than the MTP serve's (239–314 tok/s) — the draft head's overhead lands on prompt processing, not just decode.

**p-min 0.60 vs 0.75, A/B on the same serve** (container restarted between arms, same 400-token probe plus live agent traffic):

| setting | Acceptance | Decode | Verdict |
|---|---|---|---|
| **p-min 0.75 (n-max 3)** | 0.87–0.96, avg ~0.93 | ~47–48 tok/s | **shipped** |
| p-min 0.60 (n-max 3) | 0.67–0.77 | ~50.5 tok/s idle, +7% | rejected |

At 0.60 the head drafts past its confidence gate and acceptance drops ~0.15–0.25; the speed gain is a flat ~7% idle and the long-turn acceptance collapses to 0.67 under real traffic. This card is the counterpoint to the RX 9070 pair's finding that 0.60 "makes deeper n-max nearly free": with 16 GB at 96% VRAM and n-max 3, the gate at 0.75 already keeps rejection cheap, and loosening it just buys noise. The 0.60 number is also a single-run screen (the 9070 section's own warning applies: medians of 3 before deploying), but the acceptance delta is too large to ignore.

What this section adds:

- **Spec-off baseline back-filled 2026-08-16:** 28.5–28.8 tok/s unassisted at 35–45K context versus 44.5–53.8 with MTP — at least **+54%** at adjacent context on the same card (context bands differ by ~1.5K tokens; equal-context figure would be a few points higher, not claimed).
- **The deployment was speed-driven, not VRAM-forced:** without MTP the same 90K slot fits with ~2.2 GB to spare (13.77 GB of 16 GB). MTP cost ~1.58 GB of draft context and bought the speed.
- **n-max 3 holds as a daily driver at 0.93 average acceptance on RDNA3 16 GB** — the 5090 sweep's "run 2 as your daily" did not apply here; at 0.75 the third draft slot was accepted almost all the time, so there was nothing to give back.
- **VRAM headroom is a hidden variable.** Every row in the main table runs with 40%+ of the card's VRAM free; this one runs at 96%. The acceptance and speed bands above are what MTP does on a fully packed 16 GB card, not a comfortable 24 GB one.
- **The p-min knob is workload- and VRAM-dependent**, not just card-size-dependent: 0.60 won on the 2×9070 pool, 0.75 wins here.


### RX 7900 XTX 24GB (Linux/ROCm 10/HIP): n-max sweep
*by [@vijay-14](https://github.com/vijay-14)*

Single RX 7900 XTX 24GB (`gfx1100`) on Ubuntu 24.04.4, kernel 6.8.0-138, AMDGPU DKMS 7.1.3 and ROCm 10.0. The server is repo-local HIP llama.cpp `62acc89`, built against the ROCm 10 libraries; host is Ryzen 7 7700X with 32 GB RAM. Model is unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf` (16,464,440,224 B, sha256 `322e194f…23482`) at 131K context with q4_0 K/V, flash attention, all 66 layers on ROCm0, and `--parallel 1`. No `--spec-draft-p-min` flag was set.

Method: unchanged upstream `probe.py` at `668cb10`, one discarded warmup, then three runs of each of its Python, prose, and Bash prompts (nine measured requests per arm, 400-token ceiling). Baseline has no speculative flags; N=2/3/4 add only `--spec-type draft-mtp --spec-draft-n-max N`. Acceptance is parsed from the server's nine measured `draft acceptance` lines; range is per request and parenthesis is aggregate.

| n-max | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Overall median | Acceptance |
|---|---|---|---|---|---|
| off | 36.3 | 36.2 | 36.3 | 36.3 | — |
| **2** | 66.6 | **50.1** | **62.6** | **62.6** | 0.56-0.94 (0.80) |
| 3 | 74.7 | 47.5 | 56.7 | 56.7 | 0.44-0.94 (0.65) |
| 4 | **75.3** | 41.7 | 60.2 | 60.2 | 0.33-0.90 (0.63) |

**N=2 is the mixed-workload optimum: 36.3 → 62.6 tok/s (+72.5%).** Deeper drafts keep paying for the Python prompt (66.6 → 74.7 → 75.3), but prose falls at every extra depth (50.1 → 47.5 → 41.7) and Bash peaks at N=2. That makes N=2 the right table row even though N=4 is faster on code alone.

All weights remained resident: baseline/N=2/N=3/N=4 used 18.06/19.41/19.56/19.71 GiB VRAM after load, while GTT stayed 45 MiB throughout. Post-probe N=2 temperatures were 56 C edge, 75 C junction, and 78 C memory; N=4 reached 81 C memory. These are Linux ROCm/HIP numbers, so they are a separate stack from the existing XTX ROCm launch row, Linux RADV/Vulkan row, and Windows/Vulkan row; compare the paired delta only within this configuration.

Quality caveat: a separate local fixed-seed, short-prompt check returned deterministic text within each arm but slightly different baseline/MTP wording. This section reports the upstream probe's throughput measurement, not an identical-output speed claim.


### RX 7900 XTX 24GB (Windows/Vulkan): first Windows XTX A/B
*by [@pparuzel](https://github.com/pparuzel), PR #53*

Single RX 7900 XTX 24GB on Windows 11, official `ggml.llamacpp` WinGet build 9553 (`9e3b928fd`, Clang 19.1.5) with the Vulkan backend — no ROCm installed; the AMD driver runs the iGPU too, but the serve is Vulkan0 (XTX) only. unsloth UD-Q4_K_XL, 131K context, q4_0 KV cache (K and V), flash attention on, `--parallel 1` all arms, thinking off, host Ryzen 7 9800X3D. Spec arms: `--spec-type draft-mtp` ungated (p-min 0.00), n-max 2, 3, and 4. Method: unchanged `probe.py` at `c7bc415`, three runs x three prompts, warmup discarded.

| arm | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Overall median | Acceptance |
|---|---|---|---|---|---|
| spec-off | 40.9 | 41.1 | 40.8 | 41.0 | — |
| n-max 2 | 83.6 | 65.4 | 76.9 | 76.9 | 0.55-0.95 (0.79) |
| **n-max 3** | **97.9** | 63.6 | **85.4** | **85.4** | 0.47-0.90 (0.73) |
| n-max 4 | 89.8 | 45.1 | 70.8 | 70.8 | 0.33-0.91 (0.61) |

**Optimum at n-max 3: +108% overall (median)** against +88% at n-max 2, the same shape as the RADV XTX below — code keeps climbing (97.9, 0.90 aggregate acceptance) while prose speed sits flat and its acceptance collapses (0.48 aggregate at n-max 3). n-max 4 is past the optimum: +73% overall, prose drops to 45.1 tok/s on 0.33 aggregate acceptance (0.87 code / 0.34 prose / 0.64 bash), the first arm where an extra draft slot costs more than it returns. The spec-off arm was back-filled on the same serve with only the spec flags removed, everything else identical. The n-max 2 arm is the contributor's original run, re-run once to supply acceptance and reproduced within session noise (82.2 / 67.1 / 73.9, overall median 68.3, mean 72.7): aggregate 0.79 (1681/2130 draft tokens, warmup excluded), per-request range 0.55–0.95, code 0.94 / prose 0.60 / bash 0.74. n-max 3 aggregate 0.73 (1740/2369), per-prompt code 0.90 / prose 0.48 / bash 0.75. n-max 4 aggregate 0.61 (1709/2791). Prose carries the low end in every spec arm, exactly as on the RADV XTX.

VRAM (Windows GPU perf counter): 24,560 MiB card with 811 MiB in use before serve (i.e. a clean desktop, not a rule-7 case); 19.9 GiB dedicated baseline, 20.9 GiB at n-max 2, 21.1 GiB at n-max 3, 21.2 GiB at n-max 4; the MTP draft context is ~712 MiB.

Cross-check against the Linux/RADV XTX row above (28.8 → 70.7 at n-max 3, Q4_K_M): this Windows/Vulkan baseline reads far higher (41.0 vs 28.8) and its n-max 3 arm lands above that row's on absolute with-flag numbers. Quant (UD-Q4_K_XL vs Q4_K_M) and backend differ, so the delta comparison is soft — but for anyone holding an XTX, the Windows Vulkan driver at batch-1 decode is not the slow path here.


### RX 7900 XTX (Vulkan): n-max sweep, gate A/B, the desktop tax, and flag stacks don't travel
*by [@Splizard](https://github.com/Splizard), PR #35*

Same card, same config as the row, unchanged `probe.py`, medians of three runs x three prompts. Acceptance from server logs (per-request range, aggregate in parentheses):

| n-max | Overall median | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| off | 28.8 | 28.6 | 28.8 | 28.9 | — |
| 2 | 57.9 | 61.8 | 47.2 | 57.9 | 0.51-0.95 (0.80) |
| **3** | **70.7** | 78.8 | **50.4** | **70.7** | 0.43-0.95 (0.72) |
| 4 | 70.2 | **86.3** | 49.7 | 70.2 | 0.32-1.00 (0.63) |
| 4, `p-min 0.60` | 60.7 | 79.9 | 39.6 | 60.7 | 0.53-0.94 (0.78) |

Same shape as every sweep above: code climbs through n-max 4, prose peaks at 3, acceptance decays monotonically. Overall optimum is 3 (+145% over spec-off, +22% over the launch-command n-max 2), with 4 a statistical tie on the strength of code alone — daily mixed use 3, pure code 4, prose-heavy 2. And rule 2 inverts here exactly as on the three desktop 5090s: the 0.60 gate raised aggregate acceptance from 0.63 to 0.78 while dropping the overall median 13% and prose by 20%. Vulkan verification on this card is cheap enough that every draft is worth attempting — another card class where acceptance is a vanity metric.

Two further screens from the same card, both from a custom streaming harness (not `probe.py`), labeled as screens per the contributing rules:

- **A shared desktop halves everything, silently.** Serving IQ4_XS at 16K next to a live compositor and browser holding 7.3 GB, 3.5 GB of weights spilled to GTT (host RAM over PCIe) and decode read 16.8 baseline / 21.4-29.7 with n-max 2 — prompt processing fell from ~740 to ~60-200 tok/s. The server starts fine and `/health` is green; nothing tells you the weights aren't resident. Check `mem_info_gtt_used` (or your vendor's equivalent) before trusting any number measured on a desk machine.
- **Rule 1 extends to whole flag stacks.** Importing the Strix Halo config from the issues verbatim (`draft-mtp,ngram-mod --spec-draft-n-max 12 --spec-ngram-mod-n-min 24`) measured 7.7 tok/s on prose — *below the unassisted baseline* — and 17.5 on code, against 21.4/29.7 for plain n-max 2 on the same degraded setup. Deep drafts plus ngram chaining that pay on a bandwidth-starved 256-bit APU invert on a 960 GB/s card. Re-derive the stack on your own hardware class, not just n-max.


### AMD Radeon AI PRO R9700 32GB (Windows/Vulkan): first Windows stack A/B
*by [@misterkerns](https://github.com/misterkerns)*

Same silicon as the Linux Vulkan/RADV R9700 row above, different OS and driver: Windows 11 Pro build 26200, AMD driver 32.0.22042.14002 (not RADV), official `ggml.llamacpp` WinGet b10711 (`9723942ad`) `win-vulkan-x64`. unsloth UD-Q4_K_XL Dynamic 3.0, 262K context, q4_0 KV, `--parallel 1`, thinking off. Method: unchanged `probe.py` at `431bf8a`, three runs x three prompts, warmup discarded. Spec arms add only `--spec-type draft-mtp --spec-draft-n-max N` unless a p-min is named. Host Ryzen 9 5900XT, 32 GB RAM. Harbor (llamaswap ROCm gfx1201, ollama ROCm, hermes, webui) and snap Ollama were stopped for the run; Firefox/Discord closed. Pre-serve adapter dedicated ~0.90 GiB; DWM stayed at ~1.24 GiB. llama-server remained fully resident (20.50 GiB baseline / 22.75 GiB at n-max 4 of 32 GB). Baseline was flat to 0.2 tok/s.

| n-max | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Overall median | Acceptance |
|---|---|---|---|---|---|
| off | 29.8 | 29.8 | 29.7 | 29.7 | — |
| 2 | 57.2 | **42.4** | 52.0 | 52.0 | 0.56-0.94 (0.81) |
| **3 ungated** | 67.2 | 41.5 | 56.3 | **56.3** | 0.39-0.97 (0.72) |
| 3, `p-min 0.60` | 66.9 | 41.4 | 56.3 | 56.3 | 0.55-0.95 (0.84) |
| 4 ungated (pass 1) | 68.8 | 39.3 | 62.7 | 62.7 | 0.33-0.91 (0.66) |
| 4 ungated (pass 2) | 69.0 | 35.9 | 51.0 | 51.0 | — |
| 4, `p-min 0.60` | **69.3** | 37.5 | **61.0** | 61.0 | 0.53-0.90 (0.80) |

**n-max 3 ungated is the mixed-workload row: 29.7 → 56.3 (+90%).** n-max 4 is not a resolved win ungated (two passes 62.7 then 51.0, means 56.0 / 52.6). Gating n-max 4 at p-min 0.60 stabilises it (mean 56.0, bash 58.1–62.6) but prose stays at 37.5 — it does not become the mixed daily. Python keeps climbing (57.2 → 67.2 → 69.3), prose peaks at n-max 2 (42.4) and falls at 4.

**p-min on this card is a vanity metric for speed, a real one for acceptance.** At n-max 3, p-min 0.60 is a wash on throughput (56.3 vs 56.3, means 54.6 vs 54.2) while aggregate acceptance rises 0.72 → 0.84 (1863/2593 vs 1898/2265). That is rule 2's "fast card" shape, not the 2×9070 shape: on the dual-9070 pool, p-min 0.60 made n-max 4 the winner; on this single 32GB Navi 48 it does not. R9700 is the same silicon family as 9070 XT with double the VRAM and a 640 GB/s bus — verification is cheap enough that every draft is worth attempting. Ship n-max 3 ungated; add p-min 0.60 only if you want fewer rejected drafts, not more tok/s.

n-max 2 is the cleanest depth-matched comparison against the existing Linux RADV row (27.0 → 43.3 at n-max 2): Windows Vulkan is faster on both arms, +75% vs +60% at the same draft depth. A WSL2 ROCm gfx1201 container on this same box (llama.cpp b10740, same GGUF/flags) read 25.3 → 46.8 at n-max 3 — a third stack, slower than native Windows Vulkan. Harbor's daily llamaswap image was not used for these numbers.

Acceptance is from `draft acceptance` log lines, warmup excluded. Aggregates: n-max 2 1621/2013 = 0.81, n-max 3 ungated 1863/2593 = 0.72, n-max 3 p-min 0.60 1898/2265 = 0.84, n-max 4 p-min 0.60 1816/2270 = 0.80. Dedicated process VRAM: 20.50 GiB baseline, 22.75 GiB at n-max 4.

**Cookbook / landmines** (same box, documented so nobody re-learns them): the sudoingX launch shape (UD-Q4_K_M, 131K, n-max 2) measured **52.6**, below n-max 3 XL. n-max 6 dropped prose to **27.5**, under the 29.7 baseline. `draft-dflash` without a sidecar GGUF is a silent no-op (29.6). Official `llama-b10740-bin-win-rocm-7.14-x64.zip` loads MTP then decodes at **~5 t/s** — the zip is not a gfx1201 HIP build; do not use it on R9700. Vulkan device string on this Windows driver reports **shared memory 32768** (RADV often 65536). Full hunt log: keep local `LAB.md` with the PR notes.
