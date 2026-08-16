# Qwen3.8-27B MTP: the flag was free the whole time

One llama.cpp flag unlocks +33% to +107% decode speed for Qwen3.8-27B on consumer GPUs, depending on the card. No new files, no conversion, no custom build. The MTP head already ships inside the GGUF you downloaded on launch night.

Opened hours after the Aug 14 2026 release. Within the first day the community grew it into a living record: **ten machines, eight contributors, every GPU vendor, and three tuning rules nobody knew at launch**, see [Community numbers](#community-numbers).

## The numbers

The two founding rows, measured the night of the drop. The full living table (10 machines, all vendors) lives in [Community numbers](#community-numbers).

| Card | Baseline | With the flag | Gain | Acceptance |
|---|---|---|---|---|
| RTX 3090 24GB | 31.0 tok/s | **41.3 tok/s** | **+33%** | 0.76-0.80 |
| RTX 5090 mobile 24GB | 36.7 tok/s | **50.9 tok/s** | **+39%** | 0.76-0.82 |

Paired A/B: same card, same GGUF, same config both sides. Live llama-server with a streaming client, every generated token clocked, warmup discarded, medians of 3 runs x 3 prompts, thinking off, 131K context resident, q4_0 KV cache, unsloth Q4_K_M. These are serve measurements, not llama-bench numbers.

## The flag

```
--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
```

## Full launch command

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
  -c 131072 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
```

The KV cache flags matter on their own: without them, context creation fails past roughly 90K next to 17GB of weights. With them, the full 262K window fits a 24GB card at 22.2GB (drop `-c` to 262144 and remove the spec flags if you want maximum window instead of maximum speed).

Weights: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Official model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Tuning n-max

Swept on the 5090 mobile, same method:

| n-max | Overall median | Code prompts | Prose prompts | Acceptance |
|---|---|---|---|---|
| **2** | **50.9** | 56.4 | 42.5 | 0.76-0.82 |
| 3 | 48.3 | 59.4 | 37.9 | 0.68 |
| 4 | 47.3 | 60.2 | 33.4 | 0.65 |

Acceptance decays as you draft deeper and prose pays for it first. Run 2 as your daily, 3 if your session is pure code.

## The three rules the community found

Discovered by contributors in the table within the first day, detailed in the sections and footnotes below:

1. **The n-max sweet spot is card-dependent.** 24GB cards peak at n-max 2, the 48GB A6000 peaked at 4 ([@lingster](https://github.com/lingster)). Bigger cards absorb deeper verification before acceptance decay eats the win.
2. **`--spec-draft-p-min` is the second knob.** A confidence gate (~0.60) stops the head drafting when it's unsure, which makes deeper n-max nearly free instead of harmful ([@tomertec](https://github.com/tomertec)).
3. **The gain scales with generation length.** Under ~400 output tokens the spec overhead can dominate; at 4096 tokens the same setup ran +60% ([@Spadav](https://github.com/Spadav)). Long work benefits most.

## How it works

Qwen trained multi-token-prediction (nextn) layers into Qwen3.8. The quantizers kept them: unsloth's GGUFs carry the `blk.*.nextn.*` tensors, which llama.cpp loads and, without the flag, ignores. llama.cpp added draft-mtp speculative decoding in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (July 2026): the server drafts tokens with the built-in head and verifies them with the main model, so accepted drafts cost a fraction of a full forward pass. Everything was in place on release night. The flag connects it.

## Caveats

- `--parallel 1` is required for now, single slot only
- prompt processing takes a small hit from device-to-host embedding transfers
- these are day-one llama.cpp speeds through the qwen3_5 code path, the hybrid attention kernels are young and the floor should rise with upstream work
- your absolute numbers will differ with hardware, drivers, and thermals, the deltas are the durable part

## Measure it yourself

`probe.py` is the streaming client behind every number here. It clocks every generated token (reasoning and content deltas both) against a live server and prints per-prompt medians.

```bash
python3 probe.py                 # defaults to http://127.0.0.1:8080
python3 probe.py http://127.0.0.1:8090
```

Run it once against a baseline serve and once with the flag, same everything otherwise. That pairing is the whole method.

## Community numbers

Ran the A/B on your card? Open a PR and add a row.

| Card | Baseline | With flag | n-max | Acceptance | Contributor |
|---|---|---|---|---|---|
| RTX 3090 24GB | 31.0 | 41.3 | 2 | 0.78 | [@sudoingX](https://x.com/sudoingX) |
| RTX 5090 mobile 24GB | 36.7 | 50.9 | 2 | 0.79 | [@sudoingX](https://x.com/sudoingX) |
| RTX 4090 24GB | 47.7 | 76.3 | 2 | 0.56 | [@Spadav_](https://x.com/Spadav_) |
| RTX A6000 48GB (Ada) | 26.7 | 52.5 | 2 | 0.54-0.98 | [@lingster](https://github.com/lingster) |
| RX 7900 XTX 24GB | 30.7 | 43.9 | 2 | 0.60-0.95 | [@Jqianggu](https://x.com/Jqianggu) |
| 2× RTX 3090 + 3090 Ti 24GB (TP) | 49.1 | 81.1 | 2 | 0.52-0.96 | [@guilhermedemelocabral](https://github.com/guilhermedemelocabral) |
| RTX 4090 24GB (UD-Q4_K_XL) | 36.1 | 74.8 | 2 | 0.56-0.94 | [@rkvhtd](https://github.com/rkvhtd) |
| 2x RX 9070 16GB (Vulkan) | 22.1 | 41.6 | 2 | 0.73 | [@tomertec](https://github.com/tomertec) |
| AMD Radeon AI PRO R9700 32GB | 27.0 | 43.3 | 2 | 0.60-0.94 | [@ajnytebot](https://github.com/ajnytebot) |
| Ryzen AI Max+ 395 / Radeon 8060S | 11.5 | 23.7 | 2 | 0.52-0.94 | [@shiwuxiu](https://github.com/shiwuxiu) |
| 2× RTX 5060 Ti 16GB (PP, default `-sm layer`) | 22.1 | 42.8 | 2 | 0.53-0.94 | [@Jackwwg83](https://github.com/Jackwwg83) |
| 2× RTX 5060 Ti 16GB (TP, `-sm tensor`) | 37.1 | 65.9 | 2 | 0.51-0.88 | [@Jackwwg83](https://github.com/Jackwwg83) |

\* A6000 row: unsloth Q8_K_XL, 256K context, q8_0 KV cache — 40.0 GB VRAM baseline, 41.4 GB with spec (rows above: Q4_K_M, 131K, q4_0 KV).
\* RX 7900 XTX row: unsloth Q4_K_M, 131K context, q4_0 KV cache — 18.9 GB VRAM baseline, 19.7 GB with spec.
\* 3×24GB TP row: two RTX 3090 + one 3090 Ti, Unsloth UD-Q6_K_XL, tensor-parallel `--split-mode tensor`, `--parallel 4`, 500K unified KV pool, q8_0 KV, f16 draft KV, mmproj Q8, temp 1.0. VRAM ~16.6 GB/GPU baseline, ~18.7 GB/GPU with spec (tightest card). Method: stock `probe.py`. `--parallel 1` was not required on this host.
\* RTX 4090 row: unsloth UD-Q4_K_XL, 160K context, q4_0 KV cache, llama.cpp b10360, Windows/CUDA, 275W power limit — 20.6 GB VRAM baseline, 21.9 GB with spec.
\* RX 9070 row: two 16GB cards on one 31.84 GiB pool, Vulkan build b10426, unsloth UD-Q4_K_XL, **262K context**, q8_0 KV cache — 28.0 GiB across the pool with spec. Method differs from the rows above and is spelled out under the sweep below.
\* R9700 row: unsloth UD-Q4_K_XL, 262K context, q4_0 KV cache, llama.cpp b10433, Vulkan/RADV — 22.53 GB VRAM baseline, 24.55 GB with n-max 2. Method: unchanged `probe.py` at commit `67c20536`, three runs x three prompts, thinking off.
\* Ryzen AI Max+ 395 row: 64GB unified memory, unsloth UD-Q4_K_XL, 32K context, q8_0 KV cache, llama.cpp b10437, Windows build 26200, Vulkan with AMD driver 32.0.31035.1003. Method: unchanged `probe.py` at commit `67c2053`, three runs x three prompts, thinking off.
\* RTX 4090 Spadav_ row: unsloth Q4_K_M, 200K context, q4_0 KV cache, q8_0 draft KV, mmproj loaded (888MB on GPU) — method: stock probe.py + 4096-token curl (MTP crossover: overhead dominates at ≤400 tokens, +60% at 4096 tokens).
\* 2× RTX 5060 Ti rows: unsloth UD-Q4_K_XL (sha256 `bee238bb…1372`), 131K context, q4_0 KV cache, llama.cpp built from source at commit `ece963f4` with `-DCMAKE_CUDA_ARCHITECTURES=120`, CUDA 13.0 / driver 580.173.02, PCIe 3.0 x8, `PHB` topology (no P2P). Method: unchanged `probe.py`, three runs x three prompts, thinking off. VRAM: PP 21.7 GB baseline / 23.0 GB with spec; TP 20.9 GB / 22.1 GB. The 16.68 GiB of weights do not fit one 16GB card, so two cards is the floor on this box — the split-mode choice is not optional here, which is what makes the two rows worth reading side by side. Both rows verified at `n_ctx_slot = 131072`. Details under the sweep below.

### A6000 48GB: n-max sweep

Same A6000, same config as the row above, `--spec-draft-n-max` swept 2-6. Overall and per-prompt probe medians (tok/s), draft acceptance from the server log:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 52.5 | 57.0 | 43.1 | 52.5 | 0.54-0.98 |
| 3 | 60.7 | 67.6 | 44.1 | 60.7 | 0.42-0.91 |
| **4** | **64.1** | 77.1 | 41.6 | 64.1 | 0.32-0.93 |
| 5 | 62.8 | 80.5 | 40.8 | 62.8 | 0.29-0.84 |
| 6 | 58.6 | 84.3 | 37.4 | 58.6 | 0.23-0.84 |

The overall peak is n-max 4 here, not 2 — the card has enough headroom to absorb the cost of deeper verification before the acceptance decay eats the win. Same shape as the 5090 sweep: the code prompts keep rising all the way up (84.3 at n-max 6), the prose prompt falls from the start (43.1 -> 37.4), and acceptance decays monotonically. Daily mixed use: 4, pure code sessions: 5-6, prose-heavy: 2.

### 3× RTX 3090 / 3090 Ti 24GB (TP): n-max sweep

Same 3×24GB host, same config as the row above, `--spec-draft-n-max` swept 2-6. Overall and per-prompt probe medians (tok/s), draft acceptance from the server log:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 81.1 | 96.2 | 68.8 | 81.1 | 0.52-0.96 |
| **3** | **95.6** | 108.1 | 71.1 | 95.6 | 0.38-0.92 |
| 4 | 94.9 | 114.7 | 67.2 | 94.9 | 0.37-0.92 |
| 5 | 93.3 | 115.9 | 55.7 | 93.3 | 0.29-0.84 |
| 6 | 88.7 | 112.9 | 54.1 | 88.7 | 0.25-0.80 |

Overall peaks at n-max 3. Code keeps climbing through n-max 5 (115.9); prose falls after n-max 3 (71.1 → 54.1). Same shape as the 5090 and A6000 sweeps. Daily mixed use: 2–3; pure code: 4–5; prose-heavy: 2. A separate greedy code-completion hash gate (not `probe.py`) matched n-max 2 and diverged at n-max 1/3/4; chaining `ngram-mod` made n-max 2 unstable on that gate, so this host still ships n-max 2 without ngram.

### 2x RX 9070 16GB: n-max sweep, and the second knob nobody is turning

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

Same R9700, same unsloth UD-Q4_K_XL model and serving config as the row above, at 262K context with q4_0 KV. Upstream `probe.py` unchanged: medians of three runs per prompt, thinking off. Draft acceptance is from the server log.

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 43.3 | 45.3 | **36.3** | 43.3 | 0.60-0.94 |
| **4** | **47.4** | **75.5** | 33.3 | **47.4** | 0.28-0.90 |

N-max 4 lifts the overall median another 9.5% over n-max 2, driven by a 66.7% jump on the Python prompt. Prose falls 8.3% as aggregate acceptance drops from 82.3% to 60.5%; prose-only acceptance is 29.4% at n-max 4. The speed shape supports n-max 4 as a code-specialized arm, while n-max 2 remains the mixed-workload default. Loaded VRAM: 22.53 GB spec-off, 24.55 GB at n-max 2, and 24.87 GB at n-max 4.

### Ryzen AI Max+ 395 / Radeon 8060S: n-max 2 versus 4

Same 64GB Strix Halo system, same unsloth UD-Q4_K_XL model and serving config as the row above: Windows Vulkan, 32K context, q8_0 KV, Flash Attention, batch 2048/512, and `--parallel 1`. Upstream `probe.py` was unchanged; all values are medians of three runs per prompt with thinking off. Acceptance is from the server log, excluding warmup.

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| **2** | **23.7** | 25.2 | **19.1** | **23.7** | 0.52-0.94 (78.2% aggregate) |
| 4 | 23.5 | **29.9** | 16.5 | 23.5 | 0.35-0.91 (65.8% aggregate) |

MTP n-max 2 doubles the overall median versus the 11.5 tok/s spec-off control (+106%). N-max 4 is effectively flat overall (-0.8% versus n-max 2), but the workload split is sharp: Python rises 18.7% while prose falls 13.6%. This system keeps n-max 2 for mixed use and treats n-max 4 as a code-specialized option.

### 2× RTX 5060 Ti 16GB: on a multi-GPU box the split mode is worth more than the flag

Same box, same GGUF, same serve config throughout — only `--split-mode` and the spec flags change. Stock `probe.py`, medians of three runs x three prompts, thinking off. Acceptance from the server log.

| split mode | spec | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|---|
| `layer` (default) | off | 22.1 | 22.0 | 22.1 | 22.0 | — |
| `layer` | n-max 2 | 42.8 | 47.3 | 34.4 | 42.8 | 0.53-0.94 |
| `layer` | n-max 3 | **47.5** | 54.0 | 35.4 | 47.5 | 0.48-0.75 |
| `layer` | n-max 4 | 45.2 | 60.1 | 32.1 | 45.2 | 0.39-0.72 |
| `tensor` | off | **37.1** | 37.1 | 37.3 | 37.0 | — |
| `tensor` | n-max 2 | 65.9 | 70.6 | 51.4 | 65.9 | 0.51-0.88 |
| `tensor` | n-max 3 | 69.3 | 78.7 | 53.6 | 69.3 | 0.38-0.74 |
| `tensor` | n-max 4 | **71.3** | 83.4 | 47.6 | 71.3 | 0.35-0.70 |
| `tensor` | n-max 5 | 67.1 | | | | |
| `tensor` | n-max 6 | 56.2 | | | | |
| `tensor` | n-max 8 | 59.7 | | | | |
| `tensor` | n-max 4, `p-min 0.60` | 46.2 | 71.6 | 23.3 | 46.2 | 0.41-0.82 |
| `tensor` | n-max 6, `p-min 0.60` | 45.4 | 65.8 | 24.3 | 45.4 | 0.56-0.85 |
| `row` | off | fails to load | | | | |

**The default split mode costs more than the flag gains.** `--split-mode layer` splits layers across cards, so at batch 1 there is nothing to pipeline: GPU0 computes while GPU1 idles and the pair decodes at single-card bandwidth. `--split-mode tensor` splits each matmul instead, both cards read weights at once, and the baseline goes 22.1 -> 37.1 (**+68%**) before any spec flag is involved. That is a larger free win than MTP gives on a 24GB single card.

The two levers stack cleanly: **22.1 -> 69.3, a 3.14x end-to-end gain from two flags, neither of which costs anything.** Worth spelling out because `layer` is the default, so a two-card owner who follows the launch command verbatim measures 42.8 and stops there.

The all-reduce is not free but it is cheaper than expected here: these are GeForce cards with P2P disabled, on PCIe 3.0 x8 with a `PHB` topology, so every reduction crosses host memory. Ideal scaling would put TP at 44.2 (2x the 22.1 single-card-equivalent); the measured 37.1 says the reduction costs about 16%. A host with working P2P should keep more of it.

Two things worth knowing before you try this:

- `--split-mode row` refuses to load — `device CUDA0 does not support split buffers`. `tensor` is the mode that works.
- `--split-mode tensor` skips the auto-fit pass. `llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort` is a warning, not an error, and the server starts anyway with whatever `-c` you passed. Check `n_ctx_slot` in the log before comparing against a `layer` run, or you may be comparing two different context sizes. Both arms above were verified at `n_ctx_slot = 131072`.

**The n-max sweet spot moves with the split mode, not just the VRAM pool.** Same box, same 32GB, same everything else: `layer` peaks at n-max 3 (47.5, falling to 45.2 at 4) while `tensor` peaks at n-max 4 (71.3), holds 67.1 at 5 and only collapses at 6. Rule 1 says the sweet spot is card-dependent — this says it is also *topology*-dependent, because tensor-split makes each verification step cheaper, so the pool can absorb a deeper draft before acceptance decay eats the win. If you switch split mode, re-sweep n-max; the old optimum does not carry over.

Everything else matches the other sweeps: code prompts keep climbing (83.4 at n-max 4 under `tensor`), prose falls from the start (53.6 at n-max 3 down to 47.6 at 4), acceptance decays monotonically.

**`--spec-draft-p-min` is a large net loss here**, which is the opposite of what it does on the RX 9070 pair. Gating n-max 4 at 0.60 drops the overall median from 71.3 to 46.2 — and it does so *while raising* acceptance from 0.35-0.70 to 0.41-0.82. The gate is refusing drafts that would have been accepted; the prose prompt takes the worst of it (53.6 → 23.3). Both rigs are 2x16GB, so the knob clearly does not generalize across backends — worth sweeping rather than adopting.

**The gain does not scale with output length on this box.** Running the same two arms at 4096 tokens instead of 400 (probe.py with only `MAX_TOKENS` changed): baseline 37.1, MTP n-max 3 68.7 — a 1.85x ratio against 1.87x at 400 tokens. Rule 3 holds where spec overhead dominates short generations; here the tensor-split baseline is bandwidth-bound and stable, so the full gain is already present at 400 tokens and there is nothing left to recover at 4096.

`llama-bench` on the same box as a cross-check, `-fa 1 -ctk q4_0 -ctv q4_0`:

| split mode | pp512 | tg128 | pp4096+tg128 |
|---|---|---|---|
| `layer` | 926.96 ± 18.81 | 22.58 ± 0.02 | 482.17 ± 0.22 |
| `tensor` | 1028.04 ± 2.69 | **38.37 ± 0.26** | 550.86 ± 0.77 |

`llama-bench` puts the decode gain at +70%, `probe.py` at +68% — two different tools, two different prompt sets, same answer. Prefill moves only +11%, which is the expected shape: prompt processing is compute-bound and already batched, so layer-split leaves both cards reasonably busy; decode at batch 1 is where the serialization actually bites.

#### Decode against prompt depth

`probe.py` runs at near-zero context, which is the easiest case for any spec-decode setup. Both arms re-measured under `-sm tensor` with a filler prompt sized against the server's own `/tokenize` endpoint, so the depths are token-exact rather than a chars-per-token guess. TTFT excluded from the decode rate, as in `probe.py`.

| prompt tokens | spec off: TTFT | prefill t/s | decode t/s | MTP n-3: TTFT | prefill t/s | decode t/s | ratio |
|---|---|---|---|---|---|---|---|
| 4,075 | 2.59 s | 1,572 | 36.24 | 2.80 s | 1,454 | 56.05 | 1.55x |
| 16,255 | 7.57 s | 2,148 | 34.53 | 8.25 s | 1,970 | 53.27 | 1.54x |
| 32,467 | 10.51 s | **3,089** | 32.38 | 11.53 s | 2,817 | 50.79 | 1.57x |
| 64,891 | 22.35 s | 2,904 | 28.73 | 24.50 s | 2,649 | 56.43 | 1.96x |
| 129,781 | 52.33 s | 2,480 | 23.70 | 58.11 s | 2,233 | 46.34 | 1.96x |

**The flag matters more the deeper you go, not less.** The unassisted arm loses 35% of its decode rate between 4K and 128K (36.24 → 23.70); the MTP arm loses 17% (56.05 → 46.34). At 128K of resident context this pair still generates at 46 tok/s.

Two caveats on this table. The 64K and 128K rows generated only ~42 tokens before the model stopped, so those decode figures rest on a much smaller sample than the shallower rows — treat the 1.96x as indicative, not settled. And prefill peaking mid-sweep (3,089 t/s at 32K, lower at both 4K and 128K) is the usual batching-vs-attention-cost curve, not a measurement error.

#### Concurrency, and whether MTP survives `--parallel 4`

Upstream says `--parallel 1` is required for spec-decode; the 3x3090 row reports it was not required on that host. On this one **the server starts and serves correctly with `--parallel 4` and MTP together**. Whether it *helps* is a different question. 16K prompts, `-sm tensor`, per-stream is the median stream, aggregate is total generated tokens over wall clock including prefill:

| N | spec off: per-stream | aggregate | MTP n-3: per-stream | aggregate |
|---|---|---|---|---|
| 1 | 34.19 | 6.91 | **57.99** | 10.72 |
| 2 | 19.32 | 18.53 | 20.94 | 16.34 |
| 4 | 19.13 | **29.48** | 19.21 | **29.24** |

**Speculative decoding is a single-stream optimisation.** At N=1 it is worth 1.70x. By N=2 the advantage is inside the noise, and at N=4 the two arms are indistinguishable (29.48 vs 29.24) — once the batch is full the GPU has real work queued and there is no idle verification capacity for the draft head to exploit. If you serve concurrent users, tune `--parallel` and skip the spec flags; if you are one person at a terminal, the flags are most of your speed.

One side effect worth a line: across every arm the MTP runs drew *less* power than their spec-off controls while producing more tokens — combined peak 268 W vs 293 W on the depth sweep, 267 W vs 305 W under concurrency. The draft head is not buying speed with watts.

## License

Apache-2.0. The numbers and verdicts are real, the conclusions are mine.
