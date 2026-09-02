# Multi-GPU rigs — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP, moved verbatim from the main README. Each section carries its author and original PR. Add yours via a PR to this file, row and footnote go in the main [community table](../README.md#community-numbers).

### 3× RTX 3090 / 3090 Ti 24GB (TP): n-max sweep
*by [@guilhermedemelocabral](https://github.com/guilhermedemelocabral), PR #4*

Same 3×24GB host, same config as the row above, `--spec-draft-n-max` swept 2-6. Overall and per-prompt probe medians (tok/s), draft acceptance from the server log:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 81.1 | 96.2 | 68.8 | 81.1 | 0.52-0.96 |
| **3** | **95.6** | 108.1 | 71.1 | 95.6 | 0.38-0.92 |
| 4 | 94.9 | 114.7 | 67.2 | 94.9 | 0.37-0.92 |
| 5 | 93.3 | 115.9 | 55.7 | 93.3 | 0.29-0.84 |
| 6 | 88.7 | 112.9 | 54.1 | 88.7 | 0.25-0.80 |

Overall peaks at n-max 3. Code keeps climbing through n-max 5 (115.9); prose falls after n-max 3 (71.1 → 54.1). Same shape as the 5090 and A6000 sweeps. Daily mixed use: 2–3; pure code: 4–5; prose-heavy: 2. A separate greedy code-completion hash gate (not `probe.py`) matched n-max 2 and diverged at n-max 1/3/4; chaining `ngram-mod` made n-max 2 unstable on that gate, so this host still ships n-max 2 without ngram.


### 2× RTX 5060 Ti 16GB: on a multi-GPU box the split mode is worth more than the flag
*by [@Jackwwg83](https://github.com/Jackwwg83), PR #24*

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


### 2× Tesla P40 24GB (tensor split): n-max sweep on Pascal
*by [@lyesrock](https://github.com/lyesrock), PR #29*

Same box as the row above — two P40s (sm_61, 732 GB/s each, no P2P, separate NUMA nodes), unsloth UD-Q5_K_XL, 131K context, q4_0 KV, `--tensor-split 1,1` under `numactl --interleave=all`, `--parallel 1`, thinking off. `probe.py` unchanged, medians of three runs x three prompts, only the spec flags changing between arms. Draft acceptance from the server log:

| config | Overall median | Overall mean | Prefill 20K | Acceptance |
|---|---|---|---|---|
| spec off | 11.7 | 11.8 | 354 | — |
| n-max 2, p-min 0.75 | 21.1 | 20.0 | 300 | 0.73-0.94 |
| **n-max 4, p-min 0.75** | **22.6** | **21.6** | 295 | 0.68-0.87 |
| n-max 6, p-min 0.75 | 21.1 | 20.5 | 290 | 0.63-0.81 |
| n-max 6, p-min 0.60 | 20.1 | 19.7 | 290 | 0.46-0.74 |

**+92% at the peak (22.6 vs 11.7) — the largest multi-GPU delta in the table, on the oldest architecture here.** Pascal has no tensor cores and the dense 27B decode is bandwidth-starved exactly like the 890M and RX 9070 rows, so the rule-1/rule-2 shape holds: amortising the weight read across accepted drafts is the whole game.

Two specifics that differ from the newer cards:

- **n-max 4 beats n-max 6, not the other way around.** The 3×3090 and 5060 Ti tensor rows peak at 3-4 too, but here n-max 6 loses ground (acceptance 0.63-0.81 vs 0.68-0.87) and n-max 8 was not run because the trend was already falling. The verification batch on sm_61 is expensive enough that the deeper draft stops paying.
- **p-min 0.75 helps, p-min 0.60 hurts** — the RX 9070 direction, not the 5090-desktop inversion. Gating at 0.60 collapsed acceptance to 0.46-0.74 and cost ~2 tok/s. On this class of card the gate rescues the deep draft; it is not a vanity knob.

The prefill tax is the usual ~15-17% (354 → 295 tok/s), the standard cost of device-to-host embedding transfers noted in the caveats section.

Production follow-up on the same box, **f16 KV instead of q4_0** (the cards have room; q4_0 only matters when the 131K-262K context must fit a tight pool): n-max 4 at p-min 0.75 still measures 22.3 mean / 24.0 median over the same probe — the q4_0 in the table row was a fit decision, not a speed one, and f16 KV reproduces it. Daily mixed use: 4.


### Companion: the MTP gain and the n-max optimum are per-model
*by [@lyesrock](https://github.com/lyesrock), PR #29*

Same box, same build (b10453), same method, each model at its production context. KV cache: q4_0 for the Qwen3.8-27B arm (the paired A/B above), f16 for the rest — the production follow-up verified f16 reproduces the 3.8 peak, so the table stays comparable. Baseline / with-flag decode medians (tok/s), `probe.py` unchanged:

| Model (UD-Q5_K_XL) | Baseline | Best spec arm | Peak | Gain | Acceptance |
|---|---|---|---|---|---|
| Qwen3.8-27B dense | 11.8 | n-max 4, p-min 0.75 | 22.6 | +92% | 0.68-0.87 |
| Qwen3.6-27B dense | 12.0 | n-max 6, p-min 0.75 | 22.2 (median 24.3) | +85% | 0.78-1.00 |
| Qwen3.6-35B-A3B MoE | 49.6 | n-max 4, p-min 0.75 | 67.1 | +35% | 0.83-0.99 |
| Qwen3.5-9B | 34.2 | n-max 4, p-min 0.75 | 46.8 | +37% | 0.68-0.95 |
| Qwen3.5-4B | 53.8 | n-max 2, p-min 0.75 | 67.5 | +25% | 0.82-0.99 |

Three things the sweep adds to rule 1:

- **The optimum n-max moves between models of the same size and class.** Qwen3.8-27B (dense) peaks at n-max 4; Qwen3.6-27B (dense, same quant, same cards) peaks at n-max 6. Same hardware, same method, different winner — re-sweep when the model changes, not just when the GPU does.
- **MoE collapses the gain.** The 35B-A3B is 4× faster than the dense 27B with the flag *off*, and the flag only adds +25-35% on top of that instead of +85-92%. The active-3B parameters are not bandwidth-bound at batch 1, so there is less to amortise — rule 3 in its purest form, and probably the single most useful row for anyone picking between the two.
- **Smaller models want shorter drafts.** The 4B peaks at n-max 2 (67.5) and n-max 4 is already behind (66.3); the 9B at n-max 4. With the model's own weights fitting in a small fraction of the pool, the verify cost per drafted token dominates faster.

p-min 0.75 won on every arm here; the MoE arms kept 0.83-0.99 acceptance at n-max 4, so gating is nearly free where acceptance was healthy to begin with.

### Independent Q4_K_M replication and production follow-up
*by [@mgoswick](https://github.com/mgoswick), PR #32*

A second Windows dual-5060-Ti host independently repeated the tensor-split sweep using Unsloth Q4_K_M rather than UD-Q4_K_XL. Same GGUF and serving configuration within every A/B; only the spec flags changed. Upstream `probe.py` was unchanged, with three runs x three prompts and thinking off.

| config | Overall median | Overall mean | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Aggregate acceptance |
|---|---:|---:|---:|---:|---:|---:|
| spec off | 38.3 | 38.3 | 38.3 | 38.5 | 38.3 | — |
| n-max 2 | 72.0 | 69.5 | 79.2 | 56.3 | 72.0 | 78.1% |
| **n-max 3** | **76.0** | **73.8** | 86.6 | **57.5** | **76.0** | 71.4% |
| n-max 4 | **76.0** | 72.8 | **94.4** | 51.2 | **76.0** | 64.8% |

The result reproduces the main finding above with a different quant and host: tensor split lifts the spec-off baseline from 23.1 tok/s under layer split to 38.3 (+66%), before MTP. Tensor n-max 3 reaches 76.0 tok/s, 3.29x the original layer-split baseline. N-max 3 is the mixed-workload choice because it has the highest mean and prose result; n-max 4 remains the code-specialized arm.

A separate production-layout follow-up loaded the BF16 vision projector and used q8_0 for both main and draft KV. At 65,543 active prompt tokens, tensor n-max 3 decoded at 58.75 tok/s; at 129,007 tokens it decoded at 44.26 tok/s after 707.4 tok/s prefill. Both depths returned three hidden needle codes exactly. Two deterministic 1,200-token agent tasks decoded at 62.9 tok/s for business operations and 83.4 tok/s for Python, and their streamed responses matched buffered output byte-for-byte.

One Windows-specific memory result: `--load-mode none` reduced physical RAM added after load from 16.33 GiB to 1.85 GiB versus default `auto`/mmap. The same two 1,200-token outputs remained byte-identical, with effectively unchanged decode (63.18/83.58 versus 63.06/83.95 tok/s). Executable worker gates also passed literal CRUD (5/5), deterministic scoring (5/5), and structured-agent JSON validation. These are supplemental production checks, not part of the main-table paired A/B.


### RTX 2070 8GB + RTX 5060 Ti 16GB (mismatched Turing+Blackwell, x1 riser): the free split-mode win shrinks when the interconnect is slow, and cache quant doesn't buy the context it promises once MTP is in the mix
*by [@Lucas12807](https://github.com/Lucas12807), unsolicited*

A budget mismatched pair, not a matched multi-GPU box: RTX 2070 8GB (Turing) migrated out of
the case onto a powered USB x1 riser (PCIe 2.0 x1, ~500 MB/s) because the AM4 board
(ASUS PRIME B450M-A) has only one physical x16 slot, which the RTX 5060 Ti 16GB (Blackwell)
keeps, also driving the desktop. Every other host in this file has at least a real x4/x8 PCIe
link or NVLink between cards; this is the slowest interconnect in the table by a wide margin
(~16x slower than the PCIe 3.0 x8 in the 5060 Ti pair's own row above). unsloth
Qwen3.8-27B-UD-Q4_K_XL, llama.cpp b10662 (CUDA 13.3), Windows 11, `--parallel 1` throughout.
Method: fixed-prompt `/completion` requests at temp=0 against a live server, not `probe.py`
— comparable in shape, not identical tooling, and single-digit sample counts per arm (2-3),
so treat these as directional rather than as tight as the rows above.

#### Split mode still wins, but the riser caps how much

Same GGUF, same `-c 65536`, q8_0 KV, only `--split-mode` and the spec flag changing:

| split mode | spec | tok/s |
|---|---|---|
| `layer` (auto, default) | off | 19.7 |
| `layer` | n-max 1 | 30.6 |
| `layer` | n-max 2 | 37.1 |
| **`layer`** | **n-max 3** | **37.8** |
| `layer` | n-max 4 | 35.4 |
| `tensor --tensor-split 5,2` | off | 25.5 |
| `tensor --tensor-split 5,2` | n-max 2 | **37.7** |
| `tensor --tensor-split 5,2` | n-max 3 | 36.1 |
| `tensor --tensor-split 5,2` | n-max 4 | 32.5 |

`tensor` still beats `layer` on the baseline (19.7 → 25.5, **+29%**), confirming rule 4 holds
even on a riser this slow — but +29% is well under the +68% the same card pair's own row above
measured on PCIe 3.0 x8, and under the +48% the NVLinked A5000 pair saw. The all-reduce that
tensor-split requires every layer has to cross the x1 link, so it eats more of the free win
here than on any other host in this table.

**The n-max optimum also moves down, not up, once tensor-split is in play** — the opposite of
what the 5060 Ti pair's own PCIe-x8 row found (`tensor` peaking at n-max 4 there). Here
`tensor` peaks at n-max 2 (37.7) and n-max 4 is already below the `layer` peak (32.5 vs 37.8).
Reading rule 1 alongside this row: the sweet spot depends on topology *and* how slow that
topology is — a fast interconnect can absorb a deeper draft under tensor-split, a slow one
can't, so re-sweep n-max after switching split mode even when another host with the same card
pair already published one.

Ended up landing on `tensor-split 13,7` in production (proportionally more weight on the 2070,
its floor before OOM on the MTP draft-context allocation), not `5,2` — moving load toward the
smaller card cost only ~1 tok/s (36.0→34.9) while giving it more working margin. Bisection
across ratios found the OOM boundary is much narrower than a linear extrapolation from VRAM
headroom would suggest: `13,7` (35% of weights on the 2070) loads fine, `5,3` (37.5%) OOMs
on the MTP draft buffer alone. The draft-context allocation happens last, after weights and
main KV cache, so a ratio that looks fine by every earlier log line can still fail at the very
end of load.

#### Cache K/V quantization doesn't double context once MTP owns the last few hundred MB

Expected doubling the KV cache compression (q8_0 → q4_0) to roughly double the max context at
fixed VRAM. Measured on the production config (`tensor-split 13,7`, `--no-mmproj-offload`,
MTP n-max 2):

| cache | max `-c` that loads and completes a real generation |
|---|---|
| q8_0 | 73728 |
| q4_0 | 98304 (**+33%**, not +100%) |

Both ceilings fail one step higher (81920 for q8_0, 122880 for q4_0) on the same error: OOM
on the 2070 while allocating the MTP draft context, not the main KV cache. The K/V cache
itself did shrink by roughly half as expected — the shortfall is that MTP's own draft
context and the `--split-mode tensor` bookkeeping are fixed-size overhead on the 8GB card
that doesn't shrink with the cache quant, and on a card this tight that fixed overhead eats
most of the freed headroom before it becomes usable context. Perplexity on a 200KB
real-code corpus (Flask + Express sources, `llama-perplexity`) was 1.3709 (q8_0) vs 1.3756
(q4_0) — +0.34%, inside noise — and 13/14 hand-verified coding tasks (executed, not just
read) matched between the two cache types, so the quant itself is close to free on this
hybrid-attention model; the context ceiling just isn't where a linear model of "half the
bytes, double the tokens" would put it. Worth checking before assuming a KV-quant switch pays
its full theoretical dividend on any card that's also carrying MTP's own buffers near its
limit.

`--no-mmproj-offload` (keeps the vision projector resident in host RAM instead of VRAM,
loading it back only per request) was free on this box: 34.98 vs 34.88 tok/s text-only,
identical correct output on an image-description request, and the vision projector's ~1.1 GB
moved off the 5060 Ti permanently — no VRAM spike observed even mid-request with an image.
Worth trying on any tight vision-capable multi-GPU box before spending a context-size budget
on the projector.


### RTX 5080 16GB + RTX 3090 24GB (tensor split): mixed Blackwell + Ampere
*by [@plyra](https://x.com/plyra)*

Same host, same GGUF, same serve config — only the spec flags change. Stock `probe.py`, medians of three runs x three prompts, thinking off. Acceptance from the server `draft acceptance` lines (warmup excluded).

| spec | Overall median | Overall mean | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---:|---:|---:|---:|---:|---|
| off | 55.6 | 54.9 | 55.7 | 54.0 | 55.8 | — |
| n-max 2, ungated | **92.9** | 91.4 | 105.2 | 79.2 | 92.9 | 0.51-0.95 (agg ~0.79) |

This is the first mixed-architecture row in the table (Blackwell 5080 + Ampere 3090, no NVLink). `--split-mode tensor` was set on both arms before any spec flag (rule 4). Weights + 131K q4_0 KV fit the pair with headroom: 12.0+10.3 GB baseline, 12.8+11.1 GB with spec.

Code still climbs more than prose (105.2 vs 79.2), same shape as the other tensor-split hosts. n-max was not swept; 2 is the 24GB-class default from rule 1 and was left ungated because one of the two cards is Blackwell (rule 2).

**Honest caveat:** the server logged `backend sampling not supported with SPLIT_MODE_TENSOR; using CPU sampler` on the MTP arm. The draft context still came up (`creating MTP draft context against the target model`) and the paired delta is +67%. A later n-max sweep, or a run once tensor-split GPU sampling lands, would be the right follow-up — this PR is the n-max 2 A/B only.


### 2× RTX A5000 24GB NVLink: split mode beats the flag, and a Q4_K_M with no MTP head
*by [@TheRiotCoder](https://github.com/TheRiotCoder), PR #57*

The table's first NVLink pair. Two RTX A5000 24GB on an NV4 bridge (~112 GB/s),
Threadripper PRO 5965WX, 128 GB RAM, headless Ubuntu 24.04, driver 595-open, CUDA 13.3,
ECC disabled on both cards. unsloth Q4_K_M, 131K context, q4_0 K/V, llama.cpp build 10454
(`4df29be4f`), `--parallel 1`, thinking off, upstream `probe.py` at `a4c3028` unchanged.
Every figure is the median of three complete passes.

#### Fix the split mode first — it is worth more than the flag

Rule 4 holds hard, and on this pair it is the larger of the two levers:

| Config | `--split-mode layer` | `--split-mode tensor` | Gain from tensor |
|---|---:|---:|---:|
| spec off | 35.1 | 51.9 | **+48%** |
| MTP `--spec-draft-n-max 2` | 54.8 | 81.1 | **+48%** |

Naive default (layer, no flag) to best config is **35.1 → 96.1, 2.7x**, and tensor split
contributes more of the first doubling than the flag does. A multi-GPU box benchmarked on
the default split reports a baseline about a third too low, which then inflates its
with-flag gain — worth stating next to rule 5's `--parallel 2` warning, since it distorts
the ratio the same way.

#### Ungated n-max sweep, tensor split

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---:|---:|---:|---:|---:|
| spec off | 51.9 | 52.1 | 51.9 | 51.6 | — |
| 1 | 73.1 | 75.8 | **65.4** | 73.1 | 0.62-0.97 |
| **2** | **80.5** | 87.2 | 64.5 | 80.5 | 0.56-0.96 |
| 3 | 79.6 | 95.6 | 57.6 | 79.6 | 0.41-0.96 |
| 4 | 79.5 | **96.5** | 56.2 | 79.5 | 0.33-0.90 |
| 5 | 71.1 | 95.6 | 50.5 | 71.1 | 0.26-0.89 |
| 6 | 72.2 | 89.4 | 45.0 | 72.2 | 0.25-0.80 |
| **8** | **96.1** | **135.0** | 52.4 | 96.1 | 0.18-0.71 |

The comparable n-max 2 arm is +56%. Overall peaks at **n-max 8, +85%**. These are 24GB
cards, so rule 1's "24GB cards peak at n-max 2" does not hold once the pair is on tensor
split — the same direction @EamonMcKiernan05's 3×3060 row points, at n-max 8 on a layer
split. Topology moves the sweet spot further than VRAM does.

Aggregate acceptance falls from 0.792 (1,522/1,921) at n-max 2 to 0.456 (1,894/4,149) at
n-max 8, so **the fastest arm is the one with the lowest acceptance** — tuning on
acceptance here would have selected n-max 1. Rule 2's "acceptance is a vanity metric" is
the most useful line in this repo.

`--spec-draft-p-min 0.60` at n-max 2 **cost 11%** (81.1 → 72.1) while lifting acceptance
to 0.69-0.90 — the same inversion reported for desktop Blackwell. Not adopted.

#### Code and prose want opposite settings

The overall column hides an inversion:

- **Python** climbs almost monotonically and reaches **135.0 at n-max 8**, 2.6x its own
  spec-off baseline.
- **Prose** peaks immediately at **n-max 1 (65.4)** and decays to 45.0 at n-max 6, which is
  **below** the 51.9 spec-off baseline. Four of the seven depths tested leave prose worse
  off than no speculation at all.

Predictable text gets its drafts accepted; prose gets them drafted, rejected and paid for.
That refines rule 3 — at a fixed 400-token cap, *what* is generated matters more than *how
much*. Coding lane: n-max 8. Mixed or prose-heavy: n-max 1-2.

#### Not every Q4_K_M carries the MTP head

The first full pass of this benchmark failed at load, on every n-max, with a **different**
non-unsloth Q4_K_M of the same model:

```
llama_init_from_model: context type MTP requested but model doesn't contain MTP layers
common_speculative_init_result: failed to create MTP context
srv    load_model: failed to create MTP context
```

| Build | Size (bytes) | `qwen35.nextn_predict_layers` | MTP loads? |
|---|---:|---|---|
| other Q4_K_M (sha256 `31629f53…`) | 18,973,870,432 | absent | **no** |
| unsloth Q4_K_M (sha256 `7e78da5d…`) | 17,106,775,008 | present | yes |

The working file is the **smaller** of the two, so size is no signal. The README's "the MTP
head already ships inside the GGUF you downloaded" holds for unsloth's builds and should not
be read as universal — the quantizer has to keep the nextn tensors. The failure mode is a
**startup crash, not a slow server**, which is easy to misread as a bad flag or a broken
build. Check for the metadata key before tuning.

#### Variance: adjacent n-max values are not separable

Three passes per arm, three `probe.py` invocations each:

| Arm | Pass A | Pass B | Pass C | Spread |
|---|---:|---:|---:|---:|
| spec off | 52.0 | 51.6 | 51.9 | 0.8% |
| n-max 2 | 81.1 | 80.5 | 82.9 | 3.0% |
| n-max 8 | 88.9 | 96.1 | 107.9 | **21%** |

Baseline arms repeat to under 1%; deep MTP arms move up to 21% between passes, because
acceptance is content-dependent and the drafts differ run to run. **n-max 2, 3 and 4 are one
plateau, not a ranking**, and a single `probe.py` invocation cannot order them — worth
adding to the method notes alongside @dcrey7's medians-of-three rule, since the deeper the
draft the more passes it takes to say anything.

#### Method warning: `probe.py` counts SSE events, which is engine-specific

I first reported a 27% *regression* from the same MTP weights under vLLM. **That was
wrong, and the cause is worth everyone's attention: `probe.py` counts streamed SSE delta
events, not tokens.** That is exactly right for llama.cpp and silently wrong for engines
that pack multiple accepted tokens into one chunk.

Measured on the same box, same prompt, comparing `probe.py`'s event count against
`usage.completion_tokens` from `stream_options: {"include_usage": true}`:

| Engine / arm | SSE delta events | real tokens | tokens per chunk | by chunks | by tokens |
|---|---:|---:|---:|---:|---:|
| llama.cpp spec off | 358 | 359 | **1.00** | 52.8 | 53.0 |
| llama.cpp MTP n-max 2 | 400 | 400 | **1.00** | 91.5 | 91.5 |
| llama.cpp MTP n-max 8 | 336 | 337 | **1.00** | 146.7 | 147.2 |
| vLLM 0.25.1 + MTP | 199 | 395 | **1.98** | 33.1 | 65.8 |

**llama.cpp emits one token per chunk in every arm, so every number in this repo is
safe** — `probe.py` is a correct instrument for the engine it was written against, spec
on or off. vLLM batches the accepted draft token together with the verified token, so
`probe.py` reads it at roughly half rate.

Corrected vLLM figures for this box, counting tokens rather than chunks, thinking off:

| Config | single stream | np=8 aggregate | np=16 aggregate |
|---|---:|---:|---:|
| AWQ-INT4, no speculation | 60.0 | 386 | 700 |
| AWQ-INT4 + MTP | **78.5 (+31%)** | **503 (+30%)** | **776 (+11%)** |

So MTP pays on vLLM too. Two knobs that looked promising did nothing: raising
`--max-num-batched-tokens` to 8192/16384 cleared vLLM's
`max_num_scheduled_tokens is set to 2048` warning but moved throughput not at all
(43.5 → 43.5 → 43.5 by the chunk metric), and neither did dropping `--max-num-seqs` to 8
or renaming the deprecated `qwen3_5_mtp` method to `mtp`. The clamp warning is a red
herring; the metric was the problem.

**The transferable lesson:** before comparing two engines with a streaming client, check
tokens-per-chunk on each. One line of `stream_options: {"include_usage": true}` settles
it, and without it a speculative-decoding gain can read as a loss.
