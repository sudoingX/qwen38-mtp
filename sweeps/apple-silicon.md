# Apple Silicon — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP on Apple Silicon / Metal. Add yours via a PR to this file, row and footnote go in the main [community table](../README.md#community-numbers).

### Apple M4 24GB (Metal): the flag is a code-for-prose trade, not a speedup
*by [@sternryan](https://github.com/sternryan), PR #TBD*

First Apple row in the table, and the first one that does not move the overall number. M4 (base, 10-core GPU, ~120 GB/s), 24GB unified memory, macOS 15, llama.cpp b10450 (`ece963f41`, Homebrew bottle). unsloth UD-Q3_K_XL, 32K context, q4_0 KV, `--parallel 1`, `-b 512 -ub 512`, thinking off. Unmodified `probe.py` at `a4c3028da1`.

**Two independent full passes per arm.** `probe.py` already takes the median of three runs per prompt, so each pass satisfies the three-run floor on its own; the second pass exists because a single pass crowns the wrong sign here, the same lesson the RX 9070 section reports. Six samples per prompt per arm in total:

| arm | Run A | Run B | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|---|
| spec off | 5.8 | 5.8 | 5.7 / 5.8 | 5.8 / 6.0 | 5.8 / 5.8 | — |
| n-max 2 | 5.6 | 5.9 | **6.3 / 6.2** | **4.4 / 4.7** | 5.6 / 5.9 | 0.48-0.95 |

Run A is -3.4%, Run B is +1.7%. **The overall number is parity within noise** (the run-to-run band here is about ±0.2 tok/s on ~6, so anything under ~4% is unresolvable). Reporting it as a gain or a loss would both be overclaiming.

The per-prompt split is the part that reproduces:

- **code +9 to +10%** (5.7→6.3, 5.8→6.2)
- **prose -22 to -24%** (5.8→4.4, 6.0→4.7)

Same shape every other sweep in the repo reports — code climbs, prose falls, acceptance decays — except here the two halves cancel instead of netting positive. Draft acceptance is healthy (0.48-0.95, comparable to the NVIDIA rows), so this is not a drafting-quality problem.

#### Context length makes it worse

Same machine, same build, same quant, `--spec-draft-n-max 2`, only context and batch changing:

| context | `-b` | baseline | n-max 2 | delta |
|---|---|---|---|---|
| 8K | 512 | 6.0 | 5.9 | -1.7% |
| 32K | 512 | 5.8 | 5.6 / 5.9 | parity |
| 32K | 2048 | 5.9 | 5.4 | -8.5% |

Directionally what a non-amortizing verify predicts: the attention term grows with KV, and it is paid per verify row instead of once. Anyone testing this on an Apple machine at 131K+ should expect worse, not better — I could not measure there, 24GB will not hold it.

#### Gotcha: the spec arm OOMs at the default batch size

On 24GB, `--spec-type draft-mtp` dies mid-generation at the default `-b 2048`:

```
ggml_metal_synchronize: error: command buffer 1 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

Load-time allocation is only 13.9 GB against a 21.5 GB `iogpu.wired_limit_mb`, so nothing warns you — it is the transient prompt-batch allocation that overflows, and it takes the server down with a `GGML_ASSERT` rather than an error response. `-b 512 -ub 512` fixes it. Raise `iogpu.wired_limit_mb` (default 18000 on a 24GB machine, too low for a 12.5 GiB model plus spec contexts) before anything else.

Metal allocation for the row's config: **13.2 GB baseline, 13.9 GB with spec** (model 12609 MiB, KV 576, recurrent state 150→449, compute 164→294).

#### Practical verdict

If you are on a 24GB Apple machine, this flag is not a speedup. It is roughly free on code and costs about a quarter of your throughput on prose. Leave it off for mixed use; turn it on if your session is pure code generation and you have measured your own split. Larger Macs with the memory to run Q8_0 or BF16 weights are the interesting untested case — the batch scaling in the sections below says that is where Apple would start winning, and I do not have the hardware to check.


### The same model on two backends: a matched CUDA control
*by [@sternryan](https://github.com/sternryan), PR #TBD*

Why the row above nets to zero, and how to check that it is real.

**First, what this is not.** Qwen3.8-27B is dense, and Apple Silicon is a poor fit for dense models by construction — lots of unified memory, modest bandwidth, so every token pays for all the weights. That is well understood and is not the finding here. The finding is that **the standard remedy for being bandwidth-bound is also unavailable**: speculative decoding is exactly what you reach for when batch-1 decode is starved, and on this backend it cannot pay off, for a reason that has nothing to do with bandwidth. An Apple user is not just slow, they are slow *and* out of the usual escape hatch. (MoE models, where a fraction of the weights activate per token, are the case Apple is actually built for; nothing here speaks to those.)

Same GGUF file, same flags, same llama.cpp commit (`ece963f41`), same `probe.py`, same operator, two independent full passes each (three runs per prompt inside each pass). Only the backend differs:

| | baseline | n-max 2 | delta | P1 code | P2 prose | P3 bash |
|---|---|---|---|---|---|---|
| RTX 3090, run A | 43.8 | 73.9 | **+69%** | 43.8→80.8 | 43.9→57.3 | 43.4→73.9 |
| RTX 3090, run B | 43.2 | 78.5 | **+82%** | 43.2→80.7 | 43.4→58.2 | 42.5→78.5 |
| Apple M4, run A | 5.8 | 5.6 | -3.4% | 5.7→6.3 | 5.8→**4.4** | 5.8→5.6 |
| Apple M4, run B | 5.8 | 5.9 | +1.7% | 5.8→6.2 | 6.0→**4.7** | 5.8→5.9 |

On CUDA every prompt type gains, prose included (+30-34%). On Metal prose inverts to -22-24% while code still gains. Acceptance is comparable on both (0.46-0.93 CUDA, 0.48-0.95 Metal), so the drafts are equally good and only the verify differs.

The 3090 baseline is deliberately checkable: 43.2-43.8 against [@hauntedhost](https://github.com/hauntedhost)'s 41.3 on the same b10450 and [@dcrey7](https://github.com/dcrey7)'s 43.9 overclocked. If this rig measures 3090s the way the rest of the table does, the Apple number is not a harness artifact — and anyone with a 3090 can check that half directly.

### Metal small-batch decode does not amortize
*by [@sternryan](https://github.com/sternryan), PR #TBD*

Where the difference in the control above comes from, and the part that may generalize past Apple.

Speculative decoding assumes verifying `n+1` tokens costs about one forward pass — the weights are read once and serve every row. `llama-batched-bench` measures that directly, and on Metal it does not hold. Same command both machines (`-npp 128 -ntg 64 -npl 1,2,4,8 -fa 1 -ngl 999`), decode throughput at batch 8 against batch 1, where 8x would be perfect amortization:

| model / format | Metal (M4, b10450) | CUDA (RTX 3090, master) |
|---|---|---|
| Qwen3-0.6B BF16 | **4.99x** | 5.89x |
| Qwen3-0.6B Q4_0 | **1.66x** | 4.18x |
| Qwen3-0.6B Q4_K_M | **1.74x** | 3.85x |
| Qwen3.8-27B (Q3_K_XL / Q4_0) | **1.22x** | 3.34x |

On the 27B, eight rows cost 6.54x one row on Metal versus 2.39x on CUDA. **Quantized decode on Metal barely amortizes across a batch; BF16 does.** That is the whole result: verification costs close to `n+1` full passes, so the drafts are paid for and nothing comes back. It also explains why the gain is prompt-shaped — code accepts longer runs and needs fewer verify rounds per token, prose needs more.

Both sides are genuinely bandwidth-bound at batch 1 (Metal 12.5 GiB / 176 ms = 71 GB/s of ~120 peak; CUDA 15.0 GiB / 23.6 ms = 634 GB/s of 936), so this is not one backend being launch-overhead-dominated.

Two code paths are worth a look for anyone who wants to chase it, both still in current master (`ggml/src/ggml-metal/ggml-metal-ops.cpp`): `ne11_mm_min = 8` gates the batched matrix-matrix kernel to `ne11 > 8`, and a spec verify batch is `n_draft + 1` rows — 3 at n-max 2, 5 at n-max 4 — so it never qualifies. The small-batch mat-vec path admits F16/BF16/Q4_0/Q8_0/IQ4_NL at `ne11 >= 2` but K-quants only at `ne11 >= 4`. **Flagging these as leads, not as a diagnosis** — see the disclosed nulls below.

#### Disclosed nulls

- **A quant-family effect that did not survive.** On an older self-built binary (May-era master + PR #23114) the same model measured -11.7% in Q3_K_XL and +7.5% in Q4_0, which looked like the K-quant `ne11 >= 4` gate showing up end to end. On stock b10450 **both quants land at the same -1.7%**, and Q4_0's batch-8 amortization advantage disappears too (1.66x versus Q4_K_M's 1.74x). Not reproducible, so not a rule. Reporting it because the gate is real in the source and someone with a bigger Mac may be able to separate it properly.
- **llama.cpp PR #23114** (`metal: reuse K/V in flash-attn vec for spec-decode`, closed unmerged) applied and toggled via its own `GGML_METAL_FA_DISABLE_Q2`: 5.2 → 5.4 tok/s at n-max 2, **+3.8%**, matching that PR's own end-to-end MTP table (1.00-1.10x). Right subsystem, wrong term — flash attention is not where the cost is.
- **n-max 4** was measured and is worse on both quants tested (-21% on Q4_0), consistent with a 5-row verify still costing ~2.2x on Metal. No sweep past 4; there is no reason to expect a peak out there when the batch scaling is this flat.

#### What this might mean off Apple

If it generalizes, the rule is **check that your backend amortizes a small decode batch before tuning spec flags at all** — where batch 2-8 decode costs close to linear, no n-max or p-min value can recover it, and acceptance will look healthy the whole time you are losing. The CUDA column above is the counter-example that makes the flag pay everywhere else in this table. I have only two backends and one Apple device, so this is offered as a hypothesis for the rules list, not a finding.

### Open threads on Apple Silicon
*by [@sternryan](https://github.com/sternryan), PR #TBD*

Posting these so nobody duplicates the work, and because two of them need hardware I do not have. Happy to hand any of it off.

1. **MLX versus llama.cpp on the same Mac, same model.** The [mlx.fast](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge) challenge has contributors running this model at 77 tok/s with MTP on Apple hardware, which implies MLX's verify path amortizes where llama.cpp's does not. If MLX amortizes on the *same machine* that produced the row above, then this is a llama.cpp kernel gap rather than an Apple hardware limit — which is a very different conclusion and the single most useful thing a Mac owner could be told.
2. **The quant ladder on Metal.** On a bandwidth-bound box, bytes-per-token is the lever that actually moves, and [@dcrey7](https://github.com/dcrey7)'s 3090 rows already show UD-Q2_K_XL reading 38% fewer bytes and gaining accordingly. Measuring tok/s against bytes-per-token across Q2/Q3/Q4 on Apple should give a steeper curve than CUDA and a straight answer on what to actually run.
3. **The untuned small-batch kernel constants.** `ggml-metal-ops.cpp` sets `nsg = 2` for the small-batch mat-vec path under a comment that says, verbatim, "I still don't know why we should not always use the maximum available threads... my current hypothesis is that the work grid is not evenly divisible for different nsg values... need to confirm this." An unvalidated constant with a stated hypothesis and a harness sitting right here. This one could lift baseline decode for every Apple user regardless of spec flags.
4. **`ne11_mm_min` and the width of the multi-row path.** The change that would make spec decoding work on Metal rather than merely explain why it does not. Real shader work, flagged for anyone upstream who wants it.
5. **A large-memory Mac at Q8_0 or BF16.** The batch-scaling table says this is where Apple starts winning, and 24GB cannot hold those weights for a 27B. If someone with an M-series Max or Ultra runs the paired A/B at Q8_0, that is the row that would flip this section's conclusion — I would rather be wrong here than right.
