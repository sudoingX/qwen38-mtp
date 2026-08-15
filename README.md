# Qwen3.8-27B MTP: the flag was free the whole time

One llama.cpp flag unlocks +33-39% decode speed for Qwen3.8-27B on consumer GPUs. No new files, no conversion, no custom build. The MTP head already ships inside the GGUF you downloaded on launch night.

Measured hours after the Aug 14 2026 release, published so every 24GB card owner gets the speed on day one.

## The numbers

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
| RTX A6000 48GB (Ada) | 26.7 | 52.5 | 2 | 0.54-0.98 | [@lingster](https://github.com/lingster) |
| RX 7900 XTX 24GB | 30.7 | 43.9 | 2 | 0.60-0.95 | [@Jqianggu](https://x.com/Jqianggu) |
| 2x RX 9070 16GB (Vulkan) | 22.1 | 41.6 | 2 | 0.73 | [@tomertec](https://github.com/tomertec) |

\* A6000 row: unsloth Q8_K_XL, 256K context, q8_0 KV cache — 40.0 GB VRAM baseline, 41.4 GB with spec (rows above: Q4_K_M, 131K, q4_0 KV).
\* RX 7900 XTX row: unsloth Q4_K_M, 131K context, q4_0 KV cache — 18.9 GB VRAM baseline, 19.7 GB with spec.
\* RX 9070 row: two 16GB cards on one 31.84 GiB pool, Vulkan build b10426, unsloth UD-Q4_K_XL, **262K context**, q8_0 KV cache — 28.0 GiB across the pool with spec. Method differs from the rows above and is spelled out under the sweep below.

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

## License

Apache-2.0. The numbers and verdicts are real, the conclusions are mine.
