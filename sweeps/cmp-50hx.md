# NVIDIA CMP 50HX MOD 20GB — Qwen3.8 Heretic Ara MTP deep-dive

This is a contributor deep-dive for a mining-card variant, not a main-table
MTP OFF/ON row. The telemetry below is real production/OpenCode workload data:
it was not collected with the repository's paired `probe.py` MTP OFF/ON
method, so it should not be interpreted as a directly comparable A/B result.

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA CMP 50HX MOD 20GB, TU102 |
| Compute capability | 7.5 / SM75 |
| VRAM | 20,480 MiB |
| PCIe | Gen1 x16 |
| Host CPU | AMD FX-6300 |
| Host RAM | 24 GB |
| OS | Ubuntu 22.04.5 |
| NVIDIA driver | 595.71.05 |
| CUDA toolkit | 12.4 |
| llama.cpp | `d8a8beac22d450ebadf175a8ce7b6bf49b66db14` |
| Build | custom CMP build; Flash Attention enabled |

The CMP-specific CUDA measurements below used the DP2A workaround and
`--fmad=false`. The production/OpenCode telemetry is reported as workload
telemetry rather than as a paired benchmark arm; build details should be kept
with the deployment record when reproducing it.

## Working Qwen3.8 configuration

Model: **Qwen3.8-27B Heretic Ara IQ4_XS**.

The working serving configuration was:

```text
--spec-type draft-mtp --spec-draft-n-max 2
-np 2 --kv-unified -c 114688
-ctk q4_0 -ctv q4_0
-ctkd f16 -ctvd f16
-b 2048 -ub 2048 -fa on
--mmproj <native Qwen3.8 BF16 projector> --mmproj-offload
--reasoning on --reasoning-preserve
--cache-prompt --cache-ram 8096 --cache-realtime-ram 2048
--qos-strict --qos-realtime-slot 0
```

There are two logical QoS slots: slot 0 is realtime and slot 1 is background.
The shared unified KV capacity is 114,688 tokens. The target KV cache is Q4/Q4;
the draft KV cache is F16/F16.

## Real-world OpenCode telemetry

These are **206 completed slot-1 tasks** from a rolling production/OpenCode
workload observation. They are not paired MTP OFF/ON runs. PP is prompt
processing throughput, TG is decode throughput, and acceptance is the weighted
MTP draft acceptance observed for the bucket.

| Context | N | PP median (tok/s) | TG median (tok/s) | MTP acceptance |
|---|---:|---:|---:|---:|
| 0–2K | 99 | 201 | 33.46 | 83.6% |
| 2–4K | 20 | 404 | 31.97 | 72.9% |
| 4–8K | 15 | 464 | 31.58 | 75.4% |
| 8–16K | 18 | 495 | 30.70 | 75.3% |
| 16–32K | 15 | 483 | 29.14 | 81.1% |
| 32–64K | 1 | 434 | 22.90 | 53.1% |

Aggregate MTP counters for the observation were **91,730 drafted** and
**70,861 accepted** tokens, or **77.3% weighted acceptance**.

The 32–64K bucket contains only one task. It must not be treated as proof of a
context cliff. In a separate controlled MTP2 test near 40K, TG was about
24.98 tok/s with about 57.6% acceptance. A 90K validation request measured
352.88 tok/s PP and 25.72 tok/s TG.

For a separate controlled long-decode comparison on this machine, n-max 1
measured 31.15 tok/s TG and n-max 2 measured 33.86 tok/s TG. This is an
additional short comparison, not the 206-task production dataset.

### Cache and LCP observations

The production profile used an 8,096 MiB host prompt-cache budget and a 2,048
MiB realtime protected cache. In the six-hour journal window there were 44
prompt-cache eviction events and 25 realtime cache-admission rejections. The
server journal exposes cache pressure and graph reuse, but not an exact
per-request LCP/cached-token value; these events therefore describe cache
behavior, not a precise LCP distribution.

## CMP-specific CUDA behavior

The following measurements used **Qwen3.8-27B-Q4_K_S** on the same CMP 50HX.
They isolate the CUDA execution-path effect in a short `llama-bench` style
measurement; they are separate from the OpenCode table above.

| CUDA path | PP512 (tok/s) | TG128 (tok/s) |
|---|---:|---:|
| Stock llama.cpp / DP4A | 86.78 | 9.59 |
| DP2A workaround | 86.17 | 18.69 |
| DP2A workaround + `--fmad=false` | **116.66** | **27.98** |

Relative changes:

- stock → DP2A + `--fmad=false`: **+34.4% PP**, **+191.8% TG**;
- DP2A → DP2A + `--fmad=false`: **+35.4% PP**, **+49.7% TG**.

PTX inspection matched the intended paths: the stock path contained `dp4a`
and did not contain `dp2a`; the patched path did not contain `dp4a` and did
contain `dp2a`.

The DP2A workaround was discussed in
[ggml-org/llama.cpp issue #24616](https://github.com/ggml-org/llama.cpp/issues/24616)
and implemented in
[PR #25834](https://github.com/ggml-org/llama.cpp/pull/25834). At the time of
writing, PR #25834 is an upstream pull request, not a merged change; this
report does not claim that it has landed in upstream master.

These results show an unusually poor stock DP4A/FMA execution path on this CMP
card. They do **not** prove that Tensor Cores are physically disabled, nor do
they establish that Tensor Cores are the cause of the observed behavior.

## Findings

1. Qwen3.8-27B is practically usable for agentic coding on a 20GB CMP 50HX
   with this configuration.
2. In the ordinary 4–32K OpenCode range, observed PP was about 460–500 tok/s,
   TG about 29–32 tok/s, and MTP acceptance about 75–81%.
3. TG fell from 33.46 tok/s at 0–2K to 29.14 tok/s at 16–32K, approximately
   **12.9%**. The 8–16K to 16–32K decline was only about **5.1%**.
4. MTP2 had 77.3% weighted acceptance across the observed workload, consistent
   with the embedded predictor working normally on this card.
5. The single 32–64K sample is insufficient to establish degradation at that
   range.
6. The main CMP-specific anomaly is the very slow stock DP4A/FMA path. The
   DP2A workaround plus `--fmad=false` produced the largest measured decode
   improvement in the separate CUDA microbenchmark.
7. The OpenCode numbers are real-world telemetry, not the repository's
   standard paired `probe.py` benchmark, and should remain a deep-dive rather
   than a new main-table A/B row.
