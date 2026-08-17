# CMP 170HX — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP. The row and footnote live in the main [community table](../README.md#community-numbers).

### NVIDIA CMP 170HX 64GB (unlocked HBM2e): stock `probe.py` A/B
*by [@shiwuxiu](https://github.com/shiwuxiu)*

Headless Ubuntu 24.04 VM, llama.cpp `f275595dd` / CUDA 13.3 / driver 610.43.02, 200W power cap, PCIe Gen1 x4. The card is a cut-down GA100 (70 SMs) with Hynix HBM2e unlocked from the stock 8GB CMP firmware to 64GB. Both arms used unsloth `Qwen3.8-27B-Q4_K_M.gguf` (15.93 GiB), `-c 131072 -ngl 999 -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1`. Only the flag arm added `--spec-type draft-mtp --spec-draft-n-max 2`.

Method: unchanged `probe.py` at `70a699e`, warmup discarded by the script itself, three runs x three prompts, `enable_thinking: False`.

| Arm | Overall median | P1 code (py) | P2 prose (mmap) | P3 code (bash) | VRAM |
|---|---:|---:|---:|---:|---:|
| spec off | 33.0 | 33.1 | 33.0 | 32.7 | 18,864 MiB |
| n-max 2 | 46.7 | 51.4 | 38.3 | 46.7 | 20,152 MiB |

Overall mean 32.9 -> 45.5 tok/s (**+41.5%** median). Same shape as other Ampere/Ada rows: code benefits more than prose. Draft acceptance from llama-server `slot print_timing` on the nine scored requests was 0.53-0.94 (prose 0.53-0.64, bash ~0.74-0.79, Python 0.91-0.94). Warmup was 0.80 and is not in the table range.

This is the comparable n-max 2 row. No n-max 3/4 or p-min sweep in this PR. The 64GB unlock has spare VRAM, so deeper n-max is a later measurement, not a claim.

This row is llama.cpp only, as the repository asks. The same host also runs a vLLM FP8+MTP service that is faster; that number is out of scope here and is not mixed into the A/B.
