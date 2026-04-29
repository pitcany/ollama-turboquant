# TurboQuant Context Length Benchmark

Date: Wed Apr 29 02:35:00 AM PDT 2026
GPU: NVIDIA GeForce RTX 4090, 24564 MiB

| Model | KV Cache | Context | Prompt Tokens | Prompt Speed | Gen Speed | KV Size | Answer |
|-------|----------|---------|---------------|-------------|-----------|---------|--------|
| qwen2.5:7b | f16 | 4096 | 2000 | 11583.8 t/s | 240.5 t/s | 224.0 MiB | `The` |
| qwen2.5:7b | turbo4 | 4096 | 2000 | 253975.9 t/s | 326.2 t/s | ? | `The` |
| qwen2.5:7b | f16 | 32768 | 16000 | 11034.7 t/s | 222.7 t/s | ? | `The.` |
| qwen2.5:7b | turbo4 | 32768 | 16000 | 11424.2 t/s | 221.2 t/s | ? | `The.` |
| qwen2.5:7b | f16 | 65536 | 32000 | 11403.0 t/s | 280.2 t/s | ? | `ベル` |
| qwen2.5:7b | turbo4 | 65536 | 32000 | 11327.0 t/s | 173.7 t/s | ? | `The very first word of the filler text is "You".` |
| qwen2.5:7b | f16 | 131072 | 64000 | **FAIL** | - | - |
| qwen2.5:7b | turbo4 | 131072 | 64000 | **FAIL** | - | - |
| qwen3.6:27b | f16 | 4096 | 2000 | 2199.7 t/s | 43.9 t/s | ? | `` |
| qwen3.6:27b | turbo4 | 4096 | 2000 | 3037.3 t/s | 44.5 t/s | ? | `` |
| qwen3.6:27b | f16 | 32768 | 16000 | 2268.3 t/s | 12.5 t/s | ? | `` |
| qwen3.6:27b | turbo4 | 32768 | 16000 | 2264.4 t/s | 12.7 t/s | ? | `` |
| qwen3.6:27b | f16 | 65536 | 32000 | 2268.1 t/s | 12.7 t/s | ? | `` |
| qwen3.6:27b | turbo4 | 65536 | 32000 | 2264.9 t/s | 12.7 t/s | ? | `` |
| qwen3.6:27b | f16 | 98304 | 48000 | **FAIL** | - | - |
| qwen3.6:27b | turbo4 | 98304 | 48000 | **FAIL** | - | - |

