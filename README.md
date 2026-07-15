# taraference

CUDA multi-turn inference for **Qwen2.5-3B-Instruct Q4_K_M** on RTX 3050 Ti (4GB).

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --prompt "Hello" -n 64
```

Weights stay **quantized on GPU** (fused Q4/Q6 GEMV). `/quit` `/reset`.

**Physics:** ~1.8 GiB weights ÷ ~192 GB/s ≈ **~100 tok/s** hard ceiling at 100% bandwidth. **750 tok/s single-stream on 3B is not reachable** on this GPU without reading fewer bytes/token than a dense 3B has.
