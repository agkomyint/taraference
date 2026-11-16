# Local changes to cubecl-wgpu 0.6.0

This is cubecl-wgpu 0.6.0 from crates.io (MIT OR Apache-2.0) with one addition,
used through `[patch.crates-io]` in the workspace manifest.

**Persistent kernel cache.** When the environment variable `CUBECL_KERNEL_CACHE`
names a directory, every compiled SPIR-V kernel is stored there and later
processes build their pipelines from the stored SPIR-V instead of recompiling.
The driver's pipeline cache is kept in the same directory (`pipelines.bin`).
Without the variable the crate behaves exactly like upstream.

- Kernel ids contain `TypeId`s, which are only stable within one build, so the
  cache key includes the executable's size and modification time, and the
  device's feature set. A rebuilt binary starts a fresh set of entries; old
  entries are never deleted automatically.
- Only the SPIR-V (Vulkan) compiler is cached. WGSL kernels compile as before.
- Changed files: `src/backend/base.rs`, `src/compute/server.rs`.

Drop this directory and the patch entry once upstream has an equivalent cache.
