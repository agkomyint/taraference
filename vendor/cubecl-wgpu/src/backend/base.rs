use std::{borrow::Cow, sync::Arc};

use cubecl_core::{ExecutionMode, Feature, WgpuCompilationOptions, prelude::CompiledKernel};
use cubecl_runtime::DeviceProperties;
use wgpu::{
    Adapter, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderStages,
};

use crate::{AutoCompiler, AutoRepresentation, WgpuServer};

use super::wgsl;

#[cfg(feature = "spirv")]
use super::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use super::metal;
#[cfg(all(feature = "msl", target_os = "macos"))]
use cubecl_cpp::metal as cpp_metal;

impl WgpuServer {
    pub fn create_pipeline(
        &mut self,
        kernel: CompiledKernel<AutoCompiler>,
        mode: ExecutionMode,
        cache_path: Option<&std::path::Path>,
        key: &str,
    ) -> Arc<ComputePipeline> {
        let _ = (cache_path, key);
        let module = match &kernel.repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => {
                let spirv = repr.assemble();
                unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough::SpirV(
                            wgpu::ShaderModuleDescriptorSpirV {
                                label: Some(&kernel.entrypoint_name),
                                source: Cow::Borrowed(&spirv),
                            },
                        ),
                    )
                }
            }
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentation::Msl(repr)) => {
                let source = &kernel.source;
                unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough::Msl(
                            wgpu::ShaderModuleDescriptorMsl {
                                entry_point: kernel.entrypoint_name.clone(),
                                label: Some(&kernel.entrypoint_name),
                                source: Cow::Borrowed(source),
                                num_workgroups: (repr.cube_dim.x, repr.cube_dim.y, repr.cube_dim.z),
                            },
                        ),
                    )
                }
            }
            _ => {
                let source = &kernel.source;

                let checks = wgpu::ShaderRuntimeChecks {
                    // Cube does not need wgpu bounds checks - OOB behaviour is instead
                    // checked by cube (if enabled).
                    // This is because the WebGPU specification only makes loose guarantees that Cube can't rely on.
                    bounds_checks: false,
                    // Loop bounds are only checked in checked mode.
                    force_loop_bounding: mode == ExecutionMode::Checked,
                };

                // SAFETY: Cube guarantees OOB safety when launching in checked mode. Launching in unchecked mode
                // is only available through the use of unsafe code.
                unsafe {
                    self.device.create_shader_module_trusted(
                        ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                        },
                        checks,
                    )
                }
            }
        };
        let bindings = match &kernel.repr {
            Some(AutoRepresentation::Wgsl(repr)) => Some(wgsl::bindings(repr)),
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentation::Msl(repr)) => Some(cpp_metal::bindings(repr)),
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => Some(vulkan::bindings(repr)),
            _ => None,
        };
        #[cfg(feature = "spirv")]
        if let (Some(path), Some(AutoRepresentation::SpirV(repr)), Some(bindings)) =
            (cache_path, &kernel.repr, &bindings)
        {
            store_kernel(
                path,
                key,
                &kernel.entrypoint_name,
                &repr.assemble(),
                bindings,
            );
        }
        self.build_pipeline(&module, &kernel.entrypoint_name, bindings)
    }

    /// Builds a pipeline straight from a kernel stored by an earlier process.
    #[cfg(feature = "spirv")]
    pub fn cached_pipeline(
        &mut self,
        path: &std::path::Path,
        key: &str,
    ) -> Option<Arc<ComputePipeline>> {
        let (entrypoint, spirv, bindings) = load_kernel(path, key)?;
        let module = unsafe {
            self.device.create_shader_module_passthrough(
                wgpu::ShaderModuleDescriptorPassthrough::SpirV(wgpu::ShaderModuleDescriptorSpirV {
                    label: Some(&entrypoint),
                    source: Cow::Borrowed(&spirv),
                }),
            )
        };
        Some(self.build_pipeline(&module, &entrypoint, Some(bindings)))
    }

    fn build_pipeline(
        &mut self,
        module: &wgpu::ShaderModule,
        entrypoint: &str,
        bindings: Option<Vec<(usize, cubecl_core::compute::Visibility)>>,
    ) -> Arc<ComputePipeline> {
        let layout = bindings.map(|bindings| {
            let bindings = bindings
                .into_iter()
                .map(|(i, _visibility)| BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        #[cfg(not(exclusive_memory_only))]
                        ty: BufferBindingType::Storage { read_only: false },
                        #[cfg(exclusive_memory_only)]
                        ty: BufferBindingType::Storage {
                            read_only: matches!(
                                _visibility,
                                cubecl_core::compute::Visibility::Read
                            ),
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>();
            let layout = self
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bindings,
                });
            self.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                })
        });

        Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entrypoint),
                    layout: layout.as_ref(),
                    module,
                    entry_point: Some(entrypoint),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: pipeline_cache(&self.device),
                }),
        )
    }
}

#[cfg(all(not(feature = "spirv"), not(feature = "msl")))]
pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    wgsl::request_device(adapter).await
}

#[cfg(feature = "spirv")]
pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    if is_vulkan(adapter) {
        vulkan::request_vulkan_device(adapter).await
    } else {
        wgsl::request_device(adapter).await
    }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    use super::metal;

    if is_metal(adapter) {
        metal::request_metal_device(adapter).await
    } else {
        panic!("metal device not found!");
    }
}

#[cfg(all(not(feature = "spirv"), not(feature = "msl")))]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties<Feature>,
    comp_options: &mut WgpuCompilationOptions,
) {
    wgsl::register_wgsl_features(adapter, props, comp_options);
}

#[cfg(feature = "spirv")]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties<Feature>,
    comp_options: &mut WgpuCompilationOptions,
) {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options);
    } else {
        wgsl::register_wgsl_features(adapter, props, comp_options);
    }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties<Feature>,
    comp_options: &mut WgpuCompilationOptions,
) {
    if is_metal(adapter) {
        metal::register_metal_features(adapter, props, comp_options);
    } else {
        panic!("metal device not found!");
    }
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan, _, _>(|adapter| adapter.is_some()) }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
fn is_metal(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Metal, _, _>(|adapter| adapter.is_some()) }
}

/// Persistent driver pipeline cache, stored as `pipelines.bin` in the kernel cache directory.
struct DiskPipelineCache {
    cache: wgpu::PipelineCache,
    path: std::path::PathBuf,
    dirty: std::sync::atomic::AtomicBool,
    saved: std::sync::Mutex<std::time::Instant>,
}

static PIPELINE_CACHE: std::sync::OnceLock<Option<DiskPipelineCache>> = std::sync::OnceLock::new();

fn disk_pipeline_cache(device: &Device) -> Option<&'static DiskPipelineCache> {
    PIPELINE_CACHE
        .get_or_init(|| {
            let path = cache_dir()?.join("pipelines.bin");
            if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
                return None;
            }
            let data = std::fs::read(&path).ok();
            // SAFETY: the data is only ever written by `save_pipeline_cache` below, and
            // `fallback` makes the driver start empty if it rejects the contents.
            let cache = unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("cubecl"),
                    data: data.as_deref(),
                    fallback: true,
                })
            };
            Some(DiskPipelineCache {
                cache,
                path,
                dirty: std::sync::atomic::AtomicBool::new(false),
                saved: std::sync::Mutex::new(std::time::Instant::now()),
            })
        })
        .as_ref()
}

fn pipeline_cache(device: &Device) -> Option<&'static wgpu::PipelineCache> {
    let disk = disk_pipeline_cache(device)?;
    disk.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    Some(&disk.cache)
}

/// Writes the cache to disk if pipelines were created since the last save.
pub(crate) fn save_pipeline_cache() {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(Some(disk)) = PIPELINE_CACHE.get() else {
        return;
    };
    if !disk.dirty.load(Relaxed) {
        return;
    }
    let mut saved = disk.saved.lock().unwrap();
    if saved.elapsed() < std::time::Duration::from_millis(500) {
        return;
    }
    disk.dirty.store(false, Relaxed);
    *saved = std::time::Instant::now();
    if let Some(data) = disk.cache.get_data() {
        let temp = disk.path.with_extension("tmp");
        if std::fs::write(&temp, data).is_ok() {
            let _ = std::fs::rename(&temp, &disk.path);
        }
    }
}

/// Directory named by `CUBECL_KERNEL_CACHE`, created on first use.
fn cache_dir() -> Option<&'static std::path::PathBuf> {
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::path::PathBuf::from(std::env::var_os("CUBECL_KERNEL_CACHE")?);
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    })
    .as_ref()
}

/// Cache file and full key for a kernel. Kernel ids embed `TypeId`s, which are only
/// stable for one build, so the executable's size and timestamp are part of the key.
pub(crate) fn kernel_cache_entry(
    id: &impl core::fmt::Debug,
) -> Option<(std::path::PathBuf, String)> {
    use std::hash::{Hash, Hasher};
    static BUILD: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let build = BUILD
        .get_or_init(|| {
            let meta = std::fs::metadata(std::env::current_exe().ok()?).ok()?;
            Some(format!("{}:{:?}", meta.len(), meta.modified().ok()?))
        })
        .as_ref()?;
    let key = format!("{build}|{id:?}");
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    Some((
        cache_dir()?.join(format!("{:016x}.spv", hasher.finish())),
        key,
    ))
}

#[cfg(feature = "spirv")]
fn store_kernel(
    path: &std::path::Path,
    key: &str,
    entrypoint: &str,
    spirv: &[u32],
    bindings: &[(usize, cubecl_core::compute::Visibility)],
) {
    let mut out = Vec::with_capacity(spirv.len() * 4 + key.len() + 64);
    for text in [key, entrypoint] {
        out.extend((text.len() as u32).to_le_bytes());
        out.extend(text.as_bytes());
    }
    out.extend((bindings.len() as u32).to_le_bytes());
    for (_, visibility) in bindings {
        out.push(matches!(visibility, cubecl_core::compute::Visibility::Read) as u8);
    }
    out.extend(spirv.iter().flat_map(|word| word.to_le_bytes()));
    let temp = path.with_extension("tmp");
    if std::fs::write(&temp, out).is_ok() {
        let _ = std::fs::rename(&temp, path);
    }
}

#[cfg(feature = "spirv")]
#[allow(clippy::type_complexity)]
fn load_kernel(
    path: &std::path::Path,
    key: &str,
) -> Option<(
    String,
    Vec<u32>,
    Vec<(usize, cubecl_core::compute::Visibility)>,
)> {
    use cubecl_core::compute::Visibility;
    fn take<'a>(data: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
        let (part, rest) = data.split_at_checked(len)?;
        *data = rest;
        Some(part)
    }
    fn number(data: &mut &[u8]) -> Option<usize> {
        Some(u32::from_le_bytes(take(data, 4)?.try_into().ok()?) as usize)
    }
    fn text(data: &mut &[u8]) -> Option<String> {
        let len = number(data)?;
        String::from_utf8(take(data, len)?.to_vec()).ok()
    }
    let bytes = std::fs::read(path).ok()?;
    let mut data = bytes.as_slice();
    if text(&mut data)? != key {
        return None;
    }
    let entrypoint = text(&mut data)?;
    let count = number(&mut data)?;
    let bindings = take(&mut data, count)?
        .iter()
        .enumerate()
        .map(|(index, &read)| {
            let visibility = if read == 1 {
                Visibility::Read
            } else {
                Visibility::ReadWrite
            };
            (index, visibility)
        })
        .collect();
    let rest = data;
    if rest.is_empty() || rest.len() % 4 != 0 {
        return None;
    }
    let spirv = rest
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    Some((entrypoint, spirv, bindings))
}
