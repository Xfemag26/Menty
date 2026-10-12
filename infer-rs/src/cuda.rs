//! Minimal CUDA Driver API backend for Rust-native inference.
//!
//! Uses the installed NVIDIA driver and a small embedded PTX matrix-vector
//! kernel: no nvcc, Python, CUDA toolkit or external inference server required.
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;
pub type DevicePtr = u64;
type Ctx = *mut c_void;
type Module = *mut c_void;
type Function = *mut c_void;
type ResultCode = c_int;

#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub total_bytes: usize,
    pub free_bytes: usize,
    pub compute_major: c_int,
    pub compute_minor: c_int,
}

#[derive(Clone, Copy, Default)]
pub struct CudaLayerMatrices {
    pub weights: [DevicePtr; 7], // q, k, v, o, gate, up, down
}

// The context is explicitly activated before every driver call, so the owner may
// safely be moved to the inference worker thread while retaining unique access.
unsafe impl Send for DriverApi {}
unsafe impl Send for CudaBackend {}

struct DriverApi {
    library: *mut c_void,
    init: unsafe extern "C" fn(c_uint) -> ResultCode,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> ResultCode,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> ResultCode,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> ResultCode,
    device_get_compute_capability:
        unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> ResultCode,
    device_total_mem: unsafe extern "C" fn(*mut usize, c_int) -> ResultCode,
    ctx_create: unsafe extern "C" fn(*mut Ctx, c_uint, c_int) -> ResultCode,
    ctx_set_current: unsafe extern "C" fn(Ctx) -> ResultCode,
    ctx_destroy: unsafe extern "C" fn(Ctx) -> ResultCode,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> ResultCode,
    mem_alloc: unsafe extern "C" fn(*mut DevicePtr, usize) -> ResultCode,
    mem_free: unsafe extern "C" fn(DevicePtr) -> ResultCode,
    memcpy_htod: unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> ResultCode,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> ResultCode,
    module_load_data_ex: unsafe extern "C" fn(
        *mut Module,
        *const c_void,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> ResultCode,
    module_get_function: unsafe extern "C" fn(*mut Function, Module, *const c_char) -> ResultCode,
    module_unload: unsafe extern "C" fn(Module) -> ResultCode,
    launch_kernel: unsafe extern "C" fn(
        Function,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> ResultCode,
    ctx_synchronize: unsafe extern "C" fn() -> ResultCode,
}

impl DriverApi {
    unsafe fn symbol<T: Copy>(library: *mut c_void, name: &'static [u8]) -> Result<T, String> {
        let symbol = dlsym(library, name.as_ptr().cast());
        if symbol.is_null() {
            return Err(format!(
                "CUDA driver symbol {} is unavailable",
                String::from_utf8_lossy(&name[..name.len() - 1])
            ));
        }
        Ok(std::mem::transmute_copy(&symbol))
    }

    fn load() -> Result<Self, String> {
        unsafe {
            let library = dlopen(b"libcuda.so.1\0".as_ptr().cast(), RTLD_NOW);
            if library.is_null() {
                return Err("NVIDIA CUDA driver library libcuda.so.1 was not found".into());
            }
            let result = (|| {
                Ok(Self {
                    library,
                    init: Self::symbol(library, b"cuInit\0")?,
                    device_get_count: Self::symbol(library, b"cuDeviceGetCount\0")?,
                    device_get: Self::symbol(library, b"cuDeviceGet\0")?,
                    device_get_name: Self::symbol(library, b"cuDeviceGetName\0")?,
                    device_get_compute_capability: Self::symbol(
                        library,
                        b"cuDeviceComputeCapability\0",
                    )?,
                    device_total_mem: Self::symbol(library, b"cuDeviceTotalMem_v2\0")?,
                    ctx_create: Self::symbol(library, b"cuCtxCreate_v2\0")?,
                    ctx_set_current: Self::symbol(library, b"cuCtxSetCurrent\0")?,
                    ctx_destroy: Self::symbol(library, b"cuCtxDestroy_v2\0")?,
                    mem_get_info: Self::symbol(library, b"cuMemGetInfo_v2\0")?,
                    mem_alloc: Self::symbol(library, b"cuMemAlloc_v2\0")?,
                    mem_free: Self::symbol(library, b"cuMemFree_v2\0")?,
                    memcpy_htod: Self::symbol(library, b"cuMemcpyHtoD_v2\0")?,
                    memcpy_dtoh: Self::symbol(library, b"cuMemcpyDtoH_v2\0")?,
                    module_load_data_ex: Self::symbol(library, b"cuModuleLoadDataEx\0")?,
                    module_get_function: Self::symbol(library, b"cuModuleGetFunction\0")?,
                    module_unload: Self::symbol(library, b"cuModuleUnload\0")?,
                    launch_kernel: Self::symbol(library, b"cuLaunchKernel\0")?,
                    ctx_synchronize: Self::symbol(library, b"cuCtxSynchronize\0")?,
                })
            })();
            if result.is_err() {
                dlclose(library);
            }
            result
        }
    }
}

impl Drop for DriverApi {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.library);
        }
    }
}

/// One CUDA context and its resident layer weights. Operations set this context
/// current on the calling thread, allowing model load and inference on different
/// UI/runtime threads without transferring ownership of the CUDA context.
pub struct CudaBackend {
    api: DriverApi,
    context: Ctx,
    module: Module,
    matvec: Function,
    allocations: Vec<DevicePtr>,
    pub layers: Vec<CudaLayerMatrices>,
    input_buffer: DevicePtr,
    output_buffer: DevicePtr,
    max_input: usize,
    max_output: usize,
    disabled_reason: Option<String>,
    pub device_info: CudaDeviceInfo,
}

impl CudaBackend {
    pub fn new(max_input: usize, max_output: usize) -> Result<Self, String> {
        let api = DriverApi::load()?;
        unsafe {
            check((api.init)(0), "cuInit")?;
            let mut count = 0;
            check((api.device_get_count)(&mut count), "cuDeviceGetCount")?;
            if count == 0 {
                return Err("No CUDA-capable NVIDIA device found".into());
            }

            let mut device = 0;
            check((api.device_get)(&mut device, 0), "cuDeviceGet")?;
            let mut context = ptr::null_mut();
            check((api.ctx_create)(&mut context, 0, device), "cuCtxCreate")?;
            check((api.ctx_set_current)(context), "cuCtxSetCurrent")?;

            let mut total_bytes = 0;
            check(
                (api.device_total_mem)(&mut total_bytes, device),
                "cuDeviceTotalMem",
            )?;
            let mut free_bytes = 0;
            let mut runtime_total = 0;
            check(
                (api.mem_get_info)(&mut free_bytes, &mut runtime_total),
                "cuMemGetInfo",
            )?;

            let mut name = [0i8; 256];
            check(
                (api.device_get_name)(name.as_mut_ptr(), name.len() as c_int, device),
                "cuDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let mut compute_major = 0;
            let mut compute_minor = 0;
            check(
                (api.device_get_compute_capability)(&mut compute_major, &mut compute_minor, device),
                "cuDeviceComputeCapability",
            )?;

            let mut module = ptr::null_mut();
            let ptx = CString::new(MATVEC_PTX).expect("PTX has no interior NUL");
            let mut jit_error_log = [0i8; 4096];
            let mut options: [c_int; 2] = [5, 6]; // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
            let mut option_values = [
                jit_error_log.as_mut_ptr() as *mut c_void,
                jit_error_log.len() as *mut c_void,
            ];
            let result = (api.module_load_data_ex)(
                &mut module,
                ptx.as_ptr().cast(),
                options.len() as c_uint,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            );
            if result != 0 {
                let log = jit_error_log
                    .iter()
                    .map(|byte| *byte as u8)
                    .take_while(|byte| *byte != 0)
                    .collect::<Vec<_>>();
                let log = String::from_utf8_lossy(&log);
                let _ = (api.ctx_destroy)(context);
                return Err(format!(
                    "cuModuleLoadDataEx(matvec) failed with CUDA driver error {result}: {log}"
                ));
            }
            let mut matvec = ptr::null_mut();
            check(
                (api.module_get_function)(
                    &mut matvec,
                    module,
                    b"momentum_matvec_f32\0".as_ptr().cast(),
                ),
                "cuModuleGetFunction(matvec)",
            )?;

            let mut backend = Self {
                api,
                context,
                module,
                matvec,
                allocations: Vec::new(),
                layers: Vec::new(),
                input_buffer: 0,
                output_buffer: 0,
                max_input,
                max_output,
                disabled_reason: None,
                device_info: CudaDeviceInfo {
                    name,
                    total_bytes,
                    free_bytes,
                    compute_major,
                    compute_minor,
                },
            };
            backend.input_buffer = backend.allocate(max_input.saturating_mul(4).max(4))?;
            backend.output_buffer = backend.allocate(max_output.saturating_mul(4).max(4))?;
            let mut free_after_setup = 0;
            check(
                (backend.api.mem_get_info)(&mut free_after_setup, &mut runtime_total),
                "cuMemGetInfo(after init)",
            )?;
            backend.device_info.free_bytes = free_after_setup;
            Ok(backend)
        }
    }

    fn set_current(&self) -> Result<(), String> {
        unsafe { check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent") }
    }

    fn allocate(&mut self, bytes: usize) -> Result<DevicePtr, String> {
        self.set_current()?;
        let mut ptr = 0;
        unsafe {
            check((self.api.mem_alloc)(&mut ptr, bytes.max(1)), "cuMemAlloc")?;
        }
        self.allocations.push(ptr);
        Ok(ptr)
    }

    pub fn free_bytes(&self) -> Result<usize, String> {
        self.set_current()?;
        let mut free = 0;
        let mut total = 0;
        unsafe {
            check(
                (self.api.mem_get_info)(&mut free, &mut total),
                "cuMemGetInfo",
            )?;
        }
        Ok(free)
    }

    pub fn allocation_checkpoint(&self) -> usize {
        self.allocations.len()
    }
    pub fn disable(&mut self, reason: String) {
        self.disabled_reason = Some(reason);
    }
    pub fn disabled_reason(&self) -> Option<&str> {
        self.disabled_reason.as_deref()
    }

    pub fn rollback_allocations(&mut self, checkpoint: usize) {
        let _ = self.set_current();
        while self.allocations.len() > checkpoint {
            if let Some(ptr) = self.allocations.pop() {
                unsafe {
                    let _ = (self.api.mem_free)(ptr);
                }
            }
        }
    }

    pub fn upload_f16(&mut self, weights: &[half::f16]) -> Result<DevicePtr, String> {
        let bytes = weights
            .len()
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| "GPU tensor size overflow".to_string())?;
        let ptr = self.allocate(bytes)?;
        self.set_current()?;
        unsafe {
            check(
                (self.api.memcpy_htod)(ptr, weights.as_ptr().cast(), bytes),
                "cuMemcpyHtoD(weights)",
            )?;
        }
        Ok(ptr)
    }

    pub fn add_layer(&mut self, matrices: CudaLayerMatrices) {
        self.layers.push(matrices);
    }

    pub fn matvec(
        &mut self,
        matrix: DevicePtr,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        if rows > self.max_output || cols > self.max_input {
            return Err(format!(
                "CUDA matvec exceeds workspace ({}x{}, capacity {}x{})",
                rows, cols, self.max_output, self.max_input
            ));
        }
        self.set_current()?;
        let input_bytes = cols
            .checked_mul(4)
            .ok_or_else(|| "input size overflow".to_string())?;
        let output_bytes = rows
            .checked_mul(4)
            .ok_or_else(|| "output size overflow".to_string())?;
        unsafe {
            check(
                (self.api.memcpy_htod)(self.input_buffer, input.as_ptr().cast(), input_bytes),
                "cuMemcpyHtoD(input)",
            )?;
            let mut weight = matrix;
            let mut input_ptr = self.input_buffer;
            let mut output_ptr = self.output_buffer;
            let mut ncols = cols as c_uint;
            let mut nrows = rows as c_uint;
            let mut params: [*mut c_void; 5] = [
                (&mut weight as *mut DevicePtr).cast(),
                (&mut input_ptr as *mut DevicePtr).cast(),
                (&mut output_ptr as *mut DevicePtr).cast(),
                (&mut ncols as *mut c_uint).cast(),
                (&mut nrows as *mut c_uint).cast(),
            ];
            let block = 128u32;
            let grid = ((rows as u32).saturating_add(block - 1) / block).max(1);
            check(
                (self.api.launch_kernel)(
                    self.matvec,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(matvec)",
            )?;
            check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
            check(
                (self.api.memcpy_dtoh)(
                    output.as_mut_ptr().cast(),
                    self.output_buffer,
                    output_bytes,
                ),
                "cuMemcpyDtoH(output)",
            )?;
        }
        Ok(())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.api.ctx_set_current)(self.context);
            for allocation in self.allocations.drain(..).rev() {
                let _ = (self.api.mem_free)(allocation);
            }
            if !self.module.is_null() {
                let _ = (self.api.module_unload)(self.module);
            }
            if !self.context.is_null() {
                let _ = (self.api.ctx_destroy)(self.context);
            }
        }
    }
}

fn check(code: ResultCode, operation: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed with CUDA driver error {code}"))
    }
}

/// The GPU kernel is deliberately simple and deterministic. It computes one
/// output row per thread; CPU handles normalization, attention and activation.
const MATVEC_PTX: &str = r#"
.version 6.0
.target sm_61
.address_size 64
.visible .entry momentum_matvec_f32(
    .param .u64 pW,
    .param .u64 pX,
    .param .u64 pY,
    .param .u32 pCols,
    .param .u32 pRows
) {
    .reg .pred %p<3>;
    .reg .b16 %h<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [pW];
    ld.param.u64 %rd2, [pX];
    ld.param.u64 %rd3, [pY];
    ld.param.u32 %r1, [pCols];
    ld.param.u32 %r2, [pRows];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r6, %r1;
    shl.b64 %rd4, %rd4, 1;
    add.u64 %rd5, %rd1, %rd4;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;
LOOP:
    setp.ge.u32 %p2, %r7, %r1;
    @%p2 bra STORE;
    cvt.u64.u32 %rd6, %r7;
    shl.b64 %rd6, %rd6, 1;
    add.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd8, %rd6, 1;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.b16 %h1, [%rd7];
    cvt.f32.f16 %f2, %h1;
    ld.global.f32 %f3, [%rd8];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1;
    bra LOOP;
STORE:
    mul.wide.u32 %rd9, %r6, 4;
    add.u64 %rd9, %rd3, %rd9;
    st.global.f32 [%rd9], %f1;
DONE:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_ptx_matvec_matches_cpu_reference() {
        let mut cuda = match CudaBackend::new(4, 2) {
            Ok(cuda) => cuda,
            Err(reason)
                if reason.contains("libcuda.so.1")
                    || reason.contains("No CUDA-capable")
                    || reason.contains("cuInit failed")
                    || reason.contains("cuDeviceGetCount failed") =>
            {
                eprintln!("CUDA hardware/runtime unavailable; GPU smoke test skipped: {reason}");
                return;
            }
            Err(reason) => {
                panic!("CUDA driver is present but backend initialization failed: {reason}")
            }
        };
        let weights = [1.0f32, 2.0, 3.0, 4.0].map(half::f16::from_f32);
        let ptr = cuda.upload_f16(&weights).unwrap();
        let mut output = [0.0f32; 2];
        cuda.matvec(ptr, &[2.0, 3.0], &mut output, 2, 2).unwrap();
        assert!((output[0] - 8.0).abs() < 0.01);
        assert!((output[1] - 18.0).abs() < 0.01);
    }
}
