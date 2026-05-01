use std::thread;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeResourceSnapshot {
    pub(crate) logical_cpu_count: usize,
    pub(crate) memory_bytes: usize,
}

pub(crate) fn detect_local_node_resources() -> Result<NodeResourceSnapshot, String> {
    let logical_cpu_count = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let memory_bytes = detect_total_memory_bytes().ok_or_else(|| {
        "failed to detect total system memory for distributed resource discovery".to_string()
    })?;
    Ok(NodeResourceSnapshot {
        logical_cpu_count,
        memory_bytes,
    })
}

#[cfg(target_os = "linux")]
fn detect_total_memory_bytes() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<usize>().ok())?;
            return kib.checked_mul(1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn detect_total_memory_bytes() -> Option<usize> {
    use std::ffi::{CString, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const i8,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }

    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len != std::mem::size_of::<u64>() {
        return None;
    }
    usize::try_from(value).ok()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_total_memory_bytes() -> Option<usize> {
    None
}
