//! Compiles the C++ chat shim for the `llama-cpp` provider. A no-op in every
//! other configuration, so the default (model-free) build, the server wheel and
//! mobile targets compile nothing here.
//!
//! This also used to build a vendored Apple Foundation Models Swift bridge,
//! removed 2026-08-01 along with the provider. Nothing links Swift now.

fn main() {
    // The `llama-cpp` provider needs a small C++ shim over llama.cpp's
    // `common_chat_*` API: template rendering with tool schemas, tool-call
    // parsing, and the reasoning/content split. The Rust bindings expose only
    // the LEGACY role+content chat template, but `libllama-common.a` (which
    // llama-cpp-sys-2 already builds and links under its default `common`
    // feature) carries the real thing, so this bridges to it rather than
    // reimplementing any of it.
    #[cfg(feature = "llama-cpp")]
    {
        let llama = find_llama_src().expect(
            "llama-cpp feature is on but the vendored llama.cpp source was not found; \
             it ships inside the llama-cpp-sys-2 crate",
        );
        println!("cargo:rerun-if-changed=src/generation/llama_cpp/chat_shim.cpp");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("src/generation/llama_cpp/chat_shim.cpp")
            .include(llama.join("common"))
            .include(llama.join("include"))
            .include(llama.join("ggml/include"))
            .include(llama.join("vendor"))
            .include(&llama)
            // Upstream headers are not warning-clean and are not ours to fix.
            .flag_if_supported("-w")
            .compile("askfaro_chat_shim");
    }
}

/// Locate the llama.cpp tree vendored inside llama-cpp-sys-2.
///
/// Only COMPILATION needs it; linking resolves for free because llama-cpp-sys-2
/// emits `rustc-link-lib=static=llama-common`, and that archive is where the
/// `common_chat_*` symbols live.
#[cfg(feature = "llama-cpp")]
fn find_llama_src() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let cargo_home = match std::env::var("CARGO_HOME") {
        Ok(v) => PathBuf::from(v),
        Err(_) => PathBuf::from(std::env::var("HOME").ok()?).join(".cargo"),
    };
    let registry = cargo_home.join("registry").join("src");
    for index in std::fs::read_dir(registry).ok()?.flatten() {
        let Ok(crates) = std::fs::read_dir(index.path()) else { continue };
        for c in crates.flatten() {
            if c.file_name().to_string_lossy().starts_with("llama-cpp-sys-2-") {
                let src = c.path().join("llama.cpp");
                if src.join("common").join("chat.h").exists() {
                    return Some(src);
                }
            }
        }
    }
    None
}
