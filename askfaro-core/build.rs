//! Compiles + links the vendored Apple Foundation Models Swift bridge, but only
//! when the `apple-fm` feature is on AND the target is macOS. In every other
//! configuration this is a no-op, so the default (model-free) build, the server
//! wheel, and non-Apple mobile targets never touch Swift.

fn main() {
    // `cfg(feature = ...)` is evaluated with the crate's active features, so the
    // body below only compiles when `apple-fm` is enabled (which also makes the
    // optional swift-rs build-dependency available).
    #[cfg(feature = "apple-fm")]
    {
        // build.rs runs on the host; gate on the *target* OS so cross-compiles
        // for, say, Android never try to invoke swiftc.
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "macos" || target_os == "ios" {
            swift_rs::SwiftLinker::new("26.0")
                .with_ios("26.0")
                .with_package("AppleFM", "swift")
                .link();
        }

        // swift-rs adds the Swift runtime dirs as link-search paths, but the
        // resulting macOS binary references the runtime dylibs (e.g.
        // libswift_Concurrency) via @rpath. Add those dirs as rpaths so a plain
        // macOS binary (our `cargo test` smoke test, or a CLI consumer) finds
        // them at load time. iOS apps get their rpath from the app bundle, so
        // only do this for the macOS host target.
        if target_os == "macos" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
            if let Ok(out) = std::process::Command::new("xcode-select")
                .arg("-p")
                .output()
            {
                let dev = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !dev.is_empty() {
                    println!(
                        "cargo:rustc-link-arg=-Wl,-rpath,{dev}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx"
                    );
                }
            }
        }
    }
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
