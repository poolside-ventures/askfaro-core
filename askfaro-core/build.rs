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
        let candidates = find_llama_src();
        assert!(
            !candidates.is_empty(),
            "llama-cpp feature is on but the vendored llama.cpp source was not found; \
             it ships inside the llama-cpp-sys-2 crate",
        );
        println!("cargo:rerun-if-changed=src/generation/llama_cpp/chat_shim.cpp");
        // Compile against each candidate until one works, newest first.
        //
        // There is usually exactly one, and then this is a single compile. More
        // than one appears as soon as a machine builds two projects that pin
        // different llama-cpp-sys-2 versions, and the shim tracks upstream's
        // `common_chat_*` API closely enough that the wrong headers do not
        // build (`thinking_end_tag`, `common_reasoning_budget_init` both moved
        // inside 0.1.15x). This used to take whatever the directory listing
        // returned first, so which one got picked was down to readdir order:
        // the same tree built or failed depending on what some other project
        // had left in the registry cache.
        //
        // Compiling IS the version check. A header set that satisfies the shim
        // is by definition the one to use, and no parsing of versions or of
        // cargo's resolution can say that as directly.
        let mut errors = Vec::new();
        let mut compiled = false;
        for llama in &candidates {
            let result = cc::Build::new()
                .cpp(true)
                .std("c++17")
                .file("src/generation/llama_cpp/chat_shim.cpp")
                .include(llama.join("common"))
                .include(llama.join("include"))
                .include(llama.join("ggml/include"))
                .include(llama.join("vendor"))
                .include(llama)
                // Upstream headers are not warning-clean and are not ours to fix.
                .flag_if_supported("-w")
                .cargo_warnings(candidates.len() == 1)
                .try_compile("askfaro_chat_shim");
            match result {
                Ok(()) => {
                    if candidates.len() > 1 {
                        println!(
                            "cargo:warning=askfaro-core: built the chat shim against {}",
                            llama.display()
                        );
                    }
                    compiled = true;
                    break;
                }
                Err(e) => errors.push(format!("{}: {e}", llama.display())),
            }
        }
        assert!(
            compiled,
            "the chat shim did not compile against any vendored llama.cpp found on this \
             machine. Tried:\n  {}",
            errors.join("\n  "),
        );
    }
}

/// Every llama.cpp tree vendored inside a llama-cpp-sys-2 in the registry,
/// newest version first.
///
/// Only COMPILATION needs it; linking resolves for free because llama-cpp-sys-2
/// emits `rustc-link-lib=static=llama-common`, and that archive is where the
/// `common_chat_*` symbols live. Cargo tells a build script nothing about which
/// version of a TRANSITIVE dependency it resolved (`DEP_*` metadata reaches
/// direct dependents only), so the caller settles it by compiling.
///
/// `FARO_LLAMA_CPP_SRC` overrides the search for a vendored or unpacked tree
/// that is not in the registry at all.
#[cfg(feature = "llama-cpp")]
fn find_llama_src() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    println!("cargo:rerun-if-env-changed=FARO_LLAMA_CPP_SRC");
    if let Ok(dir) = std::env::var("FARO_LLAMA_CPP_SRC") {
        return vec![PathBuf::from(dir)];
    }
    let cargo_home = match std::env::var("CARGO_HOME") {
        Ok(v) => PathBuf::from(v),
        Err(_) => match std::env::var("HOME") {
            Ok(h) => PathBuf::from(h).join(".cargo"),
            Err(_) => return Vec::new(),
        },
    };
    let registry = cargo_home.join("registry").join("src");
    let mut found: Vec<(Vec<u32>, PathBuf)> = Vec::new();
    let Ok(indexes) = std::fs::read_dir(registry) else { return Vec::new() };
    for index in indexes.flatten() {
        let Ok(crates) = std::fs::read_dir(index.path()) else { continue };
        for c in crates.flatten() {
            let name = c.file_name().to_string_lossy().to_string();
            let Some(version) = name.strip_prefix("llama-cpp-sys-2-") else { continue };
            let src = c.path().join("llama.cpp");
            if !src.join("common").join("chat.h").exists() {
                continue;
            }
            // Numeric, so 0.1.9 sorts below 0.1.153 rather than above it.
            let parts: Vec<u32> = version
                .split('.')
                .map(|p| p.parse().unwrap_or(0))
                .collect();
            found.push((parts, src));
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, p)| p).collect()
}
