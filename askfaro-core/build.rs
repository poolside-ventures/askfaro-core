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
}
