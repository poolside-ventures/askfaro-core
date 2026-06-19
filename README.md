# askfaro-core

The public Rust on-device SDK for [Faro](https://www.askfaro.com) — run Faro capabilities
locally on the device, falling back to the API for the rest.

This is the native counterpart to the [`askfaro`](https://pypi.org/project/askfaro/) Python SDK:
the same on-device capability layer, packaged as Rust crates with bindings for desktop and mobile
(Tauri, iOS/Android via UniFFI, and WebAssembly).

## Crates

| Crate | What it does |
|-------|--------------|
| [`askfaro-core-stt`](core-stt) | On-device speech-to-text (NVIDIA Parakeet via ONNX Runtime). Opt-in; pulls a native runtime, so depend on it only when you want voice input. |

More crates (free tools, the local-first client) land here as the SDK grows.

## Speech-to-text quick start

```toml
[dependencies]
askfaro-core-stt = { git = "https://github.com/poolside-ventures/askfaro-core" }
```

```rust
use askfaro_core_stt::SttEngine;

let mut engine = SttEngine::load("/path/to/parakeet-model")?;
let transcript = engine.transcribe(pcm_f32, 16_000, 1)?;
println!("{}", transcript.text);
```

The model is **not** bundled — fetch it once at runtime and point `load` at the directory. It runs
fully offline thereafter, multilingual, faster than real time on commodity hardware.

## License

MIT. See [LICENSE](LICENSE).

> Generated mirror — crate sources are maintained in the Faro monorepo and synced here. Do not edit
> the mirrored crate sources in this repo directly.
