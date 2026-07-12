# askfaro-core

The public Rust on-device SDK for [Faro](https://www.askfaro.com) — run Faro capabilities
locally on the device, falling back to the API for the rest.

This is the native counterpart to the [`askfaro`](https://pypi.org/project/askfaro/) Python SDK:
the same on-device capability layer, packaged as a single Rust crate with bindings for desktop and
mobile (Tauri, iOS/Android via UniFFI, and WebAssembly).

## One crate, capabilities as features

`askfaro-core` is a single crate; every capability is a cargo feature, so you add one dependency and
switch on exactly what you need. Heavy/platform dependencies are pulled only by the feature that
needs them, so the default build stays light and cross-compiles unchanged.

| Feature | What it does |
|---------|--------------|
| `search` *(default)* | Embedded hybrid retrieval — FTS5 lexical + bag-of-words semantic, RRF fusion. |
| `embeddinggemma` | Adds the on-device EmbeddingGemma vector embedder (ONNX). Heavy. |
| `generation` *(default)* | Provider-agnostic text generation + tool-calling (model-free types). |
| `apple-fm` | Apple Foundation Models generation provider (macOS/iOS 26+). |
| `progressive` *(default)* | Catalog selector: relevance ranking + budget-bounded pcx expansion. |
| `free-tools` | Deterministic in-process tools — calc, units, datetime, encoding, phone, random, timer, timezone, astronomy. Pure-Rust, no network; the same code the Faro API runs, so a local run and a cloud run are identical. Opt-in. |
| `stt` | On-device speech-to-text (NVIDIA Parakeet via ONNX Runtime). Opt-in; pulls a native runtime. |

```toml
[dependencies]
# Just speech-to-text, for example:
askfaro-core = { git = "https://github.com/poolside-ventures/askfaro-core", default-features = false, features = ["stt"] }
```

## Speech-to-text quick start

```rust
use askfaro_core::stt::SttEngine;

let mut engine = SttEngine::load("/path/to/parakeet-model")?;
let transcript = engine.transcribe(pcm_f32, 16_000, 1)?;
println!("{}", transcript.text);
```

The model is **not** bundled — fetch it once at runtime and point `load` at the directory. It runs
fully offline thereafter, multilingual, faster than real time on commodity hardware.

## Free / local tools quick start

Enable `free-tools` and run the deterministic tools in-process — no network, no credentials, no
billing. A host app runs a capability locally when it is available on the device and falls back to
the API for the rest.

```rust
use askfaro_core::free_tools;

// Discover what this build can run locally:
for name in free_tools::available() { println!("{name}"); }

// Execute by name; every tool returns the canonical `SkillResult` envelope:
let result = free_tools::execute("units", serde_json::json!({
    "value": 100, "from_unit": "celsius", "to_unit": "fahrenheit",
}));
println!("{}", result.to_json()?); // -> { "status": "success", "result": { "data": { "result": 212 } }, ... }

// Or the binding-friendly form — JSON string in, envelope JSON out (no Rust types cross FFI):
let json = free_tools::execute_free_tool_json("calc", r#"{"expression": "2 + 2 * 3"}"#);
```

## License

MIT. See [LICENSE](LICENSE).

> Generated mirror — crate sources are maintained in the Faro monorepo and synced here. Do not edit
> the mirrored crate sources in this repo directly.
