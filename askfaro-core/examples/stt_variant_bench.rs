//! Transcribe a manifest of wavs with one Parakeet model directory and dump the
//! hypotheses, so two weight variants can be scored against the same references.
//!
//! One process per variant, same reason as `embed_variant_bench`: the resident
//! footprint is one of the answers, and two encoders in one process makes it
//! unattributable.
//!
//! ```text
//! cargo run --release --features stt --example stt_variant_bench -- \
//!   --model <model_dir> --manifest /tmp/asr-eval/manifest.jsonl \
//!   --label int4 --out /tmp/asr-eval/hyp-int4.json
//! ```
//!
//! The manifest is one JSON object per line with at least `wav`; everything else
//! is copied through to the output untouched, so the scorer keeps the reference,
//! the language and the set label without this binary knowing what they mean.

#[cfg(not(feature = "stt"))]
fn main() {
    eprintln!("build with --features stt");
}

#[cfg(feature = "stt")]
fn main() {
    use askfaro_core::stt::SttEngine;
    use std::time::Instant;

    let (mut model, mut manifest, mut out, mut label) =
        (String::new(), String::new(), String::new(), "variant".to_string());
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < argv.len() {
        let v = argv[i + 1].clone();
        match argv[i].as_str() {
            "--model" => model = v,
            "--manifest" => manifest = v,
            "--out" => out = v,
            "--label" => label = v,
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    assert!(!model.is_empty() && !manifest.is_empty() && !out.is_empty());

    let before = footprint();
    let t = Instant::now();
    let mut engine = SttEngine::load(&model).unwrap_or_else(|e| panic!("load {model}: {e}"));
    let load_ms = t.elapsed().as_millis();
    let after_load = footprint();

    let clips: Vec<serde_json::Value> = std::fs::read_to_string(&manifest)
        .expect("manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("manifest line"))
        .collect();
    eprintln!("{label}: {} clips", clips.len());

    let mut results = Vec::with_capacity(clips.len());
    let mut audio_secs = 0.0f64;
    let mut wall_secs = 0.0f64;
    for (n, clip) in clips.iter().enumerate() {
        let path = clip["wav"].as_str().expect("wav");
        let mut reader = hound::WavReader::open(path).expect("open wav");
        let spec = reader.spec();
        let pcm: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
        };
        let dur = pcm.len() as f64 / spec.sample_rate as f64 / spec.channels as f64;
        let t = Instant::now();
        let text = match engine.transcribe(pcm, spec.sample_rate, spec.channels) {
            Ok(t) => t.text,
            Err(e) => {
                eprintln!("{label}: {path} FAILED: {e}");
                String::new()
            }
        };
        let wall = t.elapsed().as_secs_f64();
        audio_secs += dur;
        wall_secs += wall;
        let mut row = clip.clone();
        row["hyp"] = serde_json::Value::String(text);
        row["wallSecs"] = serde_json::json!(wall);
        results.push(row);
        if n % 25 == 0 {
            eprintln!("  {label}: {n}/{}", clips.len());
        }
    }
    let after = footprint();

    let payload = serde_json::json!({
        "label": label,
        "model": model,
        "modelBytes": dir_bytes(&model),
        "loadMs": load_ms,
        "audioSecs": audio_secs,
        "wallSecs": wall_secs,
        "rtfx": audio_secs / wall_secs,
        "footprint": { "before": before, "afterLoad": after_load, "afterAll": after },
        "results": results,
    });
    std::fs::write(&out, serde_json::to_string(&payload).unwrap()).expect("write");
    eprintln!(
        "{label}: {:.0}s audio in {:.0}s (RTFx {:.1}), {} MiB on disk, {:.0} MiB resident after load -> {out}",
        audio_secs,
        wall_secs,
        audio_secs / wall_secs,
        dir_bytes(&model) / 1048576,
        after_load["physFootprintMiB"].as_f64().unwrap_or(0.0)
    );
}

#[cfg(feature = "stt")]
fn dir_bytes(dir: &str) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Same measurement as `embed_variant_bench`: dirty plus compressed, from
/// `vmmap`, because the weights are mmapped and RSS counts clean file pages that
/// cost the machine nothing under pressure.
#[cfg(feature = "stt")]
fn footprint() -> serde_json::Value {
    let pid = std::process::id();
    let Ok(out) = std::process::Command::new("/usr/bin/vmmap")
        .args(["--summary", &pid.to_string()])
        .output()
    else {
        return serde_json::json!({ "error": "vmmap unavailable" });
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut phys, mut dirty, mut clean, mut swapped) = (0u64, 0u64, 0u64, 0u64);
    for line in text.lines() {
        let l = line.trim();
        if let Some(v) = l.strip_prefix("Physical footprint:") {
            phys = parse_size(v);
        } else if l.starts_with("TOTAL") && !l.starts_with("TOTAL,") {
            let cols: Vec<&str> = l.split_whitespace().collect();
            if cols.len() >= 5 && cols[0] == "TOTAL" {
                let resident = parse_size(cols[2]);
                dirty = parse_size(cols[3]);
                swapped = parse_size(cols[4]);
                clean = resident.saturating_sub(dirty);
            }
        }
    }
    serde_json::json!({
        "physFootprintMiB": phys as f64 / 1048576.0,
        "dirtyMiB": dirty as f64 / 1048576.0,
        "cleanResidentMiB": clean as f64 / 1048576.0,
        "swappedMiB": swapped as f64 / 1048576.0,
    })
}

#[cfg(feature = "stt")]
fn parse_size(s: &str) -> u64 {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix('G') {
        (n, 1u64 << 30)
    } else if let Some(n) = s.strip_suffix('M') {
        (n, 1u64 << 20)
    } else if let Some(n) = s.strip_suffix('K') {
        (n, 1u64 << 10)
    } else {
        (s, 1)
    };
    num.trim().parse::<f64>().map(|v| (v * mult as f64) as u64).unwrap_or(0)
}
