// Apple Foundation Models bridge for askfaro-core-generation.
//
// Two @_cdecl entry points are exposed to Rust over swift-rs:
//   - afm_availability() -> SRString   : cheap capability probe, no model load
//   - afm_generate(SRString) -> SRString : run one turn (JSON in, JSON out)
//
// The JSON-Schema -> DynamicGenerationSchema conversion and transcript .toolCalls
// parsing are lifted from the validated F-7 spike
// (scope/desktop/spikes/f7-tool-calling/apple-bench). The system language model
// is a process-resident singleton, so it stays warm across calls automatically.
import FoundationModels
import Foundation
import SwiftRs

// MARK: - JSON Schema -> DynamicGenerationSchema (from the F-7 spike)

func buildSchema(_ j: [String: Any], name: String) -> DynamicGenerationSchema {
    let type_ = j["type"] as? String ?? "string"
    let desc = j["description"] as? String

    if let choices = j["enum"] as? [String] {
        return DynamicGenerationSchema(name: name, description: desc, anyOf: choices)
    }
    switch type_ {
    case "object":
        let props = j["properties"] as? [String: Any] ?? [:]
        let required = j["required"] as? [String] ?? []
        let dynProps: [DynamicGenerationSchema.Property] = props.keys.sorted().compactMap { key in
            guard let val = props[key] as? [String: Any] else { return nil }
            let child = buildSchema(val, name: key)
            return DynamicGenerationSchema.Property(
                name: key,
                description: val["description"] as? String,
                schema: child,
                isOptional: !required.contains(key)
            )
        }
        return DynamicGenerationSchema(name: name, description: desc, properties: dynProps)
    case "array":
        let items = j["items"] as? [String: Any] ?? [:]
        return DynamicGenerationSchema(arrayOf: buildSchema(items, name: name + "_item"))
    case "boolean":
        return DynamicGenerationSchema(type: Bool.self)
    case "integer", "number":
        return DynamicGenerationSchema(type: Int.self)
    default:
        return DynamicGenerationSchema(type: String.self)
    }
}

// MARK: - Dynamic tool (GeneratedContent as Arguments)

struct DynamicTool: Tool {
    nonisolated let name: String
    nonisolated let description: String
    nonisolated let parameters: GenerationSchema

    typealias Arguments = GeneratedContent
    typealias Output = String

    init(name: String, description: String, parameters: [String: Any]) throws {
        self.name = name
        self.description = description
        let dyn = buildSchema(parameters, name: name + "_args")
        self.parameters = try GenerationSchema(root: dyn, dependencies: [])
    }

    // The model's selection is read from the transcript; the call body is a no-op
    // because the host (Rust) executes tools, not this bridge.
    nonisolated func call(arguments: GeneratedContent) async throws -> String { "ok" }
}

// MARK: - Availability

@_cdecl("afm_availability")
public func afm_availability() -> SRString {
    switch SystemLanguageModel.default.availability {
    case .available:
        return SRString("available")
    case .unavailable(let reason):
        switch reason {
        case .deviceNotEligible:
            return SRString("unsupported:device not eligible for Apple Intelligence")
        case .appleIntelligenceNotEnabled:
            return SRString("notenabled:Apple Intelligence is not enabled in Settings")
        case .modelNotReady:
            return SRString("notenabled:the model is downloading or not yet ready")
        @unknown default:
            return SRString("notenabled:model unavailable")
        }
    @unknown default:
        return SRString("unsupported:unknown availability state")
    }
}

// MARK: - JSON helpers

private func jsonString(from object: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: object),
        let str = String(data: data, encoding: .utf8)
    else { return "{}" }
    return str
}

private func errorResponse(_ message: String) -> SRString {
    SRString(jsonString(from: ["error": message]))
}

// MARK: - Generate

/// Carries the result JSON (a Sendable String) out of the detached task. Marked
/// @unchecked Sendable because access is serialized by the semaphore: the task
/// writes `json` exactly once before signalling, and the caller reads it only
/// after waiting.
private final class ResultBox: @unchecked Sendable {
    var json: String = "{\"error\":\"generation did not complete\"}"
}

/// Run one turn. All parsing, tool building, and inference happen here so that
/// nothing non-Sendable crosses the `Task.detached` boundary — only the request
/// String goes in and a result String comes out.
private func runGeneration(requestJson raw: String) async -> String {
    guard let data = raw.data(using: .utf8),
        let req = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
        return jsonString(from: ["error": "invalid request json"])
    }

    let system = req["system"] as? String ?? ""
    let messages = req["messages"] as? [[String: Any]] ?? []
    let toolDefs = req["tools"] as? [[String: Any]] ?? []

    // Build the tool set from the already-selected subset.
    var tools: [any Tool] = []
    do {
        for def in toolDefs {
            let name = def["name"] as? String ?? ""
            let description = def["description"] as? String ?? ""
            let parameters = def["parameters"] as? [String: Any] ?? ["type": "object"]
            tools.append(try DynamicTool(name: name, description: description, parameters: parameters))
        }
    } catch {
        return jsonString(from: ["error": "failed to build tool schema: \(error)"])
    }

    // Instructions = system prompt. Prompt = the conversation, oldest first.
    let instructions = system.isEmpty ? "You are a helpful on-device assistant." : system
    let prompt = messages
        .map { ($0["role"] as? String ?? "user") + ": " + ($0["content"] as? String ?? "") }
        .joined(separator: "\n")

    let start = DispatchTime.now()
    let session = LanguageModelSession(tools: tools, instructions: instructions)
    do {
        let response = try await session.respond(to: prompt)
        let elapsedMs = (DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000

        // Read tool calls back out of the transcript.
        var toolCalls: [[String: Any]] = []
        for entry in session.transcript {
            if case .toolCalls(let calls) = entry {
                for call in calls {
                    let argsString = call.arguments.jsonString
                    let argsObject: Any =
                        (argsString.data(using: .utf8)
                            .flatMap { try? JSONSerialization.jsonObject(with: $0) }) ?? [:]
                    toolCalls.append(["name": call.toolName, "arguments": argsObject])
                }
            }
        }

        let text = response.content
        let abstained =
            toolCalls.isEmpty && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return jsonString(from: [
            "text": text,
            "tool_calls": toolCalls,
            "abstained": abstained,
            "model_ms": elapsedMs,
        ])
    } catch {
        // Map the context-window overflow to the typed sentinel the Rust side
        // turns into GenError::ContextWindowExceeded.
        if String(describing: error).contains("exceededContextWindowSize") {
            return jsonString(from: ["error": "context_window_exceeded"])
        }
        return jsonString(from: ["error": "\(error)"])
    }
}

@_cdecl("afm_generate")
public func afm_generate(_ requestJson: SRString) -> SRString {
    let raw = requestJson.toString()
    let box = ResultBox()
    let sem = DispatchSemaphore(value: 0)
    // Bridge the async API into the synchronous @_cdecl seam.
    Task.detached {
        box.json = await runGeneration(requestJson: raw)
        sem.signal()
    }
    sem.wait()
    return SRString(box.json)
}
