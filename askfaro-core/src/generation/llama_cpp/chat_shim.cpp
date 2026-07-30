// C ABI over llama.cpp's common_chat_* machinery.
//
// The llama-cpp-2 crate binds the LEGACY llama_chat_apply_template (role and
// content only, no tools). Everything we actually need for a tool-calling agent
// lives in common/chat.h, which is already compiled into libllama-common.a and
// already linked by llama-cpp-sys-2 (it emits `rustc-link-lib=static=llama-common`
// under its default `common` feature). So this file is a thin bridge, not a
// reimplementation: the template rendering, the grammar generation, the tool-call
// parsing and the reasoning/content split are all upstream's.
//
// State (the templates handle and the last apply result) is kept on this side so
// common_chat_params never has to be serialized across FFI. common_chat_parse
// needs the format from the apply step, and common_chat_parser_params has a
// constructor taking common_chat_params, so holding it here is the cheap path.

#include "chat.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <string>

using json = nlohmann::ordered_json;

struct scope_chat_ctx {
    common_chat_templates_ptr tmpls;
    common_chat_params        last;
    bool                      has_last = false;
};

static char * dup_cstr(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

extern "C" {

/// Build a templates handle from the model's own Jinja template string.
///
/// Passing the template rather than the llama_model pointer keeps this
/// independent of the Rust crate's private model handle: the crate already
/// exposes `model.chat_template()`, so Rust reads it and hands it over.
scope_chat_ctx * scope_chat_init(const char * template_str) {
    try {
        auto * ctx = new scope_chat_ctx();
        ctx->tmpls = common_chat_templates_init(nullptr, template_str ? template_str : "");
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

/// Render a prompt from messages + tool schemas through the model's real template.
///
/// `messages_json`: [{"role": "...", "content": "..."}]
/// `tools_json`:    [{"name": "...", "description": "...", "parameters": {...}}]
///
/// Returns a JSON object with the rendered prompt plus everything the caller
/// needs to decode the reply: the auto-generated grammar for constrained tool
/// calls, and the thinking tags derived from the template itself.
char * scope_chat_apply(scope_chat_ctx * ctx,
                        const char *     messages_json,
                        const char *     tools_json,
                        bool             enable_thinking) {
    if (!ctx) {
        return nullptr;
    }
    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = enable_thinking;
        // deepseek is the format that yields a separate reasoning_content rather
        // than leaving the think block inline in content.
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

        for (const auto & m : json::parse(messages_json)) {
            common_chat_msg msg;
            msg.role    = m.at("role").get<std::string>();
            msg.content = m.value("content", std::string());
            inputs.messages.push_back(msg);
        }

        if (tools_json && *tools_json) {
            for (const auto & t : json::parse(tools_json)) {
                common_chat_tool tool;
                tool.name        = t.at("name").get<std::string>();
                tool.description = t.value("description", std::string());
                tool.parameters  = t.at("parameters").dump();
                inputs.tools.push_back(tool);
            }
        }

        ctx->last     = common_chat_templates_apply(ctx->tmpls.get(), inputs);
        ctx->has_last = true;

        json out;
        out["prompt"] = ctx->last.prompt;
        // Separate from `prompt` in the struct, and it matters: feed only
        // `prompt` and the model has to emit its own turn opener, which arrives
        // as a literal "<|turn>model" at the head of the output and defeats the
        // parser. common_chat_parser_params copies this field for the same
        // reason, so the parse side needs it to agree with what was generated.
        out["generation_prompt"] = ctx->last.generation_prompt;
        out["grammar"]           = ctx->last.grammar;
        out["grammar_lazy"]      = ctx->last.grammar_lazy;
        out["supports_thinking"] = ctx->last.supports_thinking;
        out["thinking_start"]    = ctx->last.thinking_start_tag;
        out["thinking_end"]      = ctx->last.thinking_end_tag;
        out["format"]            = common_chat_format_name(ctx->last.format);
        out["n_stops"]           = ctx->last.additional_stops.size();
        return dup_cstr(out.dump());
    } catch (const std::exception & e) {
        return dup_cstr(json{{"error", e.what()}}.dump());
    }
}

/// Parse raw model output into content / reasoning_content / tool_calls.
///
/// This is the function under test: the open upstream reports describe Gemma 4
/// tool calls arriving as raw tokens in `content` instead of being parsed out.
char * scope_chat_parse(scope_chat_ctx * ctx, const char * text) {
    if (!ctx || !ctx->has_last) {
        return nullptr;
    }
    try {
        common_chat_parser_params params(ctx->last);
        params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        params.parse_tool_calls = true;

        // THE subtle bit. The convenience constructor
        // `common_chat_parser_params(const common_chat_params &)` copies only
        // `format` and `generation_prompt`; it leaves `parser` (the compiled PEG
        // arena) default-constructed. Gemma 4 resolves to format `peg-gemma4`,
        // so without this the PEG match has nothing to match with and every
        // tool call falls through to raw `content` with an empty
        // `reasoning_content`. That is exactly the symptom in the open upstream
        // reports, and it is a caller mistake rather than an upstream bug.
        if (!ctx->last.parser.empty()) {
            params.parser = common_peg_arena::from_json(json::parse(ctx->last.parser));
        }

        const common_chat_msg msg = common_chat_parse(text, /*is_partial=*/false, params);

        json calls = json::array();
        for (const auto & tc : msg.tool_calls) {
            calls.push_back({{"name", tc.name}, {"arguments", tc.arguments}, {"id", tc.id}});
        }
        json out;
        out["content"]           = msg.content;
        out["reasoning_content"] = msg.reasoning_content;
        out["tool_calls"]        = calls;
        return dup_cstr(out.dump());
    } catch (const std::exception & e) {
        return dup_cstr(json{{"error", e.what()}}.dump());
    }
}

void scope_chat_free(char * p) { std::free(p); }

void scope_chat_ctx_free(scope_chat_ctx * ctx) { delete ctx; }

} // extern "C"
