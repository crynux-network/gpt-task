# DeepSeek-V3.2 Verified Coverage Sample

## Evidence Scope

This page records the gpt-task family adapter for the loaded `deepseek_v32` configuration. It documents the bundled official encoder integration and MUST NOT be treated as evidence for other DeepSeek versions.

The adapter MUST match `config.model_type=deepseek_v32` inside its own module and MUST take precedence over the generic tokenizer chat-template path. Repository IDs and local paths MUST NOT select it.

## Task Validation

The adapter consumes canonical text messages, optional tools, assistant tool-call history, tool results, and template arguments.

Image capability is not implemented by this family adapter. An image task MUST follow the common processor path rather than infer support from the DeepSeek model name.

## Runtime and Artifact Resolution

DeepSeek-V3.2 prompt adaptation does not select a model loader, AutoClass, classic backend, tensor-parallel backend, or world size. Those decisions MUST follow the common runtime and loaded-config rules.

No DeepSeek-V3.2-specific TP plan or reduced-world-size result is verified by this page.

Classic and TP execution MUST use the same adapter identity and rendered DSML prompt.

## Input Rendering

The adapter MUST use the bundled official `encode_messages` implementation instead of the tokenizer Jinja chat template.

When tools are present, the adapter MUST place them in the first system message. If a system message already contains a tool list, the request tools MUST be appended in order. Otherwise the adapter MUST add the tool list to the existing system message or create a leading system message.

The encoder MUST convert canonical assistant tool calls and tool-result history into the DeepSeek DSML representation. Its authoritative interface calls `json.loads` on each assistant-history `function.arguments` value. The supplied value MUST therefore be a JSON string containing an object; an object-valued Transformers representation is not valid for this family adapter.

The adapter MUST accept these template controls:

- `thinking`;
- `enable_thinking`;
- `thinking_mode`;
- `context`;
- `drop_thinking`;
- `add_default_bos_token`.

`thinking_mode` MUST be `thinking` or `chat`. When it is absent, truthy `thinking` or `enable_thinking` MUST select `thinking`; all other values MUST select `chat`.

`drop_thinking` MUST default to true when the last message role is `user`. `add_default_bos_token` MUST default to true. Unsupported template controls MUST NOT be forwarded to the official encoder.

## Generation and Raw Output

The rendered DSML prompt MUST enter the common generation path without downstream prompt rewriting.

gpt-task MUST decode generated DeepSeek thinking, ordinary content, and DSML tool-call syntax as raw assistant text. It MUST NOT convert DSML output into OpenAI tool calls or remove reasoning.

Direct gpt-task streaming MUST emit the same raw syntax as assistant deltas and one terminal `stop` or `length` finish reason.
