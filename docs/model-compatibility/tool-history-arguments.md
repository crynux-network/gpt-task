# Tool-History Argument Representation

## Evidence Scope

This page records verified input-rendering contracts for assistant-history `tool_calls[].function.arguments`. It covers the Transformers chat-template contract and published templates or encoders for Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, Llama 3.1, Llama 3.3, DeepSeek V3, DeepSeek R1, and DeepSeek V3.2.

These models are coverage samples. They MUST NOT be used as a runtime allowlist or as evidence for an uninspected model interface.

## Transformers Contract

[Transformers tool-use documentation](https://huggingface.co/docs/transformers/main/en/chat_extras) defines assistant-history function arguments as dictionaries. It explicitly distinguishes this representation from the OpenAI JSON string and states that passing the OpenAI representation to Transformers can cause errors or unexpected model behavior.

The generic tokenizer and processor paths MUST therefore receive object-valued `function.arguments`. gpt-task MUST pass the supplied representation to `apply_chat_template` without OpenAI API conversion.

## Verified Object Templates

The [Qwen2.5 template](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json) applies `tojson` to `tool_call.arguments`. The input MUST be an object; a JSON string is serialized again and changes the rendered tool call.

The [Qwen3 template](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json) explicitly renders a string directly and applies `tojson` to non-string values. It accepts both representations. Object input MUST be used at the Transformers boundary because it satisfies the common contract without changing the rendered argument object.

The published Qwen3.5 and Qwen3.6 templates iterate `tool_call.arguments` as object items. Their input MUST be an object.

The [Llama 3.1 model contract](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/README.md) supplies assistant-history arguments as a dictionary. Llama 3.1 and Llama 3.3 templates serialize that value into the model tool-call format. Their input MUST be an object.

## Verified String Interfaces

The [DeepSeek V3 template](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json) and [full DeepSeek R1 template](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json) concatenate `tool['function']['arguments']` with prompt strings. Their input MUST be a JSON string containing an object.

DeepSeek R1 Distill checkpoints use their Qwen or Llama base-model templates. The full DeepSeek R1 string contract MUST NOT be applied to those checkpoints.

The [DeepSeek V3.2 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py) calls `json.loads` on assistant-history arguments before expanding them into DSML parameters. Its input MUST be a JSON string containing an object.

## Detection Boundary

Transformers does not expose machine-readable metadata declaring the required nested argument type. Successful trial rendering MUST NOT be treated as type detection: Qwen2.5 and Llama templates accept a string but serialize it as a JSON string, while Qwen3 accepts both forms with equivalent object semantics.

The component that creates template-ready task messages MUST select a verified representation before gpt-task rendering. gpt-task MUST NOT retry a failed render with a different representation.
