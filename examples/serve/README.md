# Online Serving Examples with `trtllm-serve`

We provide a CLI command, `trtllm-serve`, to launch a FastAPI server compatible with OpenAI APIs, here are some client examples to query the server, you can check the source code here or refer to the [command documentation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) and [examples](https://nvidia.github.io/TensorRT-LLM/examples/trtllm_serve_examples.html) for detailed information and usage guidelines.

## Anthropic Messages API and Claude Code

See the [Anthropic compatibility example](anthropic_compatibility/usage.md) to
build this checkout, launch `trtllm-serve`, point Claude Code at `/v1/messages`,
and run deterministic client-tool benchmarks.
