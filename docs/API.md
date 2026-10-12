# APIs

Default Base URL: `http://127.0.0.1:30001/v1`. Model name: `Qwen3.8-flash-next`. With authentication enabled, add `Authorization: Bearer <key>`.

## Chat Completions

```bash
curl http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-flash-next","messages":[{"role":"user","content":"Explain Python generators."}],"max_tokens":512,"reasoning_effort":"none","stream":true}'
```

Supported `reasoning_effort` values: `none`, `low`, `medium`, `high`. The server parses reasoning separately from the answer.

## Responses

```bash
curl http://127.0.0.1:30001/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-flash-next","input":"Explain Python generators.","max_output_tokens":512,"reasoning":{"effort":"none"},"stream":true}'
```

## Images

Send a local image as a data URL:

```python
import base64
import requests

image = base64.b64encode(open("picture.png", "rb").read()).decode()
result = requests.post("http://127.0.0.1:30001/v1/chat/completions", json={
    "model": "Qwen3.8-flash-next",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}
    ]}],
    "max_tokens": 256, "reasoning_effort": "none"
}, timeout=300)
result.raise_for_status()
print(result.json())
```

## Tool calls

Use OpenAI's `tools` and `tool_choice` format. Function arguments are returned as a JSON string. The client executes the tool and sends its result back in a `role: "tool"` message.

```json
{
  "model": "Qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Use add to calculate 2+3."}],
  "tools": [{"type": "function", "function": {
    "name": "add",
    "description": "Add two integers",
    "parameters": {"type": "object", "properties": {
      "a": {"type": "integer"}, "b": {"type": "integer"}
    }, "required": ["a", "b"]}
  }}],
  "tool_choice": {"type": "function", "function": {"name": "add"}},
  "max_tokens": 128,
  "reasoning_effort": "none"
}
```

## Check the service

```bash
curl http://127.0.0.1:30001/health
curl http://127.0.0.1:30001/v1/models
curl http://127.0.0.1:30001/v1/status
.venv/bin/python tools/smoke.py
```

Anthropic Messages is available at `/v1/messages`. JSON Schema output uses prompting and result validation; constrained decoding is not provided.
