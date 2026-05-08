

1. Open LM Studio.
2. Load a vision-capable model.
3. Open **Developer**.
4. Start the local server.
5. Keep the server running while ComfyUI uses the node.

Default endpoint:

```text
http://127.0.0.1:1234/v1/chat/completions
```

### Get the model ID

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:1234/v1/models | ConvertTo-Json -Depth 10
```

Copy the exact `id` value into the node's `model` field.

## If the FLUX prompt output is not clean

If the caption is cut off, contains too much reasoning, or does not cleanly follow the FLUX prompt style, increase both sides of the token budget:

```text
ComfyUI node max_tokens: 4096 or higher
LM Studio context length: 16384 or higher if your hardware allows it
```

The ComfyUI `max_tokens` value controls the requested generation length. LM Studio context length controls the total available prompt + image + output budget. Both can matter.
