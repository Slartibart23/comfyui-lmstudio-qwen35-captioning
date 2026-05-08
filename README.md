# ComfyUI LM Studio Qwen 3.5 Captioning

A ComfyUI custom node and reference workflow for sending images from ComfyUI to a local LM Studio vision model, receiving an English FLUX-style prompt or image caption, and showing/saving the result back inside ComfyUI.

## Important install note

This repository is structured so it can be cloned directly into `ComfyUI/custom_nodes`.

## Recommended model

Recommended model for this workflow:

```text
HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
```

Hugging Face:

```text
https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
```

NVIDIA RTX 5090 class GPU recommended for the recommended model. If your GPU is smaller, choose a different vision model or a smaller/more heavily quantized GGUF vision model.

Author note: In practical testing, this LM Studio + Qwen 3.5 workflow produced stronger adult/NSFW captioning than JoyCaptionBetaOne. Results depend heavily on model, quantization, prompt, GPU memory, and LM Studio load settings.

## Installation

### Direct clone, recommended

From your ComfyUI `custom_nodes` folder:

```bash
git clone https://github.com/Slartibart23/comfyui-lmstudio-qwen35-captioning.git
```

Final path must be:

```text
ComfyUI/custom_nodes/comfyui-lmstudio-qwen35-captioning/__init__.py
```

Restart ComfyUI completely.

### Manual copy

Manual copying is not needed for this repository. Clone the repository directly into `ComfyUI/custom_nodes` instead.

## LM Studio setup

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

## Recommended node settings

```text
server_url: http://127.0.0.1:1234/v1/chat/completions
model: your exact LM Studio model ID
thinking_mode: no_think
image_format: data_uri_jpeg
fallback_mode: final_or_reasoning
send_seed_to_lmstudio: off
temperature: 0.3
max_tokens: 4096
debug_response: on
```

## Reference workflow

The reference workflow is included here:

```text
workflows/QWEN35_LLM_Captioning.json
```

Depending on your ComfyUI installation, you may need:

- WAS Node Suite
- ComfyUI Easy Use
- ComfyUI Custom Scripts
- optional: `ComfyUI-LoadResizeImageWithFilenameV02`

## Troubleshooting

### Node does not appear in ComfyUI

Check the install path. For direct clone, it must be:

```text
ComfyUI/custom_nodes/comfyui-lmstudio-qwen35-captioning/__init__.py
```

Then restart ComfyUI fully and check the ComfyUI console for Python import errors.

### LM Studio error: Invalid url

Use:

```text
image_format: data_uri_jpeg
```

Do not use `raw_base64` if your LM Studio setup rejects it.

### Empty final caption

Use:

```text
thinking_mode: no_think
fallback_mode: final_or_reasoning
max_tokens: 4096
```

If the console shows `reasoning_content` but empty `content`, the fallback mode will still return usable text.


### Caption output is not clean or does not follow the FLUX style

If the returned caption is cut off, overly analytical, contains reasoning text, or does not cleanly follow the FLUX prompt style, increase both token budgets:

```text
ComfyUI node max_tokens: 4096 or higher
LM Studio context length: 16384 or higher if your hardware allows it
```

Reasoning-style models may spend many tokens before producing the final answer. If the output still looks messy, keep:

```text
thinking_mode: no_think
fallback_mode: final_or_reasoning
```

and simplify the prompt so it asks for only one final FLUX prompt with no analysis or bullet points.

### Context size exceeded

Increase LM Studio context length or lower the node's `max_tokens`.

## Adult-content note

This project is intended for legal, consenting-adult image captioning and dataset workflows only. Do not use it for minors, non-consensual imagery, exploitation, or illegal content.

## License

MIT License.
