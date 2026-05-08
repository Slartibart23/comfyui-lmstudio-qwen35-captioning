import base64
import io
import json
import random
import re
import requests
import torch
import numpy as np
from PIL import Image


class LMStudioImageCaption:
    DESCRIPTION = (
        "Send an image from ComfyUI to a local LM Studio vision model, "
        "receive a caption back, and expose it as STRING output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Input image from ComfyUI. The image is sent to LM Studio "
                            "without resizing."
                        )
                    },
                ),

                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:1234/v1/chat/completions",
                        "tooltip": "LM Studio OpenAI-compatible chat completions endpoint."
                    },
                ),

                "model": (
                    "STRING",
                    {
                        "default": "DEINE_MODEL_ID_HIER_EINTRAGEN",
                        "tooltip": (
                            "Exact model ID as reported by LM Studio, for example from /v1/models."
                        )
                    },
                ),

                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Erstelle eine deutsche Bildcaption in 1 bis 3 Sätzen. "
                            "Keine Analyse. Keine Stichpunkte. Nur die finale Caption."
                        ),
                        "tooltip": "Instruction sent to the vision model together with the image."
                    },
                ),

                "thinking_mode": (
                    ["no_think", "normal"],
                    {
                        "default": "no_think",
                        "tooltip": (
                            "no_think prepends /no_think to the prompt. "
                            "This helps some Qwen reasoning models output final content. "
                            "Use normal for models that do not understand /no_think."
                        ),
                    },
                ),

                "image_format": (
                    ["data_uri_jpeg", "data_uri_png", "raw_base64"],
                    {
                        "default": "data_uri_jpeg",
                        "tooltip": (
                            "How the image is embedded in the request. "
                            "data_uri_jpeg sends data:image/jpeg;base64,... "
                            "data_uri_png sends data:image/png;base64,... "
                            "raw_base64 sends plain base64 only; your LM Studio setup previously rejected this."
                        ),
                    },
                ),


                "seed_mode": (
                    ["fixed", "random"],
                    {
                        "default": "fixed",
                        "tooltip": (
                            "fixed keeps the same seed value. "
                            "random creates a new seed on every run and also helps force ComfyUI to re-execute the node."
                        ),
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Seed value used when seed_mode is fixed."
                    },
                ),

                "send_seed_to_lmstudio": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "If off, the seed is only used inside ComfyUI. "
                            "Recommended: off, because some local vision models return empty content when seed is sent."
                        ),
                    },
                ),

                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Controls randomness. Lower values are more deterministic. "
                            "For captioning, 0.1 to 0.4 is usually good."
                        ),
                    },
                ),

                "max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 32,
                        "max": 262144,
                        "step": 32,
                        "tooltip": (
                            "Maximum number of output tokens. "
                            "For reasoning models, use 2048-4096 so the model can finish after reasoning. "
                            "Actual usable size depends on LM Studio context length."
                        ),
                    },
                ),

                "debug_response": (
                    ["off", "on"],
                    {
                        "default": "on",
                        "tooltip": "Print the full LM Studio JSON response to the ComfyUI console."
                    },
                ),

                "fallback_mode": (
                    ["final_or_reasoning", "final_only", "reasoning_only", "german_draft_from_reasoning"],
                    {
                        "default": "final_or_reasoning",
                        "tooltip": (
                            "final_only uses choices[0].message.content only. "
                            "final_or_reasoning uses content and falls back to reasoning_content if content is empty. "
                            "reasoning_only always outputs reasoning_content. "
                            "german_draft_from_reasoning tries to extract a German draft section from reasoning_content."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    OUTPUT_TOOLTIPS = ("Caption text returned by LM Studio.",)

    FUNCTION = "caption"
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True

    def _get_message_fields(self, data):
        if not isinstance(data, dict):
            return "", "", None

        choices = data.get("choices", [])
        if not choices:
            return "", "", None

        choice = choices[0]
        message = choice.get("message", {})

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        finish_reason = choice.get("finish_reason", None)

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(str(item["content"]))
            content = "\n".join(parts)

        if not isinstance(content, str):
            content = ""

        if not isinstance(reasoning_content, str):
            reasoning_content = ""

        return content.strip(), reasoning_content.strip(), finish_reason

    def _extract_german_draft(self, reasoning):
        if not reasoning:
            return ""

        markers = [
            "Drafting the description in German:",
            "**Drafting the description in German:**",
            "German:",
            "Deutsch:",
            "Beschreibung in German:",
            "Beschreibung auf Deutsch:",
        ]

        start_index = -1
        marker_used = ""

        for marker in markers:
            idx = reasoning.find(marker)
            if idx != -1:
                start_index = idx + len(marker)
                marker_used = marker
                break

        if start_index == -1:
            return reasoning.strip()

        text = reasoning[start_index:].strip()

        # Remove common planning/refinement tails if present
        stop_markers = [
            "Refining",
            "Final",
            "Now",
            "Let's",
            "I will",
            "The final",
        ]

        for marker in stop_markers:
            idx = text.find("\n" + marker)
            if idx > 0:
                text = text[:idx].strip()

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Remove markdown bullets and labels, but keep useful German text.
            line = re.sub(r"^[\*\-\u2022]\s*", "", line)
            line = re.sub(r"^\*\*(.*?)\*\*:\s*", r"\1: ", line)

            # Convert common labeled bullet lines into sentence-like parts.
            if ":" in line:
                label, value = line.split(":", 1)
                value = value.strip()
                if value:
                    line = value

            if line:
                lines.append(line)

        result = " ".join(lines).strip()

        # Clean repeated whitespace
        result = re.sub(r"\s+", " ", result)

        return result

    def _extract_caption(self, data, fallback_mode):
        content, reasoning, finish_reason = self._get_message_fields(data)

        if fallback_mode == "final_only":
            return content

        if fallback_mode == "reasoning_only":
            return reasoning

        if fallback_mode == "german_draft_from_reasoning":
            if content:
                return content
            return self._extract_german_draft(reasoning)

        # Default: final_or_reasoning
        if content:
            return content

        if reasoning:
            return reasoning

        return ""

    def _encode_image(self, pil_image, image_format):
        """
        No resizing is applied here.
        No jpeg_quality field is exposed in the node.
        """

        buffer = io.BytesIO()

        if image_format == "data_uri_png":
            pil_image.save(buffer, format="PNG")
            mime = "image/png"
        else:
            pil_image.save(buffer, format="JPEG")
            mime = "image/jpeg"

        image_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        if image_format == "raw_base64":
            image_url = img_b64
        else:
            image_url = f"data:{mime};base64,{img_b64}"

        return image_url, len(image_bytes), mime

    def _make_payload(
        self,
        model,
        prompt,
        image_url,
        temperature,
        max_tokens,
        used_seed,
        send_seed_to_lmstudio,
    ):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        if send_seed_to_lmstudio == "on":
            payload["seed"] = int(used_seed)

        return payload

    def caption(
        self,
        image,
        server_url,
        model,
        prompt,
        thinking_mode,
        image_format,
        seed_mode,
        seed,
        send_seed_to_lmstudio,
        temperature,
        max_tokens,
        debug_response,
        fallback_mode="final_or_reasoning",
    ):
        used_seed = seed

        try:
            if seed_mode == "random":
                used_seed = random.randint(0, 2147483647)

            # ComfyUI IMAGE is usually [batch, height, width, channels]
            img_tensor = image[0]

            if isinstance(img_tensor, torch.Tensor):
                img_array = img_tensor.detach().cpu().numpy()
            else:
                img_array = np.array(img_tensor)

            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array).convert("RGB")
            image_size = pil_image.size

            if thinking_mode == "no_think":
                final_prompt = "/no_think\n\n" + prompt
            else:
                final_prompt = prompt

            image_url, image_bytes_len, mime = self._encode_image(
                pil_image=pil_image,
                image_format=image_format,
            )

            payload = self._make_payload(
                model=model,
                prompt=final_prompt,
                image_url=image_url,
                temperature=temperature,
                max_tokens=max_tokens,
                used_seed=used_seed,
                send_seed_to_lmstudio=send_seed_to_lmstudio,
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer lm-studio",
            }

            print("----- LM Studio Caption Request -----")
            print(f"Model: {model}")
            print(f"Server URL: {server_url}")
            print(f"Thinking mode: {thinking_mode}")
            print(f"Image format: {image_format}")
            print(f"Image MIME: {mime}")
            print(f"Original/sent image size: {image_size}")
            print(f"Encoded image bytes: {image_bytes_len}")
            print(f"Fallback mode: {fallback_mode}")
            print(f"Seed mode: {seed_mode}")
            print(f"Seed used: {used_seed}")
            print(f"Send seed to LM Studio: {send_seed_to_lmstudio}")
            print(f"Temperature: {temperature}")
            print(f"Max tokens: {max_tokens}")

            response = requests.post(
                server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )

            print(f"HTTP status: {response.status_code}")

            if response.status_code != 200:
                caption = f"LM Studio error {response.status_code}: {response.text}"
            else:
                data = response.json()

                if debug_response == "on":
                    print("----- LM Studio Raw Response -----")
                    print(json.dumps(data, indent=2, ensure_ascii=False))

                content, reasoning, finish_reason = self._get_message_fields(data)
                caption = self._extract_caption(data, fallback_mode)

                if not caption:
                    caption = (
                        "LM Studio returned no usable text.\n\n"
                        f"finish_reason: {finish_reason}\n"
                        f"model: {model}\n"
                        f"thinking_mode: {thinking_mode}\n"
                        f"image_format: {image_format}\n"
                        f"fallback_mode: {fallback_mode}\n"
                        f"image_size: {image_size}\n"
                        f"max_tokens: {max_tokens}\n"
                        f"seed: {used_seed}\n"
                        f"send_seed_to_lmstudio: {send_seed_to_lmstudio}\n\n"
                        "Check the ComfyUI console under 'LM Studio Raw Response'."
                    )

                # Add a useful warning in console when the model used all tokens for reasoning.
                if finish_reason == "length" and content == "" and reasoning:
                    print(
                        "WARNING: LM Studio returned empty final content but non-empty reasoning_content. "
                        "Increase max_tokens, use fallback_mode=final_or_reasoning, or use a non-reasoning vision model."
                    )

        except Exception as e:
            caption = f"Captioning error: {str(e)}"

        print("----- LM Studio Caption Result -----")
        print(caption)

        return {
            "ui": {
                "text": [caption]
            },
            "result": (caption,),
        }


NODE_CLASS_MAPPINGS = {
    "LMStudioImageCaption": LMStudioImageCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioImageCaption": "LM Studio Image Caption",
}
