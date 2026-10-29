"""LLaVA wrapper for checkpoints in the Hugging Face ``transformers`` format.

The vendored :mod:`conceptgraph.llava.llava_model` loads the *original* LLaVA
release format (``LlavaLlamaForCausalLMTweaked``).  The LLaVA checkpoints that
are available on this machine are stored in the Hugging Face format
(``llava-hf/llava-v1.6-mistral-7b-hf``, ``llava-hf/llava-onevision-qwen2-7b-ov-hf``,
...), therefore this module provides a drop-in replacement with the same API::

    chat = LLaVAHFChat(model_path)
    pixels = chat.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    features = chat.encode_image(pixels[None, ...].half().cuda())
    chat.reset()
    answer = chat(query="Describe the central object in the image.",
                  image_features=features)

``model_path`` may be a local directory or a Hugging Face repo id; with
``HF_HOME`` pointing at the local hub the models are loaded from disk
(``HF_HUB_OFFLINE=1`` works as well).
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor

__all__ = ["LLaVAHFChat"]


class LLaVAHFChat:
    """Minimal chat wrapper around a Hugging Face LLaVA model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 1024,
    ) -> None:
        # CUDA-only: resolve and validate the target device before any
        # processor/tokenizer/model loading happens, so a bad device or a
        # missing GPU fails clearly instead of loading onto the CPU.
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError(
                "LLaVAHFChat is CUDA-only: device "
                f"{resolved_device} is not a CUDA device. Pass a CUDA device "
                "such as 'cuda' or 'cuda:0'."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLaVAHFChat is CUDA-only: CUDA is not available on this "
                "machine. Refusing to load on CPU or without a GPU."
            )

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        # ``image_processor`` is the CLIP/anyres processor of the checkpoint.
        self.image_processor = self.processor.image_processor

        from transformers import AutoModelForImageTextToText

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

        self._image: Image.Image | None = None
        self._messages: list[dict] = []
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Start a new conversation."""
        self._image = None
        self._messages = []

    def upload_image(self, image: Image.Image) -> None:
        """Store the image that the next query refers to."""
        self._image = image

    # ------------------------------------------------------------------
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image into the model's ``pixel_values`` layout."""
        self._image = image
        pixel_values = self.processor.image_processor(image, return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values

    def encode_image(self, pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        """Project the image into the language model's embedding space.

        ``pixel_values`` is accepted for API compatibility with the original
        LLaVA wrapper; the layout of the checkpoint's own processor is used
        whenever an image was uploaded.
        """
        image_sizes = None
        if self._image is not None:
            image_inputs = self.processor.image_processor(
                self._image, return_tensors="pt"
            )
            pixel_values = image_inputs["pixel_values"]
            # LLaVA-NeXT (Hugging Face) processors also return the
            # multi-resolution ``image_sizes`` that get_image_features needs.
            image_sizes = image_inputs.get("image_sizes")
        if pixel_values is None:
            raise ValueError("no image to encode")
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)
        with torch.inference_mode():
            if image_sizes is not None:
                # Multi-resolution LLaVA-NeXT checkpoints need image_sizes.
                try:
                    features = self.model.get_image_features(
                        pixel_values, image_sizes=image_sizes
                    )
                except TypeError:  # e.g. Onevision needs its own kwarg variants
                    features = self.model.get_image_features(
                        pixel_values,
                        vision_feature_layer=self.model.config.vision_feature_layer,
                        vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                    )
            else:
                try:
                    features = self.model.get_image_features(pixel_values)
                except TypeError:  # onevision needs the extra keyword arguments
                    features = self.model.get_image_features(
                        pixel_values,
                        vision_feature_layer=self.model.config.vision_feature_layer,
                        vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                    )
        if not isinstance(features, torch.Tensor):
            # Recent Transformers versions return a structured output whose
            # ``pooler_output`` carries the per-image feature tensors.
            pooler_output = getattr(features, "pooler_output", None)
            if pooler_output is not None:
                features = pooler_output
        if isinstance(features, (list, tuple)):
            features = features[0]
        return features

    # ------------------------------------------------------------------
    def __call__(self, query: str, image_features: torch.Tensor | None = None):
        """Answer ``query`` about the uploaded image."""
        if self._image is None:
            raise ValueError(
                "no image uploaded - call upload_image()/set chat._image before querying"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, images=self._image, return_tensors="pt"
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        answer = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        return answer
