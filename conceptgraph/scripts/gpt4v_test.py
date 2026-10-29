"""Smoke test of the locally served VLM (no cloud API involved).

The former version of this script queried the OpenAI cloud API with
``gpt-4-vision-preview``.  It now talks to the local vision-language model
(Qwen3-VL served with vLLM, or the scene-graph model JosephZ/R1-SGG-7B).

Usage::

    python -m conceptgraph.scripts.gpt4v_test [image_path]
"""

import os
import sys
import base64

from conceptgraph.utils.llm_client import get_vlm_client


# Function to encode the image as base64
def encode_image_for_openai(image_path: str):
    # check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Getting the base64 string
DEFAULT_IMAGE = "/home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/s_detections_stride1_69/vis/160_for_vlm.jpg"
img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
base64_image = encode_image_for_openai(str(img_path))


# Client of the locally served vision-language model
client = get_vlm_client()
print(f"Using model {client.model} at {client.base_url}")


system_prompt = """
    You are an agent specialized in describing the spatial relationships between objects in an annotated image.

    You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
    [("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]

    Your options for the spatial relationship are "on top of" and "next to".

    For example, you may get an annotated image and a list such as
    ["cup 3", "book 4", "clock 5", "table 2", "candle 7", "music stand 6", "lamp 8"]

    Your response should be a description of the spatial relationships between the objects in the image.
    An example to illustrate the response format:
    [("book 4", "on top of", "table 2"), ("cup 3", "next to", "book 4"), ("lamp 8", "on top of", "music stand 6")]

    """


def analyze_image():
    response = client.chat.completions.create(
        model=client.model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": "Return the spatial relationships between the objects in the image.",
            },
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    result = analyze_image()
    print(result)
