import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
MOSAIC_PATH = os.path.join(PROJECT_ROOT, "third_party", "MOSAIC")
sys.path.append(MOSAIC_PATH)
os.chdir(MOSAIC_PATH)
from diffusers import FluxPipeline

from environ_config import FLUX_PATH, MOSAIC_MODEL_PATH
from third_party.MOSAIC.src.flux_omini import Condition, generate
from third_party.MOSAIC.utils import process_image


def load_pipeline(device="cuda"):
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)

    pipe.load_lora_weights(MOSAIC_MODEL_PATH, weight_name="subject_512.safetensors", adapter_name="subject")
    pipe.set_adapters(["subject"], [1])

    return pipe


def generate_composite_image(
    pipe,
    prompt,
    ref_image_paths,
    height=512,
    width=512,
    ref_size=512,
    num_inference_steps=28,
    guidance_scale=3.5,
):
    ref_imgs = []
    for ref_image_path in ref_image_paths:
        pil_img = process_image(ref_image_path, target_size=512, pad_color=(255, 255, 255), scale=0.9)
        # if the processed image is all white, discard it
        if np.array(pil_img).mean() > 250:
            print(f"Skipping empty reference image: {ref_image_path}")
            continue

        ref_imgs.append(pil_img)

    position_deltas = []
    for i in range(len(ref_imgs)):
        position_deltas.append([0, -(ref_size * (i + 1)) // 16])

    conditions = [Condition(appearance, "subject", position_deltas[i]) for i, appearance in enumerate(ref_imgs)]

    with torch.inference_mode():
        generated_image = generate(
            pipe,
            conditions=conditions,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )[0]

    return generated_image[0]
