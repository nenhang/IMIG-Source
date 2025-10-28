import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from insightface.app.face_analysis import FaceAnalysis
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
DREAMO_PATH = os.path.join(PROJECT_ROOT, "third_party", "DreamO")
sys.path.append(DREAMO_PATH)
os.chdir(DREAMO_PATH)
from environ_config import DREAMO_MODEL_PATH, FACE_MODEL_ROOT, FLUX_PATH, FLUX_TURBO_PATH
from third_party.DreamO.dreamo_generator import Generator, img2tensor, resize_numpy_image_area


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--version",
        type=str,
        default="v1.1",
        choices=["v1.1", "v1"],
        help="default will use the latest v1.1 model, you can also switch back to v1",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable 'quant=nunchaku' and 'offload' to reduce the original 24GB VRAM to 6.5GB.",
    )
    parser.add_argument(
        "--no_turbo", action="store_true", help="Use turbo to reduce the original 25 steps to 12 steps."
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="none",
        choices=["none", "int8", "nunchaku"],
        help="Quantize to use: none(bf16), int8, nunchaku",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: auto, cuda, mps, or cpu")
    args = parser.parse_args()
    return args


def get_args_wo_parsing():
    args = argparse.Namespace()
    args.port = 8080
    args.version = "v1.1"
    args.offload = False
    args.no_turbo = False
    args.quant = "none"
    args.device = "cuda"
    args.dreamo_model_path = DREAMO_MODEL_PATH
    args.model_path = FLUX_PATH
    args.turbo_model_path = FLUX_TURBO_PATH
    return args


# Input condition preprocessing
def pre_condition(self, ref_images, ref_tasks, ref_res, seed):
    ref_conds = []
    debug_images = []
    for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
        if ref_image is not None:
            # if ref_task == "id":
            #     if self.offload:
            #         self.facexlib_to_device(self.device)
            #     ref_image = resize_numpy_image_long(ref_image, 1024)
            #     ref_image = self.get_align_face(ref_image)
            #     if self.offload:
            #         self.facexlib_to_device(torch.device("cpu"))
            # elif ref_task != "style":
            #     if self.offload:
            #         self.ben_to_device(self.device)
            #     ref_image = self.bg_rm_model.inference(Image.fromarray(ref_image))
            #     if self.offload:
            #         self.ben_to_device(torch.device("cpu"))
            # use face_model to detect if there is a face

            if ref_task != "id":
                ref_res_ = ref_res
                ref_image_np = np.array(ref_image)
                if len(self.face_model.get(ref_image_np)) > 0:
                    ref_res_ = 768  # use higher res for face images
                ref_image = resize_numpy_image_area(ref_image_np, ref_res_ * ref_res_)
            debug_images.append(ref_image)
            ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
            ref_image = 2 * ref_image - 1.0
            ref_conds.append(
                {
                    "img": ref_image,
                    "task": ref_task,
                    "idx": idx + 1,
                }
            )
    # cleaning
    self.torch_empty_cache()
    seed = torch.Generator(device="cpu").seed() if seed == "-1" else int(seed)
    return ref_conds, debug_images, seed


@torch.inference_mode()
def generate_composite_image(
    pipe: Generator,
    prompt,
    ref_image_paths,
    width=1024,
    height=1024,
    ref_res=512,
    num_steps=12,
    guidance=4.5,
    seed="-1",
    true_cfg=1,
    cfg_start_step=0,
    cfg_end_step=0,
    neg_prompt="",
    neg_guidance=3.5,
    first_step_guidance=0,
):
    ref_tasks = []
    ref_images = []
    for ref_image_path in ref_image_paths:
        if ref_image_path.endswith(".png"):
            pil_img = Image.open(ref_image_path).convert("RGBA")
            alpha_channel = pil_img.split()[-1]
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=alpha_channel)
            pil_img = bg
        else:
            pil_img = Image.open(ref_image_path).convert("RGB")
        ref_images.append(pil_img)
        ref_tasks.append("ip")

    ref_conds, debug_images, seed = pre_condition(
        self=pipe,
        ref_images=ref_images,
        ref_tasks=ref_tasks,
        ref_res=ref_res,
        seed=seed,
    )
    image = pipe.dreamo_pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        ref_conds=ref_conds,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        true_cfg_scale=true_cfg,
        true_cfg_start_step=cfg_start_step,
        true_cfg_end_step=cfg_end_step,
        negative_prompt=neg_prompt,
        neg_guidance_scale=neg_guidance,
        first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
    ).images[0]

    return image[0]


def load_pipeline(device="cuda"):
    args = get_args_wo_parsing()
    args.device = device
    generator = Generator(
        version=args.version,
        offload=args.offload,
        quant=args.quant,
        device=device,
        no_turbo=args.no_turbo,
        dreamo_model_path=args.dreamo_model_path,
        model_path=args.model_path,
        turbo_model_path=args.turbo_model_path,
    )
    generator.dreamo_pipeline.set_progress_bar_config(disable=True)
    if hasattr(generator, "bg_rm_model"):
        del generator.bg_rm_model
        torch.cuda.empty_cache()
    if hasattr(generator, "face_helper"):
        del generator.face_helper
        torch.cuda.empty_cache()

    face_model = FaceAnalysis(
        root=FACE_MODEL_ROOT,
        providers=["CUDAExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"],
    )
    face_model.prepare(ctx_id=0 if device.startswith("cuda") else -1, det_size=(640, 640))
    generator.face_model = face_model
    return generator
