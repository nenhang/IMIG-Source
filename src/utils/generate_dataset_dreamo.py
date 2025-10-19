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
def generate_image(
    generator: Generator,
    ref_images,
    ref_tasks,
    prompt,
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
    ref_conds, debug_images, seed = pre_condition(
        self=generator,
        ref_images=ref_images,
        ref_tasks=ref_tasks,
        ref_res=ref_res,
        seed=seed,
    )
    # print(prompt, seed)

    # print("start dreamo_pipeline... ")
    image = generator.dreamo_pipeline(
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

    return image, debug_images, seed


def generate_dataset_images(
    reference_image_dir: str,
    dataset_save_path: str,
    items: list,
    gpu_id: int = 0,
):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"CUDA {gpu_id}: initializing model on device {device}")

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

    # delete generator.bg_rm_model to save memory
    if hasattr(generator, "bg_rm_model"):
        del generator.bg_rm_model
        torch.cuda.empty_cache()

    if hasattr(generator, "face_helper"):
        del generator.face_helper
        torch.cuda.empty_cache()

    # add facemodel to generator
    face_model = FaceAnalysis(
        root=FACE_MODEL_ROOT, providers=["CUDAExecutionProvider"] if gpu_id >= 0 else ["CPUExecutionProvider"]
    )
    face_model.prepare(ctx_id=int(gpu_id), det_size=(640, 640))
    generator.face_model = face_model

    for item in tqdm(items):
        ref_tasks = []
        ref_images = []
        for i in range(len(item["instance"])):
            ref_image_np = np.array(
                Image.open(os.path.join(reference_image_dir, f"{item['index']:06d}_{i}_masked.png")).convert("RGBA")
            )
            alpha_channel = ref_image_np[:, :, 3] / 255.0
            for c in range(3):
                ref_image_np[:, :, c] = ref_image_np[:, :, c] * alpha_channel + 255 * (1 - alpha_channel)
            ref_image_np = ref_image_np[:, :, :3]
            ref_images.append(ref_image_np)
            ref_tasks.append("ip")

        prompt = item["prompt"]
        image = generate_image(
            generator=generator,
            ref_images=ref_images,
            ref_tasks=ref_tasks,
            prompt=prompt,
        )[0]
        image.save(os.path.join(dataset_save_path, f"{item['index']:06d}.png"))


def parallel_generate_dataset_images(
    reference_image_dir: str,
    dataset_save_path: str,
    items: list,
):
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")
    if num_gpus == 0:
        print("No GPU found, exiting...")
        return

    import multiprocessing as mp

    # set spawn method to 'spawn' to avoid issues on some platforms
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=num_gpus)
    chunk_size = (len(items) + num_gpus - 1) // num_gpus
    processes = []

    # 无需返回值
    for i in range(num_gpus):
        chunk = items[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            p = pool.apply_async(generate_dataset_images, args=(reference_image_dir, dataset_save_path, chunk, i))
            processes.append(p)
    for p in processes:
        p.get()

    pool.close()
    pool.join()


def filter_processed_images(items):
    processed_images = set(os.listdir(dataset_save_path))
    filtered_items = []
    for item in tqdm(items, desc="Filtering processed images"):
        item_index = f"{item['index']:06d}"
        has_references = True
        if f"{item_index}.png" not in processed_images:
            for i, ref in enumerate(item["instance_prompt"]):
                ref_image_path = os.path.join(reference_image_dir, f"{item['index']:06d}_{i}_masked.png")
                if not os.path.exists(ref_image_path):
                    has_references = False
                    break
                ref_image_mask = Image.open(ref_image_path).convert("RGBA").split()[-1]
                ref_image = np.array(ref_image_mask)
                if np.all(ref_image == 0):
                    has_references = False
                    break
            if has_references:
                filtered_items.append(item)
        else:
            continue

    return filtered_items


if __name__ == "__main__":
    dataset_save_path = "/root/autodl-tmp/mig-dataset/mig-flux-composite-dataset/composite_images"
    reference_image_dir = "/root/autodl-tmp/mig-dataset/mig-flux-composite-dataset/cropped_masks"

    dataset_path = "/root/autodl-tmp/mig-dataset/mig-flux-composite-dataset/prompts.json"
    with open(dataset_path) as f:
        bench_items = json.load(f)
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path, exist_ok=True)
    items = filter_processed_images(bench_items)  # for test, limit to 10000
    print(f"Filtered items: {len(items)} out of {len(bench_items)}")
    parallel_generate_dataset_images(
        reference_image_dir=reference_image_dir, dataset_save_path=dataset_save_path, items=items
    )
    print("All tasks completed.")
