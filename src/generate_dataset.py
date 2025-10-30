import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environ_config import (
    DATASET_DIR,
    FLUX_KONTEXT_PATH,
    FLUX_PATH,
    FLUX_TURBO_PATH,
    GROUNDINGDINO_CKPT_PATH,
    NUNCHAKU_FLUX_KONTEXT_PATH,
)

PROMPT_PATH = os.path.join(DATASET_DIR, "prompts.json")
SAVE_FORMAT = "jpg"

# feel free to customize your own template and probabilities
KONTEXT_EDIT_PROMPT_TEMPLATE = [
    {
        "prefix": "Place",
        "suffix": [
            "under the morning sun, casting long shadows",
            "under the scorching midday sun.",
            "in the glow of the sunset.",
            "in rainy day light.",
            "on a sandy beach.",
            "in the warm, golden light of dawn.",
            "under the harsh, direct light of midday.",
            "look like it's captured during a fast movement.",
        ],
        "probability": 0.4,
    },
    {
        "prefix": "Make",
        "suffix": [
            "look like it's in motion.",
        ],
        "probability": 0.1,
    },
    {"prefix": "Rotate the camera 30 degrees to view slightly from the side of", "suffix": [], "probability": 0.15},
    {"prefix": "Rotate the camera 45 degrees to view from the side of", "suffix": [], "probability": 0.15},
    {"prefix": "Rotate the camera 60 degrees to view from the side of", "suffix": [], "probability": 0.1},
    {"prefix": "Rotate the camera 90 degrees to view directly from the side of", "suffix": [], "probability": 0.1},
]


def get_prompts(prompts_path, start_idx: int = 0, length=None):
    if isinstance(prompts_path, str):
        if os.path.isfile(prompts_path):
            with open(prompts_path, "r") as f:
                items = json.load(f)
                print(f"Load {len(items)} prompts from {prompts_path}")
        elif os.path.isdir(prompts_path):
            items = []
            with os.scandir(prompts_path) as it:
                for entry in sorted(it, key=lambda x: x.name):  # maintain order
                    if entry.is_file() and entry.name.endswith(".json"):
                        with open(entry.path, "r") as f:
                            file_items = json.load(f)
                            print(f"Load {len(file_items)} prompts from {entry.path}")
                            items += file_items
        else:
            items = []
    elif isinstance(prompts_path, list):
        prompts_path.sort()
        items = []
        for path in prompts_path:
            with open(path, "r") as f:
                file_items = json.load(f)
                print(f"Load {len(file_items)} prompts from {path}")
                items += file_items
    else:
        raise ValueError("prompts_path must be a string or a list of strings")

    if length is None:
        length = len(items) - start_idx
        prompts = items[start_idx:]
    else:
        prompts = items[start_idx : start_idx + length]

    print(f"Load {start_idx} to {start_idx + length} prompts from {prompts_path}")

    return prompts


def get_objects(image, prompt, model):
    from groundingdino.util.inference import annotate, predict
    from groundingdino.util.inference import load_image as dino_load_image

    from src.utils.cal import mns, normed_cxcywh_to_pixel_xyxy, size_filter

    prompt = " . ".join([p.lower() for p in prompt])
    image_source, image = dino_load_image(image)
    with torch.no_grad():
        bboxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device="cuda",
        )
    bboxes_int_xyxy = []
    for i in range(len(bboxes)):
        bbox_int_xyxy, bboxes[i] = normed_cxcywh_to_pixel_xyxy(bboxes[i], return_dtype="tensor")
        bboxes_int_xyxy.append(bbox_int_xyxy)

    # 1. filter small boxes first
    valid = size_filter(bboxes)
    bboxes = bboxes[valid]
    logits = logits[valid]
    phrases = [phrases[i] for i in range(len(phrases)) if valid[i]]
    bboxes_int_xyxy = [bboxes_int_xyxy[i] for i in range(len(bboxes_int_xyxy)) if valid[i]]

    # 2. MNS to remove redundant boxes
    keep = mns(bboxes, logits, 0.625, max_instances=8)
    bboxes = bboxes[keep]
    logits = logits[keep]
    phrases = [phrases[i] for i in keep]
    bboxes_int_xyxy = [bboxes_int_xyxy[i] for i in keep]

    annotated_img = annotate(image_source, bboxes, logits, phrases)
    return annotated_img, bboxes_int_xyxy, phrases


def generate_image(prompts, flux_model, save_dir, batch_size=16, use_turbo=True, gpu_id=None):
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"GPU {gpu_id} Generating images"):
        prompt_batch = prompts[i : i + batch_size]
        prompt = [p["prompt"] for p in prompt_batch]
        with torch.no_grad():
            imgs = flux_model(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=8 if use_turbo else 28,
            ).images
        for j, img in enumerate(imgs):
            idx = prompt_batch[j]["index"]
            img.save(f"{save_dir}/{idx:06}.{SAVE_FORMAT}")


def annotate_images(prompts, image_save_dir, prompt_save_path, dino_model=None, gpu_id=None, process_id=0):
    torch.cuda.set_device(gpu_id)
    if dino_model is None:
        from groundingdino.config import GroundingDINO_SwinT_OGC
        from groundingdino.util.inference import load_model

        dino_model = load_model(GroundingDINO_SwinT_OGC.__file__, GROUNDINGDINO_CKPT_PATH, device="cuda")

    prompt_save_path = prompt_save_path.replace(".json", f"_{process_id:02d}.json")

    print(
        f"PROCESS {process_id} on GPU {gpu_id}: {prompts[0]['index']:06d}-{prompts[-1]['index']:06d}, save to {prompt_save_path}"
    )
    for i, p in enumerate(tqdm(prompts)):
        p["reference_images"] = []
        img_path = f"{image_save_dir}/{p['index']:06d}.{SAVE_FORMAT}"
        img = Image.open(img_path)
        annotated_img, bboxes, phrases = get_objects(img, p["nouns"], dino_model)
        for idx, bbox in enumerate(bboxes):
            p["reference_images"].append({"bbox": bbox.cpu().numpy().tolist(), "index": idx, "phrase": phrases[idx]})
        annotated_img = annotated_img[..., ::-1]
        annotated_img = Image.fromarray(annotated_img)
        annotated_img.save(f"{image_save_dir}/{p['index']:06d}_annotated.{SAVE_FORMAT}")

    # save prompts with reference_images
    with open(prompt_save_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(
        f"PROCESS {process_id} on GPU {gpu_id}: Finished annotating images from {prompts[0]['index']:06d}-{prompts[-1]['index']:06d}, saved to {prompt_save_path}"
    )


def parallel_annotate_images(
    prompts,
    image_save_dir,
    prompt_save_path,
    dino_model=None,
    num_processes_per_gpu=1,
):
    import multiprocessing

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")

    gpu_count = torch.cuda.device_count()
    num_processes = gpu_count * num_processes_per_gpu

    prompt_chunks = [[] for _ in range(num_processes)]
    for i in range(len(prompts)):
        prompt_chunks[i % num_processes].append(prompts[i])

    processes = []
    for process_idx in range(num_processes):
        prompts_chunk = prompt_chunks[process_idx]
        gpu_idx = process_idx % gpu_count
        if len(prompts_chunk) == 0:
            continue
        p = multiprocessing.Process(
            target=annotate_images,
            args=(
                prompts_chunk,
                image_save_dir,
                prompt_save_path,
                dino_model,
                gpu_idx,
                process_idx,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # gather all prompt files
    all_prompts = []
    for process_idx in range(num_processes):
        prompt_file = prompt_save_path.replace(".json", f"_{process_idx:02d}.json")
        with open(prompt_file, "r") as f:
            process_prompts = json.load(f)
            all_prompts += process_prompts
        os.remove(prompt_file)

    with open(prompt_save_path, "w") as f:
        json.dump(all_prompts, f, indent=2)


def generate_images_work_process(prompts_chunk, image_save_dir, batch_size=16, gpu_id=None, use_turbo=True):
    assert gpu_id is not None
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    from diffusers import FluxPipeline

    model = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16).to(device)
    if use_turbo:
        model.load_lora_weights(FLUX_TURBO_PATH)
        model.fuse_lora()
    model.set_progress_bar_config(disable=True)
    print(
        f"GPU {gpu_id} start generating images from index {prompts_chunk[0]['index']} to {prompts_chunk[-1]['index']}\n"
        f"GPU {gpu_id}: The first prompt in this chunk is: {prompts_chunk[0]['prompt'] if prompts_chunk else 'No prompts'}\n"
        f"GPU {gpu_id}: The last prompt in this chunk is: {prompts_chunk[-1]['prompt'] if prompts_chunk else 'No prompts'}\n"
    )
    generate_image(
        prompts=prompts_chunk,
        flux_model=model,
        save_dir=image_save_dir,
        batch_size=batch_size,
        use_turbo=use_turbo,
        gpu_id=gpu_id,
    )


def parallel_generate_images(
    raw_prompts,
    image_save_dir,
    batch_size=16,
    num_gpus=None,
):
    import multiprocessing

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")

    print(f"Using {num_gpus} GPUs for parallel image generation")

    chunk_size = len(raw_prompts) // num_gpus
    prompt_chunks = []

    for i in range(num_gpus):
        chunk_start_idx = chunk_size * i
        chunk_end_idx = chunk_size * (i + 1) if i < num_gpus - 1 else len(raw_prompts)
        prompt_chunks.append(raw_prompts[chunk_start_idx:chunk_end_idx])

    processes = []
    for gpu_idx in range(num_gpus):
        prompts_chunk = prompt_chunks[gpu_idx]
        if len(prompts_chunk) == 0:
            continue
        p = multiprocessing.Process(
            target=generate_images_work_process,
            args=(prompts_chunk, image_save_dir, batch_size, gpu_idx),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def repaint_with_kontext_work_process(prompts_chunk, image_dir, image_save_dir, gpu_id=0, max_batch_size=8):
    assert gpu_id is not None
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    from nunchaku import NunchakuFluxTransformer2dModel

    from third_party.flux.pipeline_flux_kontext import FluxKontextPipeline

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(NUNCHAKU_FLUX_KONTEXT_PATH, device=device).to(device)
    pipe = FluxKontextPipeline.from_pretrained(
        FLUX_KONTEXT_PATH, transformer=transformer, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(
        f"GPU {gpu_id} start from size {prompts_chunk[0]['size']}(num: {len(prompts_chunk[0]['segments'])}) to {prompts_chunk[-1]['size']}(num: {len(prompts_chunk[-1]['segments'])}), "
        f"indices from {prompts_chunk[0]['segments'][0]['index']} to {prompts_chunk[-1]['segments'][-1]['index']}"
    )

    def generate_prompt_with_template(instance_caption):
        import random

        if not instance_caption.lower().startswith("the "):
            instance_caption = "the " + instance_caption

        remove_obstacle = f"Keep only {instance_caption} and remove any other objects."
        template = random.choices(
            KONTEXT_EDIT_PROMPT_TEMPLATE, weights=[t["probability"] for t in KONTEXT_EDIT_PROMPT_TEMPLATE]
        )[0]
        prefix = template["prefix"]
        suffix = random.choice(template["suffix"]) if template["suffix"] else ""
        return (
            f"{remove_obstacle} {prefix} {instance_caption}."
            if not suffix
            else f"{remove_obstacle} {prefix} {instance_caption} {suffix}".strip()
        )

    for p in prompts_chunk:
        target_size = p["size"]
        segments = p["segments"]

        for i in tqdm(range(0, len(segments), max_batch_size), desc=f"GPU {gpu_id}, size {target_size}"):
            batch_segments = segments[i : i + max_batch_size]
            batch_images = []
            batch_prompts = []
            for segment in batch_segments:
                img_path = f"{image_dir}/{segment['index']:06d}.{SAVE_FORMAT}"
                img = Image.open(img_path)
                bbox = segment["bbox"]
                ref_img = img.crop(bbox)
                ref_img = ref_img.resize((target_size[0], target_size[1])).convert("RGB")
                prompt = generate_prompt_with_template(segment["phrase"])
                batch_images.append(ref_img)
                batch_prompts.append(prompt)

            output_images = pipe(
                image=batch_images,
                prompt=batch_prompts,
                width=target_size[0],
                height=target_size[1],
                num_inference_steps=15,
                max_area=None,
                _auto_resize=False,
            ).images

            for j, segment in enumerate(batch_segments):
                save_image_path = f"{image_save_dir}/{segment['index']:06d}_{segment['segment_index']}.{SAVE_FORMAT}"
                output_image = output_images[j]
                output_image.save(save_image_path)


def parallel_repaint_with_kontext(raw_prompts, image_dir, image_save_dir, num_gpus=None):
    import multiprocessing

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")
    print(f"Using {num_gpus} GPUs for parallel repainting with Kontext")

    prompt_chunks = [[] for _ in range(num_gpus)]
    for i in range(len(raw_prompts)):
        segments_num = len(raw_prompts[i]["segments"])
        segments_per_gpu = (segments_num + num_gpus - 1) // num_gpus
        for gpu_idx in range(num_gpus):
            start_segment_idx = gpu_idx * segments_per_gpu
            end_segment_idx = min((gpu_idx + 1) * segments_per_gpu, segments_num)
            if start_segment_idx < end_segment_idx:
                prompt_chunk = {
                    "size": raw_prompts[i]["size"],
                    "segments": raw_prompts[i]["segments"][start_segment_idx:end_segment_idx],
                }
                prompt_chunks[gpu_idx].append(prompt_chunk)

    processes = []
    for gpu_idx in range(num_gpus):
        prompts_chunk = prompt_chunks[gpu_idx]
        if len(prompts_chunk) == 0:
            continue
        p = multiprocessing.Process(
            target=repaint_with_kontext_work_process,
            args=(prompts_chunk, image_dir, image_save_dir, gpu_idx),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def collect_segments_size_info(raw_prompts, max_area=1024**2):
    size_list = []
    for p in tqdm(raw_prompts):
        for ref in p["reference_images"]:
            bbox = ref["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)
            multiple_of = 16
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of

            ref_info = {
                "index": p["index"],
                "segment_index": ref["index"],
                "phrase": ref["phrase"],
                "bbox": ref["bbox"],
            }
            found = False
            for size_info in size_list:
                if size_info["size"] == (width, height):
                    size_info["segments"].append(ref_info)
                    found = True
                    break
            if not found:
                size_list.append({"size": (width, height), "segments": [ref_info]})

    # sort by len(segments) ascending
    sorted_size_list = sorted(size_list, key=lambda x: len(x["segments"]), reverse=False)
    print("Collected segment sizes and their counts:")
    for size_info in sorted_size_list:
        size = size_info["size"]
        segments_count = len(size_info["segments"])
        print(f"Size: {size}, Segments Count: {segments_count}")
    return sorted_size_list


def crop_images(
    raw_prompts,
    image_dir,
    crop_image_save_dir,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run parallel crop images")

    for p in tqdm(raw_prompts):
        img_path = f"{image_dir}/{p['index']:06d}.{SAVE_FORMAT}"
        img = Image.open(img_path)
        for ref in p["reference_images"]:
            bbox = ref["bbox"]
            ref_img = img.crop(bbox)
            save_image_path = f"{crop_image_save_dir}/{p['index']:06d}_{ref['index']}.{SAVE_FORMAT}"
            ref_img.save(save_image_path)


def gen_instance_bboxes_processor(
    data: list,
    image_root: str,
    process_id: int = 0,
    gpu_id: int = 0,
):
    torch.cuda.set_device(gpu_id)
    from groundingdino.config import GroundingDINO_SwinT_OGC
    from groundingdino.util.inference import load_image, load_model, predict

    from src.utils.cal import normed_cxcywh_to_pixel_xyxy

    dino_model = load_model(
        model_config_path=GroundingDINO_SwinT_OGC.__file__,
        model_checkpoint_path=GROUNDINGDINO_CKPT_PATH,
        device=f"cuda:{gpu_id}",
    )
    dino_model.requires_grad_(False).eval()

    print(f"Process {process_id} started with data {data[0]['index']} to {data[-1]['index']}")
    for item in tqdm(data, desc=f"Process {process_id} on GPU {gpu_id}"):
        reference_images = item["reference_images"]
        for ref_idx, ref in enumerate(reference_images):
            ref_image_path = os.path.join(image_root, f"{item['index']:06d}_{ref['index']}.{SAVE_FORMAT}")
            if not os.path.exists(ref_image_path):
                raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
            ref_image_source, ref_image = load_image(ref_image_path)
            prompt = ref["phrase"]
            if not prompt.endswith("."):
                prompt += " ."
            with torch.inference_mode():
                bboxes = predict(
                    model=dino_model,
                    image=ref_image,
                    caption=prompt,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device="cuda",
                )[0]
            if len(bboxes) == 0:
                print(f"No bounding boxes found for {ref_image_path} with prompt '{prompt}'")
                continue
            else:
                bbox = normed_cxcywh_to_pixel_xyxy(
                    bboxes[0], width=ref_image_source.shape[1], height=ref_image_source.shape[0], return_dtype="list"
                )[0]
                ref["valid_bbox"] = bbox

    with open(f"temp_bbox_data_{process_id:02d}.json", "w") as f:
        json.dump(data, f, indent=2)


def parallel_gen_instance_bboxes(
    data: list,
    image_root: str,
    data_save_path: str,
    num_processes_per_gpu: int = 1,
):
    num_gpus = torch.cuda.device_count()
    num_processes = num_gpus * num_processes_per_gpu
    data_chunks = [[] for _ in range(num_processes)]
    for i, item in enumerate(data):
        data_chunks[i % num_processes].append(item)

    args_list = [(data_chunks[i], image_root, i, i % num_gpus) for i in range(num_processes)]
    import multiprocessing

    processes = []
    for args in args_list:
        p = multiprocessing.Process(target=gen_instance_bboxes_processor, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # gather all data
    all_data = []
    for i in range(num_processes):
        with open(f"temp_bbox_data_{i:02d}.json", "r") as f:
            chunk_data = json.load(f)
            all_data.extend(chunk_data)
        os.remove(f"temp_bbox_data_{i:02d}.json")

    # sort by index to maintain order
    all_data.sort(key=lambda x: x["index"])

    # save final results
    with open(data_save_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Instance bounding boxes saved to {data_save_path}")


def filter_data(
    dataset_root: str,
    masked_instance_image_root: str,
    masked_repainted_image_root: str,
    max_num_instances=None,
    min_num_instances=1,
    min_instance_size=None,
    min_mask_ratio=None,
    min_fit_ratio=None,
):
    def get_valid_mask_region_np(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if mask.ndim != 2:
            mask = mask.squeeze()
        assert mask.ndim == 2, "Mask must be 2D array (H,W)"

        if not np.any(mask):
            return None

        # get non-zero rows and columns
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        y1, y2 = y_indices[[0, -1]] if len(y_indices) > 0 else (0, mask.shape[0])
        x1, x2 = x_indices[[0, -1]] if len(x_indices) > 0 else (0, mask.shape[1])

        return int(x1), int(y1), int(x2), int(y2)

    with open(os.path.join(dataset_root, "prompts_with_valid_bboxes.json")) as f:
        orig_data = json.load(f)

    filtered_data = []
    image_0_path = os.path.join(dataset_root, "data", f"000000.{SAVE_FORMAT}")
    main_image = Image.open(image_0_path).convert("RGB")
    orig_area = main_image.width * main_image.height
    for item in tqdm(orig_data, desc="Filtering data"):
        if len(item["reference_images"]) < min_num_instances:
            continue
        reference_images = item["reference_images"]
        if min_instance_size is not None:
            reference_images = [
                reference_image
                for reference_image in reference_images
                if (reference_image["bbox"][2] - reference_image["bbox"][0])
                * (reference_image["bbox"][3] - reference_image["bbox"][1])
                >= (orig_area * min_instance_size)
            ]
            if len(reference_images) < min_num_instances:
                continue

        exclude_list = []
        for ref_idx, ref in enumerate(reference_images):
            repainted_mask_path = os.path.join(
                masked_instance_image_root,
                f"{item['index']:06d}_{ref['index']}_masked.png"
                if SAVE_FORMAT == "png"
                else (
                    f"{item['index']:06d}_{ref['index']}_mask.jpg"
                    if os.path.exists(
                        os.path.join(masked_instance_image_root, f"{item['index']:06d}_{ref['index']}_mask.jpg")
                    )
                    else f"{item['index']:06d}_{ref['index']}_masked.png"  # may in 1-bit png format
                ),
            )
            repainted_ref_mask_path = os.path.join(
                masked_repainted_image_root,
                f"{item['index']:06d}_{ref['index']}_masked.png"
                if SAVE_FORMAT == "png"
                else (
                    f"{item['index']:06d}_{ref['index']}_mask.jpg"
                    if os.path.exists(
                        os.path.join(masked_repainted_image_root, f"{item['index']:06d}_{ref['index']}_mask.jpg")
                    )
                    else f"{item['index']:06d}_{ref['index']}_masked.png"  # may in 1-bit png format
                ),
            )
            if min_mask_ratio is not None or min_fit_ratio is not None:
                if SAVE_FORMAT == "png":
                    masked_image = Image.open(repainted_mask_path).convert("RGBA")
                    mask_image_np = np.array(masked_image.split()[-1])
                    repainted_masked_image = Image.open(repainted_ref_mask_path).convert("RGBA")
                    repainted_mask_image_np = np.array(repainted_masked_image.split()[-1])
                else:
                    mask_image = Image.open(repainted_mask_path).convert("L")
                    mask_image_np = np.array(mask_image)
                    repainted_mask_image = Image.open(repainted_ref_mask_path).convert("L")
                    repainted_mask_image_np = np.array(repainted_mask_image)
            if min_mask_ratio is not None:
                if (
                    np.sum(mask_image_np > 0) / mask_image_np.size < min_mask_ratio
                    or np.sum(repainted_mask_image_np > 0) / repainted_mask_image_np.size < min_mask_ratio
                ):
                    exclude_list.append(ref_idx)
                    continue
            if min_fit_ratio is not None:
                mask_valid_region = get_valid_mask_region_np(mask_image_np)
                repainted_mask_valid_region = get_valid_mask_region_np(repainted_mask_image_np)
                if mask_valid_region is None or repainted_mask_valid_region is None:
                    exclude_list.append(ref_idx)
                    continue
                valid_height, valid_width = (
                    mask_valid_region[3] - mask_valid_region[1],
                    mask_valid_region[2] - mask_valid_region[0],
                )
                bbox_height, bbox_width = ref["bbox"][3] - ref["bbox"][1], ref["bbox"][2] - ref["bbox"][0]
                if valid_height / bbox_height < min_fit_ratio or valid_width / bbox_width < min_fit_ratio:
                    exclude_list.append(ref_idx)
                    continue
                if (repainted_bbox := ref.get("valid_bbox")) is not None:
                    repainted_valid_height, repainted_valid_width = (
                        repainted_mask_valid_region[3] - repainted_mask_valid_region[1],
                        repainted_mask_valid_region[2] - repainted_mask_valid_region[0],
                    )
                    repainted_bbox_height, repainted_bbox_width = (
                        repainted_bbox[3] - repainted_bbox[1],
                        repainted_bbox[2] - repainted_bbox[0],
                    )
                    if (
                        repainted_valid_height / repainted_bbox_height < min_fit_ratio
                        or repainted_valid_width / repainted_bbox_width < min_fit_ratio
                    ):
                        exclude_list.append(ref_idx)
                        continue

        if len(exclude_list) > 0:
            reference_images = [ref for i, ref in enumerate(reference_images) if i not in exclude_list]
            if len(reference_images) < min_num_instances:
                continue

        if max_num_instances is not None:
            reference_images = sorted(
                reference_images,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
                reverse=True,
            )
            reference_images = reference_images[:max_num_instances]
        item["reference_images"] = reference_images
        filtered_data.append(item)

    print(f"Original data: {len(orig_data)}, Filtered data: {len(filtered_data)}")
    return filtered_data


def segment_images(packed_prompts, image_save_dir, masked_image_save_dir, batch_size=64, gpu_id=None):
    DERIS_PATH = PROJECT_ROOT / "third_party" / "DeRIS"
    sys.path.insert(0, str(DERIS_PATH))

    current_dir = os.getcwd()
    os.chdir(DERIS_PATH)  # switch to DeRIS directory to avoid path issues

    from third_party.DeRIS.parallel_deris import ParallelDeris, get_deris_args_wo_parsing

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda" if torch.cuda.is_available() else "cpu")

    args = get_deris_args_wo_parsing()
    args.device = device
    args.config = os.path.join(DERIS_PATH, args.config)
    segmentation_model = ParallelDeris(args)

    print(
        f"GPU {gpu_id}: Start segmenting images from {image_save_dir} to {masked_image_save_dir}, index {packed_prompts[0]['image_idx']} to {packed_prompts[-1]['image_idx']}"
    )

    for batch in tqdm(range(0, len(packed_prompts), batch_size)):
        batch_items = packed_prompts[batch : batch + batch_size]
        images = []
        expressions = []
        for item in batch_items:
            image_path = f"{image_save_dir}/{item['image_idx']:06d}_{item['segment_idx']}.{SAVE_FORMAT}"
            images.append(image_path)
            expressions.append(item["phrase"])

        if len(images) == 0:
            continue

        with torch.inference_mode():
            pred_masks = segmentation_model.inference_batch(
                image_paths=images,
                expressions=expressions,
            )

        for i, (image, mask) in enumerate(zip(images, pred_masks)):
            save_path = f"{masked_image_save_dir}/{batch_items[i]['image_idx']:06d}_{batch_items[i]['segment_idx']}"
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            if SAVE_FORMAT == "png":
                ref_image = Image.open(image).convert("RGBA")
                ref_image.putalpha(mask_pil.convert("L"))
                ref_image.save(f"{save_path}_masked.png")
            else:
                # mask_pil.save(f"{save_path}_mask.jpg")
                mask_pil.convert("1").save(f"{save_path}_mask.png")  # 1 bit png mask, less space
                ref_image = Image.open(image).convert("RGB")
                white_bg = Image.new("RGB", ref_image.size, (255, 255, 255))
                masked_image = Image.composite(ref_image, white_bg, mask_pil)
                masked_image.save(f"{save_path}_masked.jpg")

    # switch back to original directory
    os.chdir(current_dir)


def parallel_segment_images(packed_prompts, image_save_dir, masked_image_save_dir, batch_size=64, num_gpus=None):
    from multiprocessing import Process

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")
    print(f"Using {num_gpus} GPUs for parallel repainting")

    # Split the packed prompts into chunks for each GPU
    chunk_size = len(packed_prompts) // num_gpus
    prompt_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_gpus - 1 else len(packed_prompts)
        prompt_chunks.append(packed_prompts[start_idx:end_idx])
    print(f"Split prompts into {len(prompt_chunks)} chunks for parallel processing")

    processes = []
    for gpu_idx in range(num_gpus):
        p = Process(
            target=segment_images,
            args=(prompt_chunks[gpu_idx], image_save_dir, masked_image_save_dir, batch_size, gpu_idx),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def filter_exists_prompts(raw_prompts, target_dir, target_dir_2=None, task="generate_images"):
    filtered_prompts = []
    if task == "generate_images":
        for p in tqdm(raw_prompts):
            if not os.path.exists(f"{target_dir}/{p['index']:06d}.{SAVE_FORMAT}"):
                filtered_prompts.append(p)
    elif task == "annotate_images":
        for p in tqdm(raw_prompts):
            if not os.path.exists(f"{target_dir}/{p['index']:06d}.{SAVE_FORMAT}"):
                print(f"Warning: Image {target_dir}/{p['index']:06d}.{SAVE_FORMAT} does not exist for annotation!")
                continue
            if not os.path.exists(f"{target_dir_2}/{p['index']:06d}_annotated.{SAVE_FORMAT}"):
                filtered_prompts.append(p)
    elif task == "repaint_with_kontext" or task == "repaint_with_redux":
        total_instances = 0
        filtered_instances = 0
        for p in tqdm(raw_prompts):
            num_instances = len(p["reference_images"])
            if num_instances == 0:
                continue
            new_p = p.copy()
            new_p["reference_images"] = []
            for ref in p["reference_images"]:
                total_instances += 1
                save_image_path = f"{target_dir}/{p['index']:06d}_{ref['index']}.{SAVE_FORMAT}"
                if not os.path.exists(save_image_path):
                    new_p["reference_images"].append(ref)
                    filtered_instances += 1
            if len(new_p["reference_images"]) > 0:
                filtered_prompts.append(new_p)
    elif task == "segment_instance_images" or task == "segment_repainted_images":
        assert target_dir_2 is not None, "masked_instance_dir must be provided for segmentation task"
        skipped_count = 0
        missing_count = 0
        remaining_count = 0
        for p in tqdm(raw_prompts):
            skipped_instance_count = 0
            missing_instance_count = 0
            for ref in p["reference_images"]:
                from_path = os.path.join(target_dir, f"{p['index']:06d}_{ref['index']}.{SAVE_FORMAT}")
                if not os.path.exists(from_path):
                    missing_instance_count += 1
                    continue
                if os.path.exists(os.path.join(target_dir_2, f"{p['index']:06d}_{ref['index']}_masked.{SAVE_FORMAT}")):
                    skipped_instance_count += 1
                    continue
                filtered_prompts.append(
                    {
                        "image_idx": p["index"],
                        "segment_idx": ref["index"],
                        "phrase": ref["phrase"],
                        "bbox": ref["bbox"],
                    }
                )
            if skipped_instance_count == len(p["reference_images"]) and len(p["reference_images"]) > 0:
                skipped_count += 1
            elif missing_instance_count == len(p["reference_images"]) and len(p["reference_images"]) > 0:
                missing_count += 1
            elif skipped_instance_count + missing_instance_count < len(p["reference_images"]):
                remaining_count += 1

        print(
            f"Packed {len(filtered_prompts)} instances for segmentation"
            f"({skipped_count} skipped, {missing_count} missing, {remaining_count} remaining)"
        )
    print(
        f"Target dir: {target_dir}, filtered: {len(raw_prompts) - len(filtered_prompts)}, remaining: {len(filtered_prompts)}"
    )
    return filtered_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset for MIG-Flux")
    parser.add_argument(
        "--task",
        type=str,
        default="generate_images",
        choices=[
            "generate_images",
            "annotate_images",
            "crop_images",
            "repaint_with_kontext",
            "get_valid_bboxes",
            "segment_instance_images",
            "segment_repainted_images",
            "filter_prompts",
        ],
        help="Task to perform",
    )
    parser.add_argument(
        "--disable_skip", action="store_false", dest="skip_if_exists", help="Disable skipping existing files"
    )
    prompt_save_path = os.path.join(DATASET_DIR, "prompts.json")
    image_save_dir = os.path.join(DATASET_DIR, "data")

    args = parser.parse_args()

    # Parse command line arguments
    if args.task == "generate_images":
        os.makedirs(image_save_dir, exist_ok=True)
        raw_prompts = get_prompts(PROMPT_PATH)
        if args.skip_if_exists:
            raw_prompts = filter_exists_prompts(raw_prompts, image_save_dir, task=args.task)
        if len(raw_prompts) == 0:
            print("No prompts to process, exiting.")
            exit(0)
        parallel_generate_images(
            raw_prompts=raw_prompts,
            image_save_dir=image_save_dir,
            batch_size=2,
        )

    elif args.task == "annotate_images":
        raw_prompts = get_prompts(PROMPT_PATH)
        if args.skip_if_exists:
            raw_prompts = filter_exists_prompts(raw_prompts, image_save_dir, image_save_dir, task=args.task)
        if len(raw_prompts) == 0:
            print("No prompts to process, exiting.")
            exit(0)
        parallel_annotate_images(
            prompts=raw_prompts,
            image_save_dir=image_save_dir,
            prompt_save_path=prompt_save_path,
            num_processes_per_gpu=4,  # Adjust based on your CPU cores
        )

    elif args.task == "crop_images":
        cropped_image_save_dir = os.path.join(DATASET_DIR, "instance_data")
        if not os.path.exists(cropped_image_save_dir):
            os.makedirs(cropped_image_save_dir)
            print(f"Create directory {cropped_image_save_dir}")
        raw_prompts = get_prompts(prompt_save_path)
        crop_images(raw_prompts=raw_prompts, image_dir=image_save_dir, crop_image_save_dir=cropped_image_save_dir)

    elif args.task == "repaint_with_kontext":
        kontext_image_save_dir = os.path.join(DATASET_DIR, "kontext_data")
        if not os.path.exists(kontext_image_save_dir):
            os.makedirs(kontext_image_save_dir)
            print(f"Create directory {kontext_image_save_dir}")
        raw_prompts = get_prompts(prompt_save_path)
        if args.skip_if_exists:
            raw_prompts = filter_exists_prompts(raw_prompts, kontext_image_save_dir, task=args.task)
        if len(raw_prompts) == 0:
            print("No prompts to process, exiting.")
            exit(0)
        raw_prompts = collect_segments_size_info(raw_prompts)
        parallel_repaint_with_kontext(
            raw_prompts=raw_prompts,
            image_dir=image_save_dir,
            image_save_dir=kontext_image_save_dir,
        )

    elif args.task == "get_valid_bboxes":
        kontext_image_save_dir = os.path.join(DATASET_DIR, "kontext_data")
        data_save_path = os.path.join(DATASET_DIR, "prompts_with_valid_bboxes.json")
        assert os.path.exists(kontext_image_save_dir), f"Directory {kontext_image_save_dir} does not exist."
        raw_prompts = get_prompts(prompt_save_path)
        if len(raw_prompts) == 0:
            print("No prompts to process, exiting.")
            exit(0)
        parallel_gen_instance_bboxes(
            data=raw_prompts, image_root=kontext_image_save_dir, data_save_path=data_save_path, num_processes_per_gpu=4
        )

    elif args.task == "segment_instance_images":
        cropped_image_save_dir = os.path.join(DATASET_DIR, "instance_data")
        cropped_masked_image_save_dir = os.path.join(DATASET_DIR, "masked_instance_data")
        if not os.path.exists(cropped_masked_image_save_dir):
            os.makedirs(cropped_masked_image_save_dir)
            print(f"Create directory {cropped_masked_image_save_dir}")
        raw_prompts = get_prompts(prompt_save_path)
        packed_prompts = filter_exists_prompts(
            raw_prompts, cropped_image_save_dir, target_dir_2=cropped_masked_image_save_dir, task=args.task
        )
        if len(packed_prompts) == 0:
            print("No prompts to process, exiting.")
            exit(0)
        parallel_segment_images(
            packed_prompts=packed_prompts,
            image_save_dir=cropped_image_save_dir,
            masked_image_save_dir=cropped_masked_image_save_dir,
            batch_size=128,
        )

    elif args.task == "segment_repainted_images":
        kontext_image_save_dir = os.path.join(DATASET_DIR, "kontext_data")
        kontext_masked_image_save_dir = os.path.join(DATASET_DIR, "masked_kontext_data")
        if not os.path.exists(kontext_masked_image_save_dir):
            os.makedirs(kontext_masked_image_save_dir)
            print(f"Create directory {kontext_masked_image_save_dir}")
        raw_prompts = get_prompts(prompt_save_path)
        packed_prompts = filter_exists_prompts(
            raw_prompts, kontext_image_save_dir, target_dir_2=kontext_masked_image_save_dir, task=args.task
        )
        if len(packed_prompts) == 0:
            print("No prompts to process, exiting.")
            exit(0)
        parallel_segment_images(
            packed_prompts=packed_prompts,
            image_save_dir=kontext_image_save_dir,
            masked_image_save_dir=kontext_masked_image_save_dir,
            batch_size=128,
        )

    elif args.task == "filter_prompts":
        filtered_prompts_save_path = os.path.join(DATASET_DIR, "filtered_prompts.json")
        raw_prompts = get_prompts(prompt_save_path)
        filtered_prompts = filter_data(
            dataset_root=DATASET_DIR,
            masked_instance_image_root=os.path.join(DATASET_DIR, "masked_instance_data"),
            masked_repainted_image_root=os.path.join(DATASET_DIR, "masked_kontext_data"),
            max_num_instances=5,
            min_num_instances=2,
            min_instance_size=0.01,
            min_mask_ratio=None,
            min_fit_ratio=0.6,
        )
        with open(filtered_prompts_save_path, "w") as f:
            json.dump(filtered_prompts, f, indent=2)
        print(f"Filtered prompts saved to {filtered_prompts_save_path}")
