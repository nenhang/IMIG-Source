import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environ_config import (
    DATASET_DIR_2,
    DINO_MODEL_ROOT,
    FACE_MODEL_ROOT,
    FLUX_PATH,
    FLUX_TURBO_PATH,
    GROUNDINGDINO_CKPT_PATH,
)
from src.utils.cal import normed_cxcywh_to_pixel_xyxy


def generate_images_work_process(prompts, image_save_dir, batch_size=16, gpu_id=None, use_turbo=True):
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
        f"GPU {gpu_id} begin processing {len(prompts)} prompts from index {prompts[0]['image_index']} to {prompts[-1]['image_index']}"
    )
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"GPU {gpu_id} processing"):
        prompt_batch = prompts[i : i + batch_size]
        prompt = [p["prompt"] for p in prompt_batch]
        with torch.no_grad():
            imgs = model(
                prompt=prompt,
                height=768,
                width=768,
                num_inference_steps=12 if use_turbo else 28,
            ).images
        for j, img in enumerate(imgs):
            file_name = f"{prompt_batch[j]['image_index']:06d}_{prompt_batch[j]['reference_index']}.png"
            img.save(os.path.join(image_save_dir, file_name))


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

    chunk_size = len(raw_prompts) // num_gpus + 1
    prompts_chunks = [raw_prompts[i : i + chunk_size] for i in range(0, len(raw_prompts), chunk_size)]

    processes = []
    for i in range(len(prompts_chunks)):
        prompts = prompts_chunks[i]
        if not prompts:
            continue
        p = multiprocessing.Process(
            target=generate_images_work_process,
            args=(prompts, image_save_dir, batch_size, i, True),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def get_objects(image, prompt, model):
    from groundingdino.util.inference import load_image as dino_load_image
    from groundingdino.util.inference import predict

    image_source, image = dino_load_image(image)
    with torch.no_grad():
        bboxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt + " .",
            box_threshold=0.35,
            text_threshold=0.25,
            device="cuda",
        )
    if len(bboxes) == 0:
        return None
    elif len(bboxes) == 1:
        return bboxes[0]
    else:
        sorted_indices = torch.argsort(logits, descending=True)
        bboxes = bboxes[sorted_indices]
        return bboxes[0]


def crop_images(items, image_dir, output_dir, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    from groundingdino.config import GroundingDINO_SwinT_OGC
    from groundingdino.util.inference import load_model

    dino_model = load_model(GroundingDINO_SwinT_OGC.__file__, GROUNDINGDINO_CKPT_PATH, device="cuda")
    for item in tqdm(items, desc=f"GPU {gpu_id} cropping images"):
        image_path = os.path.join(image_dir, f"{item['image_index']:06d}_{item['reference_index']}.png")
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        phrase = item["prompt"]
        bbox = get_objects(image, phrase, dino_model)
        if bbox is None:
            print(f"No objects found for {phrase} in image {image_path}, copying original image.")
            output_path = os.path.join(output_dir, f"{item['image_index']:06d}_{item['reference_index']}.png")
            image.save(output_path)
            continue
        bbox = normed_cxcywh_to_pixel_xyxy(
            bbox,
            width=image_width,
            height=image_height,
            return_dtype="tuple",
        )[0]
        cropped_image = image.crop(bbox)
        output_path = os.path.join(output_dir, f"{item['image_index']:06d}_{item['reference_index']}.png")
        cropped_image.save(output_path)


def parallel_crop_images(
    items,
    image_dir,
    output_dir,
    num_processes=None,
):
    import multiprocessing

    num_gpus = torch.cuda.device_count()
    num_processes = num_processes if num_processes is not None else num_gpus

    chunk_size = len(items) // num_processes + 1
    items_chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")

    print(f"Using {num_processes} processes for parallel image cropping")

    processes = []
    for i in range(len(items_chunks)):
        gpu_id = i % num_gpus
        items_chunk = items_chunks[i]
        if not items_chunk:
            continue
        p = multiprocessing.Process(
            target=crop_images,
            args=(items_chunk, image_dir, output_dir, gpu_id),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def annotate_images(items, image_dir, ref_image_dir, annotated_image_save_dir, gpu_id=0, process_id=0):
    torch.cuda.set_device(gpu_id)
    import insightface
    from groundingdino.config import GroundingDINO_SwinT_OGC
    from groundingdino.util.inference import load_model

    from src.utils.image_process import annotate

    dino_model = load_model(GroundingDINO_SwinT_OGC.__file__, GROUNDINGDINO_CKPT_PATH, device="cuda")
    face_model = insightface.app.FaceAnalysis(root=FACE_MODEL_ROOT, providers=["CUDAExecutionProvider"])
    face_model.prepare(ctx_id=int(gpu_id), det_size=(640, 640))
    for item in tqdm(items, desc=f"GPU {gpu_id}, process {process_id} annotating images"):
        item["bbox"] = []
        item["face_bbox"] = []
        bboxes_for_annotation = []
        phrases_for_annotation = []
        image_path = os.path.join(image_dir, f"{item['index']:06d}.png")
        for i, instance in enumerate(item["instance"]):
            ref_image_path = os.path.join(ref_image_dir, f"{item['index']:06d}_{i}_masked.png")
            ref_image = Image.open(ref_image_path).convert("RGB")
            faces = face_model.get(np.array(ref_image))
            if len(faces) > 0:
                ref_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                ref_face_bbox = ref_face.bbox
            else:
                ref_face = None

            image = Image.open(image_path).convert("RGB")
            bbox = get_objects(image, instance, dino_model)
            if bbox is None:
                print(f"No objects found for {instance} in image {image_path}")
                item["bbox"].append(None)
                continue

            bbox_list = normed_cxcywh_to_pixel_xyxy(
                bbox,
                width=image.width,
                height=image.height,
                return_dtype="list",
            )[0]
            bbox_normed_tensor = normed_cxcywh_to_pixel_xyxy(
                bbox,
                width=image.width,
                height=image.height,
                return_dtype="tensor",
            )[1]
            bboxes_for_annotation.append(bbox_normed_tensor)
            phrases_for_annotation.append(instance)
            item["bbox"].append(bbox_list)

            if ref_face is not None:
                target_faces = face_model.get(np.array(image))
                if len(target_faces) > 0:
                    target_face = max(
                        target_faces,
                        key=lambda x: np.dot(x.normed_embedding, ref_face.normed_embedding),
                    )
                    # if target face bbox is not in the corresponding object bbox, skip
                    target_face_bbox = target_face.bbox
                    target_face_bbox_list = target_face_bbox.tolist()
                    if not (
                        target_face_bbox_list[0] >= bbox_list[0]
                        and target_face_bbox_list[1] >= bbox_list[1]
                        and target_face_bbox_list[2] <= bbox_list[2]
                        and target_face_bbox_list[3] <= bbox_list[3]
                    ):
                        print(
                            f"Warning: Detected face not in bounding box for {instance} in image {image_path}, skipping face bbox."
                        )
                        item["face_bbox"].append(None)
                        continue
                    item["face_bbox"].append(
                        {
                            "ref": [round(x) for x in ref_face_bbox.tolist()],
                            "target": [round(x) for x in target_face_bbox.tolist()],
                            "score": float(np.dot(target_face.normed_embedding, ref_face.normed_embedding)),
                        }
                    )
                    target_face_bbox_normed = target_face_bbox / np.array(
                        [image.width, image.height, image.width, image.height]
                    )
                    target_face_bbox_normed = torch.tensor(target_face_bbox_normed, dtype=torch.float32)
                    bboxes_for_annotation.append(target_face_bbox_normed)
                    phrases_for_annotation.append(f"{instance} (face)")
                else:
                    item["face_bbox"].append(None)
            else:
                item["face_bbox"].append(None)

        if len(bboxes_for_annotation) > 0:
            image_source = np.array(Image.open(image_path).convert("RGB"))
            bboxes_for_annotation = torch.stack([bbox for bbox in bboxes_for_annotation if bbox is not None])
            annotated_image = annotate(image_source, bboxes_for_annotation, phrases_for_annotation)[..., ::-1]
            annotated_image_save_path = os.path.join(annotated_image_save_dir, f"{item['index']:06d}.png")
            Image.fromarray(annotated_image).save(annotated_image_save_path)

    return items


def parallel_annotate_images(
    items,
    image_dir,
    ref_image_dir,
    annotated_image_save_dir,
    num_processes=None,
):
    import multiprocessing

    num_gpus = torch.cuda.device_count()
    num_processes = num_processes if num_processes is not None else num_gpus

    chunk_size = len(items) // num_processes + 1
    items_chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")

    print(f"Using {num_processes} processes for parallel image annotation")

    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.starmap(
        annotate_images,
        [
            (items_chunk, image_dir, ref_image_dir, annotated_image_save_dir, i % num_gpus, i)
            for i, items_chunk in enumerate(items_chunks)
        ],
    )

    pool.close()
    pool.join()

    prompts_with_bboxes = []
    for result in results:
        prompts_with_bboxes.extend(result)

    prompts_with_bboxes.sort(key=lambda x: x["index"])
    return prompts_with_bboxes


def get_masked_dino_features(dino_model, dino_processor, image_pil, mask_np, device="cuda"):
    image_tensor = dino_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = dino_model(**image_tensor).last_hidden_state

    image_features = features[:, 1:, :]

    num_patches = image_features.shape[1]
    patch_grid_size = int(num_patches**0.5)

    if isinstance(mask_np, np.ndarray) and mask_np.dtype == bool:
        mask_np = mask_np.astype(np.float32)

    mask_resized = cv2.resize(mask_np, (patch_grid_size, patch_grid_size), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.tensor(mask_resized, dtype=torch.bool).to(device)

    mask_expanded = mask_tensor.reshape(1, patch_grid_size * patch_grid_size, 1)
    masked_features = image_features[mask_expanded.expand_as(image_features)]

    global_feature = masked_features.view(-1, image_features.shape[-1]).mean(dim=0, keepdim=True)

    return global_feature


def cal_dino_score(prompts, ref_image_dir, ref_mask_dir, target_image_dir, target_mask_dir, gpu_id=0):
    dino_model_root = DINO_MODEL_ROOT
    from transformers import AutoImageProcessor, AutoModel

    torch.cuda.set_device(gpu_id)
    dino_model = AutoModel.from_pretrained(dino_model_root).to(device="cuda")
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_root)

    for p in tqdm(prompts, desc=f"GPU {gpu_id} calculating DINO scores"):
        p["dino_score"] = []
        for i in range(len(p["instance_prompt"])):
            ref_image_path = os.path.join(ref_image_dir, f"{p['index']:06d}_{i}.png")
            ref_mask_path = os.path.join(ref_mask_dir, f"{p['index']:06d}_{i}_masked.png")
            target_image_path = os.path.join(target_image_dir, f"{p['index']:06d}_{i}.png")
            target_mask_path = os.path.join(target_mask_dir, f"{p['index']:06d}_{i}_masked.png")

            if not os.path.exists(target_image_path) or not os.path.exists(target_mask_path):
                p["dino_score"].append(None)
                continue

            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_mask = np.array(Image.open(ref_mask_path).convert("RGBA").split()[-1]) > 0

            target_image = Image.open(target_image_path).convert("RGB")
            target_mask = np.array(Image.open(target_mask_path).convert("RGBA").split()[-1]) > 0

            # check if masks are empty
            if np.sum(ref_mask) == 0:
                print(f"Warning: Empty reference mask for {p['index']:06d}_{i}, skipping DINO score calculation.")
                p["dino_score"].append(None)
                continue
            if np.sum(target_mask) == 0:
                print(f"Empty target mask for {p['index']:06d}_{i}, using full image as target.")
                target_mask = np.ones_like(target_mask, dtype=bool)

            ref_feature = get_masked_dino_features(dino_model, dino_processor, ref_image, ref_mask)
            target_feature = get_masked_dino_features(dino_model, dino_processor, target_image, target_mask)

            score = torch.cosine_similarity(ref_feature, target_feature).item()
            if np.isnan(score):
                print(f"Warning: NaN score for {p['index']:06d}_{i}, setting score to 0.0")
                score = None
            p["dino_score"].append(score)

    return prompts


def parallel_cal_dino_score(
    prompts,
    ref_image_dir,
    ref_mask_dir,
    target_image_dir,
    target_mask_dir,
    num_processes=None,
):
    import multiprocessing

    num_gpus = torch.cuda.device_count()
    num_processes = num_processes if num_processes is not None else num_gpus

    chunk_size = len(prompts) // num_processes + 1
    prompts_chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")

    print(f"Using {num_processes} processes for parallel DINO score calculation")

    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.starmap(
        cal_dino_score,
        [
            (prompts_chunk, ref_image_dir, ref_mask_dir, target_image_dir, target_mask_dir, i % num_gpus)
            for i, prompts_chunk in enumerate(prompts_chunks)
        ],
    )

    pool.close()
    pool.join()

    prompts_with_scores = []
    for result in results:
        prompts_with_scores.extend(result)

    prompts_with_scores.sort(key=lambda x: x["index"])
    return prompts_with_scores


def filter_prompts(prompts, dino_score_threshold, face_score_threshold):
    from torchvision.ops import box_iou

    print(f"Read {len(prompts)} prompts for filtering")

    filtered_prompts = []
    sum_dino_score = 0.0
    obj_count = 0
    item_with_face = 0
    sum_face_score = 0.0
    face_count = 0
    item_wo_face = 0

    # filter prompts with None in bbox
    for p in tqdm(prompts, desc="Filtering None bboxes and calculating initial average scores"):
        contain_face = False
        # filter not detected bboxes
        if "bbox" not in p or "bbox" in p and any(b is None for b in p["bbox"]):
            continue
        # filter overlap bboxes
        if "bbox" in p:
            bboxes = [torch.tensor(b) for b in p["bbox"] if b is not None]
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    box1 = bboxes[i].unsqueeze(0)
                    box2 = bboxes[j].unsqueeze(0)
                    iou = box_iou(box1, box2)[0, 0].item()
                    if iou > 0.75:
                        break
            if iou > 0.75:
                continue
        # filter None in dino_score
        if "dino_score" not in p or "dino_score" in p and any(score is None for score in p["dino_score"]):
            continue

        if "dino_score" in p:
            for score in p["dino_score"]:
                if score is not None:
                    sum_dino_score += score
                    obj_count += 1

        if "face_bbox" in p:
            valid_face_scores = []
            valid_face_bbox = []
            for face_info, bbox in zip(p["face_bbox"], p["bbox"]):
                if face_info is not None and face_info["score"] is not None and bbox is not None:
                    target_bbox = face_info["target"]
                    if (
                        target_bbox[0] >= bbox[0]
                        and target_bbox[1] >= bbox[1]
                        and target_bbox[2] <= bbox[2]
                        and target_bbox[3] <= bbox[3]
                    ):
                        valid_face_scores.append(face_info["score"])
                        valid_face_bbox.append(face_info)

                        sum_face_score += face_info["score"]
                        face_count += 1
                        contain_face = True
                    else:
                        valid_face_scores.append(0.0)
                        valid_face_bbox.append(None)
                else:
                    valid_face_bbox.append(None)
            p["face_bbox"] = valid_face_bbox

        if contain_face:
            item_with_face += 1
        else:
            item_wo_face += 1

        filtered_prompts.append(p)

    print(
        f"Prompts count after removing invalid bboxes: {len(filtered_prompts)}\n"
        f"Average DINO score before filtering: {sum_dino_score / obj_count if obj_count > 0 else 0:.4f} over {obj_count} references, {item_wo_face} items\n"
        f"Average Face score before filtering: {sum_face_score / face_count if face_count > 0 else 0:.4f} over {face_count} references, {item_with_face} items\n"
    )

    sum_dino_score = 0.0
    obj_count = 0
    item_with_face = 0
    sum_face_score = 0.0
    face_count = 0
    item_wo_face = 0

    filtered_prompts_2 = []
    for p in tqdm(
        filtered_prompts,
        desc=f"Filtering prompts with dino threshold {dino_score_threshold} and face threshold {face_score_threshold}",
    ):
        if any(score is None or score < dino_score_threshold for score in p["dino_score"]):
            continue
        sum_dino_score += sum(score for score in p["dino_score"])
        obj_count += len(p["dino_score"])
        if any(
            fb is not None and (fb["score"] is None or fb["score"] < face_score_threshold)
            for fb in p.get("face_bbox", [])
            if fb is not None
        ):
            continue
        sum_face_score += sum(face["score"] for face in p.get("face_bbox", []) if face is not None)
        face_count += sum(1 for face in p.get("face_bbox", []) if face is not None)
        if "face_bbox" in p and any(fb is not None for fb in p["face_bbox"]):
            item_with_face += 1
        else:
            item_wo_face += 1
        filtered_prompts_2.append(p)
    filtered_prompts = filtered_prompts_2

    print(
        f"Number of prompts after filtering: {len(filtered_prompts)}\n"
        f"Average DINO score after filtering: {sum_dino_score / obj_count if obj_count > 0 else 0:.4f} over {obj_count} references, {item_wo_face} items\n"
        f"Average Face score after filtering: {sum_face_score / face_count if face_count > 0 else 0:.4f} over {face_count} references, {item_with_face} items"
    )

    return filtered_prompts


def parallel_generate_composite_images(
    reference_image_dir,
    composite_image_save_dir,
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

    from src.utils.generate_dataset_dreamo import generate_dataset_images

    for i in range(num_gpus):
        chunk = items[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            p = pool.apply_async(
                generate_dataset_images, args=(reference_image_dir, composite_image_save_dir, chunk, i)
            )
            processes.append(p)
    for p in processes:
        p.get()

    pool.close()
    pool.join()


def segment_images(packed_prompts, image_dir, masked, batch_size=64, gpu_id=None):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda" if torch.cuda.is_available() else "cpu")

    DERIS_PATH = PROJECT_ROOT / "third_party" / "DeRIS"
    sys.path.append(str(DERIS_PATH))
    cur_dir = os.getcwd()
    os.chdir(DERIS_PATH)

    from third_party.DeRIS.parallel_deris import ParallelDeris, get_deris_args_wo_parsing

    args = get_deris_args_wo_parsing()
    args.device = device
    segmentation_model = ParallelDeris(args)

    print(
        f"GPU {gpu_id}: Start generating segmented images from {image_dir} to {masked}, index {packed_prompts[0]['image_index']} to {packed_prompts[-1]['image_index']}"
    )

    for batch in tqdm(range(0, len(packed_prompts), batch_size)):
        batch_items = packed_prompts[batch : batch + batch_size]
        images = []
        expressions = []
        for item in batch_items:
            idx_str = str(item["image_index"]).zfill(6)
            reference_index = str(item["reference_index"])
            image_path = f"{image_dir}/{idx_str}_{reference_index}.png"
            images.append(image_path)
            expressions.append(item["prompt"])

        if len(images) == 0:
            continue

        with torch.inference_mode():
            pred_masks = segmentation_model.inference_batch(
                image_paths=images,
                expressions=expressions,
            )

        for i, (image, mask) in enumerate(zip(images, pred_masks)):
            save_path = f"{masked}/{batch_items[i]['image_index']:06d}_{batch_items[i]['reference_index']}"
            pred_np = np.array(mask * 255, dtype=np.uint8)
            ref_image = Image.open(image).convert("RGBA")
            ref_image.putalpha(Image.fromarray(pred_np))
            ref_image.save(f"{save_path}_masked.png")

    # restore working directory
    os.chdir(cur_dir)


def parallel_segment_images(packed_prompts, image_save_dir, masked_image_save_dir, batch_size=64, num_gpus=None):
    from multiprocessing import Process

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set, continuing...")
    print(f"Using {num_gpus} GPUs for parallel segmentation")

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


def pack_prompts(
    prompts,
    image_save_dir=None,
    composite_image_save_dir=None,
    cropped_image_save_dir=None,
    masked_image_save_dir=None,
    task="generate_reference_images",
):
    packed_prompts = []
    if task == "generate_reference_images":
        for prompt in prompts:
            for i, p in enumerate(prompt["instance_prompt"]):
                if image_save_dir is not None and os.path.exists(
                    os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}.png")
                ):
                    continue
                packed_prompts.append(
                    {
                        "image_index": prompt["index"],
                        "reference_index": i,
                        "prompt": p,
                    }
                )

    elif task == "crop_reference_images":
        assert image_save_dir is not None, "image_save_dir must be provided for cropping images"
        for prompt in prompts:
            for i, p in enumerate(prompt["instance"]):
                if not os.path.exists(os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}.png")):
                    continue
                if cropped_image_save_dir is not None and os.path.exists(
                    os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}.png")
                ):
                    continue
                packed_prompts.append(
                    {
                        "image_index": prompt["index"],
                        "reference_index": i,
                        "prompt": p,
                    }
                )

    elif task == "generate_composite_images":
        assert image_save_dir is not None, "image_save_dir must be provided for generating composite images"
        processed_images = set(os.listdir(image_save_dir))
        for prompt in prompts:
            has_all_references = True
            for i in range(len(prompt["instance"])):
                if f"{prompt['index']:06d}.png" not in processed_images:
                    ref_image_path = os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}_masked.png")
                    if not os.path.exists(ref_image_path):
                        has_all_references = False
                        break
                    ref_image_mask = Image.open(ref_image_path).convert("RGBA").split()[-1]
                    if np.all(np.array(ref_image_mask) == 0):
                        print(
                            f"Reference image mask is empty for {prompt['index']:06d}_{i}, skipping composite generation."
                        )
                        has_all_references = False
                        break
            if has_all_references:
                packed_prompts.append(prompt)

    elif task == "annotate_images":
        assert composite_image_save_dir is not None, "composite_image_save_dir must be provided for cropping images"
        for prompt in prompts:
            if not os.path.exists(os.path.join(composite_image_save_dir, f"{prompt['index']:06d}.png")):
                continue
            packed_prompts.append(prompt)

    elif task == "crop_composite_images":
        assert image_save_dir is not None, "image_save_dir must be provided for cropping images"
        for prompt in prompts:
            if not os.path.exists(os.path.join(image_save_dir, f"{prompt['index']:06d}.png")):
                continue
            # if all the cropped images already exist, skip
            if cropped_image_save_dir is not None and all(
                os.path.exists(os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}.png"))
                for i in range(len(prompt["instance"]))
            ):
                continue
            packed_prompts.append(prompt)

    elif task == "segment_reference_images" or task == "segment_instance_images":
        assert masked_image_save_dir is not None, "masked_image_save_dir must be provided for segmenting images"
        for prompt in prompts:
            for i, p in enumerate(prompt["instance"]):
                if (
                    cropped_image_save_dir is not None
                    and os.path.exists(os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}_mask.png"))
                    and os.path.exists(os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}_masked.png"))
                    or not os.path.exists(os.path.join(masked_image_save_dir, f"{prompt['index']:06d}_{i}.png"))
                ):
                    continue
                packed_prompts.append(
                    {
                        "image_index": prompt["index"],
                        "reference_index": i,
                        "prompt": p,
                    }
                )

    elif task == "cal_dino_score":
        for prompt in prompts:
            packed_prompts.append(prompt)

    else:
        raise ValueError(f"{task} not found")

    return packed_prompts


def crop_composite_images(items, image_dir, output_dir):
    # use annotated bbox to crop composite images
    for item in tqdm(items, desc="Cropping composite images using annotated bboxes"):
        for i, bbox in enumerate(item["bbox"]):
            if bbox is None:
                continue
            image_path = os.path.join(image_dir, f"{item['index']:06d}.png")
            image = Image.open(image_path).convert("RGB")
            cropped_image = image.crop(bbox)
            output_path = os.path.join(output_dir, f"{item['index']:06d}_{i}.png")
            cropped_image.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset for IMIG-Dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="generate_reference_images",
        choices=[
            "generate_reference_images",
            "crop_reference_images",
            "segment_reference_images",
            "generate_composite_images",
            "annotate_images",
            "crop_composite_images",
            "segment_instance_images",
            "cal_dino_score",
            "filter_prompts",
        ],
        help="Task to perform",
    )
    prompt_save_path = os.path.join(DATASET_DIR_2, "prompts.json")
    reference_image_dir = os.path.join(DATASET_DIR_2, "reference_images")

    args = parser.parse_args()

    if args.task == "generate_reference_images":
        if not os.path.exists(reference_image_dir):
            os.makedirs(reference_image_dir)

        with open(prompt_save_path, "r") as f:
            prompts = json.load(f)
        processed_prompts = pack_prompts(prompts, image_save_dir=reference_image_dir, task=args.task)
        print(f"Total prompts to generate: {len(processed_prompts)}")
        parallel_generate_images(
            raw_prompts=processed_prompts,
            image_save_dir=reference_image_dir,
            batch_size=8,
        )
        print(f"Images saved to {reference_image_dir}")

    elif args.task == "crop_reference_images":
        cropped_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        if not os.path.exists(cropped_image_dir):
            os.makedirs(cropped_image_dir)

        with open(prompt_save_path, "r") as f:
            prompts = json.load(f)
        processed_prompts = pack_prompts(
            prompts, image_save_dir=reference_image_dir, cropped_image_save_dir=cropped_image_dir, task=args.task
        )
        print(f"Total prompts to crop: {len(processed_prompts)}")
        parallel_crop_images(
            items=processed_prompts,
            image_dir=reference_image_dir,
            output_dir=cropped_image_dir,
            num_processes=torch.cuda.device_count(),
        )
        print(f"Cropped images saved to {cropped_image_dir}")

    elif args.task == "segment_reference_images":
        segmented_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        if not os.path.exists(segmented_image_dir):
            os.makedirs(segmented_image_dir)

        with open(prompt_save_path, "r") as f:
            prompts = json.load(f)
        prompts = pack_prompts(
            prompts, image_save_dir=reference_image_dir, masked_image_save_dir=segmented_image_dir, task=args.task
        )
        print(f"Total prompts to segment reference images: {len(prompts)}")
        parallel_segment_images(
            packed_prompts=prompts,
            image_save_dir=segmented_image_dir,
            masked_image_save_dir=segmented_image_dir,
            batch_size=64,
            num_gpus=torch.cuda.device_count(),
        )
        print(f"Segmented reference images saved to {segmented_image_dir}")

    elif args.task == "generate_composite_images":
        composite_image_dir = os.path.join(DATASET_DIR_2, "composite_images")
        masked_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        if not os.path.exists(composite_image_dir):
            os.makedirs(composite_image_dir)

        with open(os.path.join(DATASET_DIR_2, "prompts.json"), "r") as f:
            prompts = json.load(f)
        prompts = pack_prompts(
            prompts, image_save_dir=masked_image_dir, composite_image_save_dir=composite_image_dir, task=args.task
        )
        print(f"Total prompts to generate composite images: {len(prompts)}")
        parallel_generate_composite_images(
            reference_image_dir=masked_image_dir,
            composite_image_save_dir=composite_image_dir,
            items=prompts,
        )
        print(f"Composite images saved to {composite_image_dir}")

    elif args.task == "annotate_images":
        annotated_image_dir = os.path.join(DATASET_DIR_2, "annotated_images")
        composite_image_dir = os.path.join(DATASET_DIR_2, "composite_images")
        ref_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        if not os.path.exists(annotated_image_dir):
            os.makedirs(annotated_image_dir)

        with open(os.path.join(DATASET_DIR_2, "prompts.json"), "r") as f:
            prompts = json.load(f)

        prompts = pack_prompts(prompts, composite_image_save_dir=composite_image_dir, task=args.task)
        annotated_prompts = parallel_annotate_images(
            items=prompts,
            image_dir=composite_image_dir,
            ref_image_dir=ref_image_dir,
            annotated_image_save_dir=annotated_image_dir,
            num_processes=torch.cuda.device_count(),
        )
        print(f"Annotated images saved to {annotated_image_dir}")
        annotated_prompts.sort(key=lambda x: x["index"])
        with open(os.path.join(DATASET_DIR_2, "prompts_with_bboxes.json"), "w") as f:
            json.dump(annotated_prompts, f, indent=2)
        print(f"Prompts with bounding boxes saved to {os.path.join(DATASET_DIR_2, 'prompts_with_bboxes.json')}")

    elif args.task == "crop_composite_images":
        composite_image_dir = os.path.join(DATASET_DIR_2, "composite_images")
        cropped_image_dir = os.path.join(DATASET_DIR_2, "instance_masks")
        if not os.path.exists(cropped_image_dir):
            os.makedirs(cropped_image_dir)

        with open(os.path.join(DATASET_DIR_2, "prompts_with_bboxes.json"), "r") as f:
            prompts = json.load(f)
        prompts = pack_prompts(
            prompts, image_save_dir=composite_image_dir, cropped_image_save_dir=cropped_image_dir, task=args.task
        )
        print(f"Total prompts to crop: {len(prompts)}")
        crop_composite_images(
            items=prompts,
            image_dir=composite_image_dir,
            output_dir=cropped_image_dir,
        )
        print(f"Cropped composite images saved to {cropped_image_dir}")

    elif args.task == "segment_instance_images":
        segmented_image_dir = os.path.join(DATASET_DIR_2, "instance_masks")
        if not os.path.exists(segmented_image_dir):
            os.makedirs(segmented_image_dir)

        with open(os.path.join(DATASET_DIR_2, "prompts_with_bboxes.json"), "r") as f:
            prompts = json.load(f)
        prompts = pack_prompts(
            prompts, image_save_dir=segmented_image_dir, masked_image_save_dir=segmented_image_dir, task=args.task
        )
        print(f"Total prompts to segment instance images: {len(prompts)}")
        parallel_segment_images(
            packed_prompts=prompts,
            image_save_dir=segmented_image_dir,
            masked_image_save_dir=segmented_image_dir,
            batch_size=64,
            num_gpus=torch.cuda.device_count(),
        )
        print(f"Segmented instance images saved to {segmented_image_dir}")

    elif args.task == "cal_dino_score":
        ref_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        ref_mask_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        target_image_dir = os.path.join(DATASET_DIR_2, "instance_masks")
        target_mask_dir = os.path.join(DATASET_DIR_2, "instance_masks")

        with open(os.path.join(DATASET_DIR_2, "prompts_with_bboxes.json"), "r") as f:
            prompts = json.load(f)
        prompts_with_scores = parallel_cal_dino_score(
            prompts=prompts,
            ref_image_dir=ref_image_dir,
            ref_mask_dir=ref_mask_dir,
            target_image_dir=target_image_dir,
            target_mask_dir=target_mask_dir,
            num_processes=torch.cuda.device_count(),
        )
        with open(os.path.join(DATASET_DIR_2, "prompts_with_dino_scores.json"), "w") as f:
            json.dump(prompts_with_scores, f, indent=2)
        print(f"Prompts with DINO scores saved to {os.path.join(DATASET_DIR_2, 'prompts_with_dino_scores.json')}")

    elif args.task == "filter_prompts":
        dino_score_threshold = 0.75
        face_score_threshold = 0.6
        with open(os.path.join(DATASET_DIR_2, "prompts_with_dino_scores.json"), "r") as f:
            prompts = json.load(f)
        filtered_prompts = filter_prompts(
            prompts,
            dino_score_threshold=dino_score_threshold,
            face_score_threshold=face_score_threshold,
        )
        with open(os.path.join(DATASET_DIR_2, "filtered_prompts.json"), "w") as f:
            json.dump(filtered_prompts, f, indent=2)
        print(
            f"Filtered prompts saved to {os.path.join(DATASET_DIR_2, 'filtered_prompts.json')}, total: {len(filtered_prompts)}"
        )
