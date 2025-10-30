import json
import os
import sys
import warnings
from pathlib import Path

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
    FACEXLIB_MODEL_ROOT,
    FLUX_PATH,
    FLUX_TURBO_PATH,
    GROUNDINGDINO_CKPT_PATH,
)

warnings.filterwarnings("ignore", category=FutureWarning)

SAVE_FORMAT = "jpg"


def generate_images_work_process(prompts, image_save_dir, batch_size=16, gpu_id=0, use_turbo=True):
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
        # filter out the alreay existing images
        # prompt = [p["prompt"] for p in prompt_batch]
        prompt = [
            p["prompt"]
            for p in prompt_batch
            if not os.path.exists(
                os.path.join(image_save_dir, f"{p['image_index']:06d}_{p['reference_index']}.{SAVE_FORMAT}")
            )
        ]
        if len(prompt) == 0:
            continue
        with torch.no_grad():
            imgs = model(
                prompt=prompt,
                height=768,
                width=768,
                num_inference_steps=12 if use_turbo else 28,
            ).images
        for j, img in enumerate(imgs):
            file_name = f"{prompt_batch[j]['image_index']:06d}_{prompt_batch[j]['reference_index']}.{SAVE_FORMAT}"
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


def segment_reference_images(items, image_dir, output_dir, gpu_id=0):
    torch.cuda.set_device(gpu_id)

    from groundingdino.config import GroundingDINO_SwinT_OGC

    from src.utils.grounded_deris import GroundedDeris

    grounded_deris_model = GroundedDeris(
        groundingdino_model_config=GroundingDINO_SwinT_OGC.__file__,
        groundingdino_model_path=GROUNDINGDINO_CKPT_PATH,
        deris_model_type="refcoco",
        device=f"cuda:{gpu_id}",
    )

    for item in tqdm(items, desc=f"GPU {gpu_id} segmenting reference images"):
        image_path = os.path.join(image_dir, f"{item['image_index']:06d}_{item['reference_index']}.{SAVE_FORMAT}")
        expression = item["prompt"]
        result = grounded_deris_model.get_segmentation(image_path, expression)
        if result is None:
            print(f"No segmentation found for {expression} in image {image_path}, skipping.")
            continue
        mask = result["masks"][0].astype(np.uint8) * 255  # take the first mask
        if SAVE_FORMAT == "png":
            # use add alpha channel to store mask
            mask_pil = Image.fromarray(mask, mode="L")
            cropped_image = result["cropped_images"][0]
            cropped_image.putalpha(mask_pil)
            cropped_image.save(
                os.path.join(output_dir, f"{item['image_index']:06d}_{item['reference_index']}.{SAVE_FORMAT}")
            )
        else:
            mask_pil = Image.fromarray(mask, mode="L").convert("1")
            mask_pil.save(os.path.join(output_dir, f"{item['image_index']:06d}_{item['reference_index']}_mask.png"))
            cropped_image = result["cropped_images"][0]
            cropped_image.save(
                os.path.join(output_dir, f"{item['image_index']:06d}_{item['reference_index']}.{SAVE_FORMAT}")
            )
            masked_image = Image.composite(
                cropped_image,
                Image.new("RGB", cropped_image.size, (255, 255, 255)),
                mask_pil,
            )
            masked_image.save(
                os.path.join(output_dir, f"{item['image_index']:06d}_{item['reference_index']}_masked.{SAVE_FORMAT}")
            )


def parallel_segment_reference_images(
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
            target=segment_reference_images,
            args=(items_chunk, image_dir, output_dir, gpu_id),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def annotate_images(
    items,
    image_dir,
    ref_image_dir,
    annotated_image_save_dir,
    masked_instance_save_dir,
    aligned_face_save_dir,
    gpu_id=0,
    process_id=0,
):
    torch.cuda.set_device(gpu_id)
    import insightface
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from groundingdino.config import GroundingDINO_SwinT_OGC
    from transformers import AutoImageProcessor, AutoModel

    from src.utils.cal import norm_bbox_to_tensor
    from src.utils.face_helper import get_align_face
    from src.utils.grounded_deris import GroundedDeris
    from src.utils.image_process import (
        annotate,
        get_masked_dino_features,
        get_masked_dino_features_batch,
    )

    face_model = insightface.app.FaceAnalysis(root=FACE_MODEL_ROOT, providers=["CPUExecutionProvider"])
    face_model.prepare(ctx_id=-1, det_size=(512, 512))
    grounded_deris_model = GroundedDeris(
        groundingdino_model_config=GroundingDINO_SwinT_OGC.__file__,
        groundingdino_model_path=GROUNDINGDINO_CKPT_PATH,
        device="cuda",
    )
    face_restore_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device="cuda",
        model_rootpath=FACEXLIB_MODEL_ROOT,
    )
    dino_model = AutoModel.from_pretrained(DINO_MODEL_ROOT).to(device="cuda")
    dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ROOT)
    for item in tqdm(items, desc=f"GPU {gpu_id}, process {process_id} annotating images"):
        item["bbox"] = []
        item["face_bbox"] = []
        bboxes_for_annotation = []
        phrases_for_annotation = []
        image_path = os.path.join(image_dir, f"{item['index']:06d}.{SAVE_FORMAT}")
        for i, instance in enumerate(item["instance"]):
            ref_image_path = os.path.join(ref_image_dir, f"{item['index']:06d}_{i}_masked.{SAVE_FORMAT}")
            if SAVE_FORMAT == "png":
                # add white background according to alpha channel
                ref_image = Image.open(ref_image_path).convert("RGBA")
                ref_image_np = np.array(ref_image)
                ref_mask = ref_image_np[..., 3] > 127
                ref_image_with_white_bg = ref_image_np[..., :3] * (ref_image_np[..., [3]] / 255.0) + 255 * (
                    1 - ref_image_np[..., [3]] / 255.0
                )
                ref_image = Image.fromarray(ref_image_with_white_bg.astype("uint8"))
            else:
                ref_image = Image.open(ref_image_path).convert("RGB")
                ref_mask_path_prefix = os.path.join(
                    ref_image_dir,
                    f"{item['index']:06d}_{i}",
                )
                if os.path.exists(ref_mask_path_jpg := f"{ref_mask_path_prefix}_mask.{SAVE_FORMAT}"):
                    ref_mask = np.array(Image.open(ref_mask_path_jpg).convert("L")) > 127
                else:
                    assert os.path.exists(ref_mask_path_png := f"{ref_mask_path_prefix}_mask.png")
                    ref_mask = np.array(Image.open(ref_mask_path_png).convert("L")) > 127

            ref_features = get_masked_dino_features(
                dino_model=dino_model,
                dino_processor=dino_processor,
                image_pil=ref_image,
                mask_np=ref_mask,
                device="cuda",
            )

            faces = face_model.get(np.array(ref_image))
            if len(faces) > 0:
                ref_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                ref_face_bbox = ref_face.bbox
            else:
                ref_face = None

            model_output = grounded_deris_model.get_segmentation(image_path, instance)

            if not model_output:
                print(f"No objects found for {instance} in image {image_path}")
                item["bbox"].append(None)
                item["face_bbox"].append(None)
                continue
            bboxes = model_output["bboxes"]
            cropped_images = model_output["cropped_images"]
            masks = model_output["masks"]

            instance_features = get_masked_dino_features_batch(
                dino_model=dino_model,
                dino_processor=dino_processor,
                image_pil_list=cropped_images,
                mask_np_list=[mask.astype(bool) for mask in masks],
                device="cuda",
            )

            # sort by cosine similarity
            similarities = (
                torch.nn.functional.cosine_similarity(ref_features, instance_features, dim=-1).cpu().numpy().tolist()
            )

            # filter out low similarity bboxes
            threshold = 0.5
            filtered_sorted = sorted(
                (
                    (bbox, sim, mask, cropped_image)
                    for bbox, sim, mask, cropped_image in zip(bboxes, similarities, masks, cropped_images)
                    if sim >= threshold
                ),
                key=lambda x: x[1],  # 按相似度排序
                reverse=True,
            )
            bboxes, similarities, masks, cropped_images = zip(*filtered_sorted) if filtered_sorted else ([], [], [], [])

            # round bboxes to int list
            bboxes = [[round(x) for x in bbox] for bbox in bboxes]

            bbox_info = (
                {
                    "bbox": bboxes[0],
                    "score": similarities[0],
                }
                if bboxes
                else None
            )
            bbox = bboxes[0] if bboxes else None
            mask_ = masks[0] if masks else None
            cropped_image_ = cropped_images[0] if cropped_images else None

            face_info = None
            image = Image.open(image_path).convert("RGB")
            if ref_face is not None:
                target_faces = face_model.get(np.array(image))
                if target_faces:
                    best_face, best_score = max(
                        ((face, np.dot(face.normed_embedding, ref_face.normed_embedding)) for face in target_faces),
                        key=lambda x: x[1],
                    )
                    if best_score > 0.4:
                        # check if there's bbox in bboxes that contains the best_face bbox
                        best_face_bbox = best_face.bbox
                        containing_bboxes_indices = [
                            idx
                            for idx, b in enumerate(bboxes)
                            if b[0] <= best_face_bbox[0]
                            and b[1] <= best_face_bbox[1]
                            and b[2] >= best_face_bbox[2]
                            and b[3] >= best_face_bbox[3]
                        ]
                        if containing_bboxes_indices:
                            # choose the one with highest similarity score
                            best_containing_idx, _ = max(
                                [(idx, similarities[idx]) for idx in containing_bboxes_indices], key=lambda x: x[1]
                            )
                            bbox = bboxes[best_containing_idx]
                            bbox_info = {
                                "bbox": bbox,
                                "score": similarities[best_containing_idx],
                            }
                            mask_ = masks[best_containing_idx]
                            cropped_image_ = cropped_images[best_containing_idx]

                        else:
                            print(f"Face Detected but no containing bbox for {instance} in image {image_path}")
                            bbox_info = None

                        # get aligned face for reference image
                        aligned_face = get_align_face(
                            face_restore_helper, np.array(ref_image), mask=ref_mask.astype(np.uint8) * 255
                        )
                        if aligned_face is not None:
                            # convert to PIL image
                            aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
                            face_save_path = os.path.join(
                                aligned_face_save_dir,
                                f"{item['index']:06d}_{i}_aligned_face.png",  # force to save as png to avoid compression artifacts
                            )
                            aligned_face_pil.save(face_save_path)

                        # convert background to white according to Alpha channel, original aligned face is in (h, w, 4) RGBA format
                        aligned_face_bbox = None
                        if aligned_face is not None:
                            aligned_face_with_white_bg = (
                                aligned_face[..., :3] * (aligned_face[..., [3]] / 255.0)
                                + 255 * (1 - aligned_face[..., [3]] / 255.0)
                            ).astype("uint8")
                            aligned_face_detected = face_model.get(aligned_face_with_white_bg)
                            if aligned_face_detected:
                                aligned_face_best = max(
                                    aligned_face_detected,
                                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                                )
                                aligned_face_bbox = aligned_face_best.bbox
                            else:
                                print(f"No face detected in aligned face for {instance} in image {image_path}")

                        face_info = {
                            "ref": [round(x) for x in ref_face_bbox.tolist()],
                            "target": [round(x) for x in best_face_bbox.tolist()],
                            "score": float(best_score),
                            "aligned_face_bbox": [round(x) for x in aligned_face_bbox.tolist()]
                            if aligned_face_bbox is not None
                            else None,
                        }

            item["bbox"].append(bbox_info)
            item["face_bbox"].append(face_info)

            if bbox_info is not None:
                bbox_tensor = norm_bbox_to_tensor(bbox, image.width, image.height)
                bboxes_for_annotation.append(bbox_tensor)
                phrases_for_annotation.append(instance)

            if face_info is not None:
                face_bbox_tensor = norm_bbox_to_tensor(
                    best_face_bbox,
                    image.width,
                    image.height,
                )
                bboxes_for_annotation.append(face_bbox_tensor)
                phrases_for_annotation.append(f"{instance} (face)")

            # save mask and cropped image
            if bbox_info is not None:
                assert mask_ is not None and cropped_image_ is not None
                assert mask_.shape == cropped_image_.size[::-1], "Mask and cropped image size mismatch"
                mask_save_path = os.path.join(
                    masked_instance_save_dir,
                    f"{item['index']:06d}_{i}_mask.png",
                )
                mask_pil = Image.fromarray((mask_ * 255).astype(np.uint8)).convert("1")
                mask_pil.save(mask_save_path)
                cropped_image_save_path = os.path.join(
                    masked_instance_save_dir,
                    f"{item['index']:06d}_{i}.{SAVE_FORMAT}",
                )
                cropped_image_.save(cropped_image_save_path)
                # also save a white bg version
                cropped_image_with_white_bg = Image.new("RGB", cropped_image_.size, (255, 255, 255))
                cropped_image_with_white_bg.paste(cropped_image_, mask=mask_pil)
                cropped_image_with_white_bg.save(
                    os.path.join(
                        masked_instance_save_dir,
                        f"{item['index']:06d}_{i}_masked.{SAVE_FORMAT}",
                    )
                )

        if len(bboxes_for_annotation) > 0:
            image_source = np.array(Image.open(image_path).convert("RGB"))
            bboxes_for_annotation = torch.stack([bbox for bbox in bboxes_for_annotation if bbox is not None])
            annotated_image = annotate(image_source, bboxes_for_annotation, phrases_for_annotation)[..., ::-1]
            annotated_image_save_path = os.path.join(annotated_image_save_dir, f"{item['index']:06d}.{SAVE_FORMAT}")
            Image.fromarray(annotated_image).save(annotated_image_save_path)

    return items


def parallel_annotate_images(
    items,
    image_dir,
    ref_image_dir,
    annotated_image_save_dir,
    masked_instance_save_dir,
    aligned_face_save_dir,
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
            (
                items_chunk,
                image_dir,
                ref_image_dir,
                annotated_image_save_dir,
                masked_instance_save_dir,
                aligned_face_save_dir,
                i % num_gpus,
                i,
            )
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


def filter_prompts(prompts, dino_score_threshold, face_score_threshold, min_valid_instance_count=2):
    from torchvision.ops import box_iou

    print(f"Read {len(prompts)} prompts for filtering")

    filtered_prompts = []
    sum_dino_score = 0.0
    obj_count = 0
    item_with_face = 0
    sum_face_score = 0.0
    face_count = 0
    item_wo_face = 0

    instance_number_distribution = {}

    # filter prompts with None in bbox
    for p in tqdm(prompts, desc="Filtering None bboxes and calculating initial average scores"):
        contain_face = False
        if "bbox" not in p or "face_bbox" not in p:
            print(f"Prompt {p['index']} does not have bbox or face_bbox, skipping.")
            continue

        # filter overlap bboxes
        if "bbox" in p:
            bboxes = p["bbox"]
            for i in range(len(bboxes)):
                if bboxes[i] is None or len(bboxes[i]) != 4:
                    continue
                box1 = torch.tensor(bboxes[i]["bbox"]).unsqueeze(0)
                for j in range(i + 1, len(bboxes)):
                    if bboxes[j] is None or len(bboxes[j]) != 4:
                        continue
                    box2 = torch.tensor(bboxes[j]["bbox"]).unsqueeze(0)
                    iou = box_iou(box1, box2)[0, 0].item()
                    if iou > 0.75:
                        # choose the one whose score is higher and set the other to None
                        if bboxes[i]["score"] >= bboxes[j]["score"]:
                            bboxes[j] = None
                        else:
                            bboxes[i] = None
                        print(f"Removed overlapping bboxes with IoU {iou:.2f} in prompt {p['index']}")

        # filter overlap face bboxes
        if "face_bbox" in p:
            face_bboxes = p["face_bbox"]
            for i in range(len(face_bboxes)):
                if face_bboxes[i] is None or len(face_bboxes[i]) != 4:
                    continue
                box1 = torch.tensor(face_bboxes[i]["target"]).unsqueeze(0)
                for j in range(i + 1, len(face_bboxes)):
                    if face_bboxes[j] is None or len(face_bboxes[j]) != 4:
                        continue
                    box2 = torch.tensor(face_bboxes[j]["target"]).unsqueeze(0)
                    iou = box_iou(box1, box2)[0, 0].item()
                    if iou > 0.5:
                        # choose the one whose score is higher and set the other to None
                        if face_bboxes[i]["score"] >= face_bboxes[j]["score"]:
                            face_bboxes[j] = None
                        else:
                            face_bboxes[i] = None
                        print(f"Removed overlapping face bboxes with IoU {iou:.2f} in prompt {p['index']}")

        # check if the number of obj and face of difference instances are adiquate
        assert len(p["bbox"]) == len(p["face_bbox"]), "Number of bboxes and instance prompts do not match"
        valid_instance_count = 0
        for i, (bbox, face_bbox) in enumerate(zip(p["bbox"], p["face_bbox"])):
            if bbox is not None or face_bbox is not None:
                valid_instance_count += 1
        if valid_instance_count < min_valid_instance_count:
            continue

        # add scores
        for bbox in p["bbox"]:
            if bbox is not None and bbox["score"] is not None:
                sum_dino_score += bbox["score"]
                obj_count += 1

        for fb in p["face_bbox"]:
            if fb is not None and fb["score"] is not None:
                sum_face_score += fb["score"]
                face_count += 1
                contain_face = True

        if contain_face:
            item_with_face += 1
        else:
            item_wo_face += 1

        instance_num = valid_instance_count
        if instance_num not in instance_number_distribution:
            instance_number_distribution[instance_num] = 0
        instance_number_distribution[instance_num] += 1

        filtered_prompts.append(p)

    print(
        f"Prompts count after removing invalid bboxes: {len(filtered_prompts)}\n"
        f"Average DINO score before filtering: {sum_dino_score / obj_count if obj_count > 0 else 0:.4f} over {obj_count} references, {item_wo_face} items\n"
        f"Average Face score before filtering: {sum_face_score / face_count if face_count > 0 else 0:.4f} over {face_count} references, {item_with_face} items\n"
        f"Instance number distribution before filtering: {instance_number_distribution}"
    )

    sum_dino_score = 0.0
    obj_count = 0
    item_with_face = 0
    sum_face_score = 0.0
    face_count = 0
    item_wo_face = 0

    instance_number_distribution = {}

    filtered_prompts_2 = []
    for p in tqdm(
        filtered_prompts,
        desc=f"Filtering prompts with dino threshold {dino_score_threshold} and face threshold {face_score_threshold}",
    ):
        available_indices = []
        contain_face = False
        for i, (bbox, face_bbox) in enumerate(zip(p["bbox"], p["face_bbox"])):
            if (
                bbox is not None
                and bbox["score"] >= dino_score_threshold
                or face_bbox is not None
                and face_bbox["score"] >= face_score_threshold
                and face_bbox["aligned_face_bbox"] is not None
            ):
                available_indices.append(i)
        if len(available_indices) < min_valid_instance_count:
            continue

        # filter bboxes and cal new scores
        p["instance"] = [p["instance"][i] for i in available_indices]
        p["instance_prompt"] = [p["instance_prompt"][i] for i in available_indices]
        p["bbox"] = [p["bbox"][i] for i in available_indices]
        p["face_bbox"] = [p["face_bbox"][i] for i in available_indices]
        p["indices"] = available_indices

        for i, (bbox, face_bbox) in enumerate(zip(p["bbox"], p["face_bbox"])):
            if bbox is not None:
                sum_dino_score += bbox["score"]
                obj_count += 1
            if face_bbox is not None:
                sum_face_score += face_bbox["score"]
                face_count += 1
                contain_face = True

        if contain_face:
            item_with_face += 1
        else:
            item_wo_face += 1

        filtered_prompts_2.append(p)
        instance_num = len(available_indices)
        if instance_num not in instance_number_distribution:
            instance_number_distribution[instance_num] = 0
        instance_number_distribution[instance_num] += 1

    filtered_prompts = filtered_prompts_2

    print(
        f"Number of prompts after filtering: {len(filtered_prompts)}\n"
        f"Average DINO score after filtering: {sum_dino_score / obj_count if obj_count > 0 else 0:.4f} over {obj_count} references, {item_wo_face} items\n"
        f"Average Face score after filtering: {sum_face_score / face_count if face_count > 0 else 0:.4f} over {face_count} references, {item_with_face} items\n"
        f"Instance number distribution after filtering: {instance_number_distribution}"
    )

    return filtered_prompts


def generate_composite_images(reference_image_dir, composite_image_save_dir, items: list, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # You may choose from other models if you complement the corresponding API in src/model_api
    # from src.model_api.dreamo import load_pipeline, generate_composite_image
    from src.model_api.mosaic import generate_composite_image, load_pipeline

    pipeline = load_pipeline(device=device)

    for item in tqdm(items, desc=f"GPU {gpu_id} generating composite images"):
        idx_str = str(item["index"]).zfill(6)
        ref_paths = []
        for i in range(len(item["instance_prompt"])):
            ref_path = f"{reference_image_dir}/{idx_str}_{i}.{SAVE_FORMAT}"
            ref_paths.append(ref_path)
        composite_image = generate_composite_image(
            pipeline,
            ref_image_paths=ref_paths,
            prompt=item["prompt"],
        )
        save_path = f"{composite_image_save_dir}/{item['index']:06d}.{SAVE_FORMAT}"
        composite_image.save(save_path)


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

    for i in range(num_gpus):
        chunk = items[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            p = pool.apply_async(
                generate_composite_images, args=(reference_image_dir, composite_image_save_dir, chunk, i)
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
            image_path = f"{image_dir}/{idx_str}_{reference_index}.{SAVE_FORMAT}"
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
            if SAVE_FORMAT == "png":
                mask_pil = Image.fromarray(pred_np).convert("L")
                ref_image = Image.open(image).convert("RGBA")
                ref_image.putalpha(mask_pil)
                ref_image.save(f"{save_path}_masked.png")
            else:
                mask_pil = Image.fromarray(pred_np).convert("1")
                mask_pil.save(f"{save_path}_mask.png")
                ref_image = Image.open(image).convert("RGB")
                white_bg = Image.new("RGB", ref_image.size, (255, 255, 255))
                masked_image = Image.composite(ref_image, white_bg, mask_pil)
                masked_image.save(f"{save_path}_masked.{SAVE_FORMAT}")

    # restore working directory
    os.chdir(cur_dir)


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
        for prompt in tqdm(prompts):
            for i, p in enumerate(prompt["instance_prompt"]):
                # if image_save_dir is not None and os.path.exists(
                #     os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}.{SAVE_FORMAT}")
                # ):
                #     continue
                packed_prompts.append(
                    {
                        "image_index": prompt["index"],
                        "reference_index": i,
                        "prompt": p,
                    }
                )

    elif task == "crop_reference_images":
        assert image_save_dir is not None, "image_save_dir must be provided for cropping images"
        for prompt in tqdm(prompts):
            for i, p in enumerate(prompt["instance"]):
                if not os.path.exists(os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}.{SAVE_FORMAT}")):
                    continue
                if cropped_image_save_dir is not None and os.path.exists(
                    os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}.{SAVE_FORMAT}")
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
        for prompt in tqdm(prompts):
            if f"{prompt['index']:06d}.{SAVE_FORMAT}" in processed_images:
                continue
            if not all(
                os.path.exists(os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}_masked.{SAVE_FORMAT}"))
                for i in range(len(prompt["instance"]))
            ):
                continue
            num_available_references = 0
            for i, instance in enumerate(prompt["instance"]):
                has_available_reference = True
                ref_image_path = os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}_masked.{SAVE_FORMAT}")
                if SAVE_FORMAT == "png":
                    ref_image_mask = Image.open(ref_image_path).convert("RGBA").split()[-1]
                    if np.all(np.array(ref_image_mask) < 128):
                        print(f"Reference image mask is empty for {instance} in {prompt['index']:06d}_{i}.")
                        has_available_reference = False
                elif SAVE_FORMAT == "jpg":
                    base_path = os.path.join(image_save_dir, f"{prompt['index']:06d}_{i}_mask")
                    ref_mask_path = f"{base_path}.jpg" if os.path.exists(f"{base_path}.jpg") else f"{base_path}.png"
                    assert os.path.exists(ref_mask_path)
                    if np.all(np.array(Image.open(ref_mask_path).convert("L")) < 128):
                        print(f"Reference image mask is empty for {instance} in {prompt['index']:06d}_{i}.")
                        has_available_reference = False

                if has_available_reference:
                    num_available_references += 1

            if num_available_references >= 2:
                packed_prompts.append(prompt)
            else:
                print(
                    f"Not enough available reference images for prompt index {prompt['index']:06d}, skipping composite generation."
                )

    elif task == "annotate_images":
        assert composite_image_save_dir is not None, "composite_image_save_dir must be provided for cropping images"
        for prompt in tqdm(prompts):
            if not os.path.exists(os.path.join(composite_image_save_dir, f"{prompt['index']:06d}.{SAVE_FORMAT}")):
                continue
            packed_prompts.append(prompt)

    elif task == "crop_composite_images":
        assert image_save_dir is not None, "image_save_dir must be provided for cropping images"
        for prompt in tqdm(prompts):
            if not os.path.exists(os.path.join(image_save_dir, f"{prompt['index']:06d}.{SAVE_FORMAT}")):
                continue
            # if all the cropped images already exist, skip
            if cropped_image_save_dir is not None and all(
                os.path.exists(os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}.{SAVE_FORMAT}"))
                for i in range(len(prompt["instance"]))
            ):
                continue
            packed_prompts.append(prompt)

    elif task == "segment_instance_images":
        assert masked_image_save_dir is not None, "masked_image_save_dir must be provided for segmenting images"
        for prompt in prompts:
            for i, p in enumerate(prompt["instance"]):
                if (
                    masked_image_save_dir is not None
                    and os.path.exists(
                        os.path.join(masked_image_save_dir, f"{prompt['index']:06d}_{i}_masked.{SAVE_FORMAT}")
                    )
                    or not os.path.exists(
                        os.path.join(cropped_image_save_dir, f"{prompt['index']:06d}_{i}.{SAVE_FORMAT}")
                    )
                ):
                    continue
                packed_prompts.append(
                    {
                        "image_index": prompt["index"],
                        "reference_index": i,
                        "prompt": p,
                    }
                )

    else:
        raise ValueError(f"{task} not found")

    return packed_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset for IMIG-Dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="generate_reference_images",
        choices=[
            "generate_reference_images",
            "segment_reference_images",
            "generate_composite_images",
            "annotate_images",
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

    elif args.task == "segment_reference_images":
        cropped_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        if not os.path.exists(cropped_image_dir):
            os.makedirs(cropped_image_dir)

        with open(prompt_save_path, "r") as f:
            prompts = json.load(f)
        processed_prompts = pack_prompts(
            prompts, image_save_dir=reference_image_dir, cropped_image_save_dir=cropped_image_dir, task=args.task
        )
        print(f"Total prompts to segment: {len(processed_prompts)}")
        parallel_segment_reference_images(
            items=processed_prompts,
            image_dir=reference_image_dir,
            output_dir=cropped_image_dir,
            num_processes=torch.cuda.device_count(),
        )
        print(f"Segmented images saved to {cropped_image_dir}")

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
        aligned_face_save_dir = os.path.join(DATASET_DIR_2, "aligned_faces")
        ref_image_dir = os.path.join(DATASET_DIR_2, "reference_masks")
        instance_image_dir = os.path.join(DATASET_DIR_2, "instance_masks")
        os.makedirs(annotated_image_dir, exist_ok=True)
        os.makedirs(aligned_face_save_dir, exist_ok=True)
        os.makedirs(instance_image_dir, exist_ok=True)

        with open(os.path.join(DATASET_DIR_2, "prompts.json"), "r") as f:
            prompts = json.load(f)

        prompts = pack_prompts(prompts, composite_image_save_dir=composite_image_dir, task=args.task)
        annotated_prompts = parallel_annotate_images(
            items=prompts,
            image_dir=composite_image_dir,
            ref_image_dir=ref_image_dir,
            annotated_image_save_dir=annotated_image_dir,
            masked_instance_save_dir=instance_image_dir,
            aligned_face_save_dir=aligned_face_save_dir,
            num_processes=torch.cuda.device_count(),
        )
        print(f"Annotated images saved to {annotated_image_dir}")
        annotated_prompts.sort(key=lambda x: x["index"])
        with open(os.path.join(DATASET_DIR_2, "prompts_with_bboxes.json"), "w") as f:
            json.dump(annotated_prompts, f, indent=2)
        print(f"Prompts with bounding boxes saved to {os.path.join(DATASET_DIR_2, 'prompts_with_bboxes.json')}")

    elif args.task == "filter_prompts":
        dino_score_threshold = 0.82
        face_score_threshold = 0.65
        with open(os.path.join(DATASET_DIR_2, "prompts_with_bboxes.json"), "r") as f:
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
