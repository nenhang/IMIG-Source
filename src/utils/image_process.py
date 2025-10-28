import cv2
import numpy as np
import supervision as sv
import torch


def annotate(image_source: np.ndarray, boxes: torch.Tensor, phrases) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = boxes.numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in phrases]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


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


def get_masked_dino_features_batch(dino_model, dino_processor, image_pil_list, mask_np_list, device="cuda"):
    images_tensor = dino_processor(images=image_pil_list, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = dino_model(**images_tensor).last_hidden_state

    image_features = features[:, 1:, :]

    batch_size, num_patches, feature_dim = image_features.shape
    patch_grid_size = int(num_patches**0.5)

    mask_tensors = []
    for mask_np in mask_np_list:
        if isinstance(mask_np, np.ndarray) and mask_np.dtype == bool:
            mask_np = mask_np.astype(np.float32)
        mask_resized = cv2.resize(mask_np, (patch_grid_size, patch_grid_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor(mask_resized, dtype=torch.bool).to(device)
        mask_tensors.append(mask_tensor)
    mask_batch = torch.stack(mask_tensors, dim=0)
    mask_batch = mask_batch.reshape(batch_size, num_patches, 1)  # [B, N, 1]
    # 4. 计算masked特征 (更高效的方式)
    masked_features = image_features * mask_batch  # [B, N, D]
    sum_features = masked_features.sum(dim=1)  # [B, D]
    mask_sum = mask_batch.sum(dim=1)  # [B, 1]
    global_features = sum_features / mask_sum.clamp(min=1e-6)  # 避免除以0

    return global_features
