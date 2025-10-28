import os
import sys
import torch
from pathlib import Path
from PIL import Image

import warnings

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIS_PATH = PROJECT_ROOT / "third_party" / "DeRIS"
sys.path.insert(0, str(DERIS_PATH))
GROUNDINGDINO_PATH = PROJECT_ROOT / "third_party" / "GroundingDINO"
sys.path.insert(0, str(GROUNDINGDINO_PATH))

from src.utils.cal import normed_cxcywh_to_pixel_xyxy

current_dir = os.getcwd()
os.chdir(DERIS_PATH)  # switch to DeRIS directory to avoid path issues

from third_party.DeRIS.parallel_deris import ParallelDeris, get_deris_args_wo_parsing
from groundingdino.util.inference import load_model as load_gd_model, predict as gd_predict, load_image as gd_load_image


class GroundedDeris:
    def __init__(self, groundingdino_model_path, groundingdino_model_config, deris_model_type="refcoco", device="cuda"):
        # Load GroundingDINO model
        self.gdino_model = load_gd_model(groundingdino_model_config, groundingdino_model_path, device=device)
        self.device = device

        # Load DeRIS model
        deris_args = get_deris_args_wo_parsing(model_type=deris_model_type)
        deris_args.device = device
        self.deris_model = ParallelDeris(deris_args)

    @torch.inference_mode()
    def get_segmentation(self, img_path_or_pil, expression):
        if isinstance(img_path_or_pil, str):
            img_pil = Image.open(img_path_or_pil).convert("RGB")
        elif isinstance(img_path_or_pil, Image.Image):
            img_pil = img_path_or_pil
        else:
            raise ValueError("img_path_or_pil must be a file path or PIL Image.")
        # Step 1: Use GroundingDINO to get bounding boxes
        image_source, image = gd_load_image(img_pil)
        bboxes, logits, phrases = gd_predict(
            model=self.gdino_model,
            image=image,
            caption=expression + " .",
            box_threshold=0.3,
            text_threshold=0.25,
            device=self.device,
        )

        if len(bboxes) == 0:
            return None  # No boxes detected

        sorted_indices = torch.argsort(logits, descending=True)
        bboxes = bboxes[sorted_indices]

        bboxes_in_tuple = [
            normed_cxcywh_to_pixel_xyxy(bbox, width=img_pil.width, height=img_pil.height, return_dtype="tuple")[0]
            for bbox in bboxes
        ]

        bboxes_in_list = [list(bbox) for bbox in bboxes_in_tuple]

        cropped_images = [img_pil.crop(bbox) for bbox in bboxes_in_tuple]
        expressions = [expression for _ in range(len(cropped_images))]

        # Step 2: Use DeRIS to get segmentation masks for each cropped image
        pred_masks = self.deris_model.inference_batch(
            image_paths=cropped_images,
            expressions=expressions,
        )

        return {
            "bboxes": bboxes_in_list,
            "cropped_images": cropped_images,
            "masks": pred_masks,
            "phrases": phrases,
            "logits": logits,
        }
