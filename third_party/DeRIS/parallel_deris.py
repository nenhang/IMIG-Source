# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate

from deris.datasets import extract_data
from tools.demo import DERIS


def get_deris_args(model_type="refcoco"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=None, help="Image file")  # will be ignored in parallel mode
    parser.add_argument("--expression", default=None, help="text")  # will be ignored in parallel mode
    parser.add_argument("--type", default="refcoco", help="Dataset type, e.g., refcoco, refcocog, refclef")
    parser.add_argument("--output_dir", default="asserts/outdir", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.9, help="bbox score threshold")
    args = parser.parse_args()
    if model_type is not None and model_type != args.type:
        args.type = model_type
    if args.type == "refcoco":
        args.config = "configs/refcoco/DERIS-L-refcoco.py"
        args.checkpoint = "pretrain_weights/DeRIS-L-refcoco.pth"
        print("Using refcoco model")
    elif args.type == "grefcoco":
        args.config = "configs/gres/DERIS-L-grefcoco.py"
        args.checkpoint = "pretrain_weights/DeRIS-L-grefcoco.pth"
        print("Using grefcoco model")
    else:
        print(f"Model type {args.type} is not supported. Using refcoco as default.")

    return args


def get_deris_args_wo_parsing(model_type="refcoco"):
    args = argparse.Namespace()
    if model_type == "refcoco":
        args.config = "configs/refcoco/DERIS-L-refcoco.py"
        args.checkpoint = "pretrain_weights/DeRIS-L-refcoco.pth"
        print("Using refcoco model")
    elif model_type == "grefcoco":
        args.config = "configs/gres/DERIS-L-grefcoco.py"
        args.checkpoint = "pretrain_weights/DeRIS-L-grefcoco.pth"
        print("Using grefcoco model")
    else:
        print(f"Model type {model_type} is not supported. Using refcoco as default.")
        args.config = "configs/refcoco/DERIS-L-refcoco.py"
        args.checkpoint = "pretrain_weights/DeRIS-L-refcoco.pth"

    args.img = None
    args.expression = None
    args.type = model_type
    args.output_dir = "asserts/outdir"
    args.device = "cuda:0"
    args.score_thr = 0.9
    return args


class ParallelDeris(DERIS):
    def __init__(self, args):
        super().__init__(args)

    def preprocess_image(self, img_path, expression):
        results = {}
        img_bytes = self.file_client.get(img_path)
        image = mmcv.imfrombytes(img_bytes, flag="color", backend=None)
        results["filename"] = img_path
        results["img"] = image
        results["img_shape"] = image.shape
        results["ori_shape"] = image.shape
        results["empty"] = None

        cleaned_expression = self.clean_string(expression)
        tokens = self.tokenizer.tokenize(cleaned_expression)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if len(tokens) > self.max_token - 2:
            tokens = tokens[: self.max_token - 2]
        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_token - num_tokens)
        ref_expr_inds = tokens + [self.pad_token_id] * (self.max_token - num_tokens)

        results["ref_expr_inds"] = np.array(ref_expr_inds, dtype=int)
        results["text_attention_mask"] = np.array(padding_mask, dtype=int)
        results["expression"] = cleaned_expression
        results["max_token"] = self.max_token
        results["with_bbox"] = False
        results["with_mask"] = False

        result_resize = self.Trans_HierResize(results)
        result_normalize = self.Trans_Normalize(result_resize)
        result_defaultFormatBundle = self.Trans_DefaultFormatBundle(result_normalize)
        result_CollectData = self.Trans_CollectData(result_defaultFormatBundle)
        return result_CollectData

    def inference_batch(self, image_paths, expressions):
        data_list = [self.preprocess_image(p, expression) for p, expression in zip(image_paths, expressions)]
        data = collate(data_list, samples_per_gpu=len(data_list))
        inputs = extract_data(data)
        with torch.inference_mode():
            predictions = self.model(**inputs, return_loss=False, rescale=True, with_bbox=True, with_mask=True)
        pred_masks = predictions.pop("pred_masks")
        return [maskUtils.decode(pred_mask["pred_masks"]) for pred_mask in pred_masks]
