import torch
from torchvision.ops import box_convert


def cal_iou(bbox1: torch.Tensor, bboxes: torch.Tensor):
    # bbox1: (4,), bboxes: (N, 4)
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bboxes.T  # 转置后按列取值
    x5 = torch.max(x1, x3)
    y5 = torch.max(y1, y3)
    x6 = torch.min(x2, x4)
    y6 = torch.min(y2, y4)
    w = torch.clamp(x6 - x5, min=0)
    h = torch.clamp(y6 - y5, min=0)
    inter = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - inter
    iou = inter / union
    return iou


def size_filter(bboxes: torch.Tensor, min_width: float = 1 / 16, min_height: float = 1 / 16, min_area: float = 1 / 200):
    w, h = bboxes[:, 2], bboxes[:, 3]

    filtered = ((w < min_width) | (h < min_height)) & (w * h < min_area)
    return ~filtered


def mns(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, max_instances: int = 8):
    if len(bboxes) == 0:
        return torch.tensor([], dtype=torch.long)

    # 换成 xyxy 方便计算 iou
    bboxes = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
    keep = []
    indices = torch.argsort(scores, descending=True)
    while indices.numel() > 0:
        i = indices[0]
        keep.append(i)
        if indices.numel() == 1:
            break
        ious = cal_iou(bboxes[i], bboxes[indices[1:]])  # 计算 iou
        mask = ious <= iou_threshold
        indices = indices[1:][mask]

        if len(keep) >= max_instances:
            break
    return torch.tensor(keep, dtype=torch.long)


def normed_cxcywh_to_pixel_xyxy(bbox, width=1024, height=1024, return_dtype="tensor"):
    cx, cy, w, h = bbox
    cx_p, cy_p, w_p, h_p = (
        torch.round(cx * width),
        torch.round(cy * height),
        torch.round(w * width),
        torch.round(h * height),
    )
    cx_p = torch.clamp(cx_p, w_p / 2, width - w_p / 2)
    cy_p = torch.clamp(cy_p, h_p / 2, height - h_p / 2)
    x0 = torch.round(cx_p - w_p / 2).int()
    y0 = torch.round(cy_p - h_p / 2).int()
    x1 = x0 + w_p.int()
    y1 = y0 + h_p.int()
    x0 = torch.clamp(x0, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    if return_dtype == "tensor":
        bbox_pixel_xyxy = torch.stack([x0, y0, x1, y1], dim=0)
        bbox_normed_xyxy = torch.stack([x0 / width, y0 / height, x1 / width, y1 / height], dim=0)
    elif return_dtype == "list":
        bbox_pixel_xyxy = [x0.item(), y0.item(), x1.item(), y1.item()]
        bbox_normed_xyxy = [(x0 / width).item(), (y0 / height).item(), (x1 / width).item(), (y1 / height).item()]
    elif return_dtype == "tuple":
        bbox_pixel_xyxy = (x0.item(), y0.item(), x1.item(), y1.item())
        bbox_normed_xyxy = ((x0 / width).item(), (y0 / height).item(), (x1 / width).item(), (y1 / height).item())

    return bbox_pixel_xyxy, bbox_normed_xyxy


def norm_bbox_to_tensor(bbox, image_width, image_height):
    bbox_normed = [
        bbox[0] / image_width,
        bbox[1] / image_height,
        bbox[2] / image_width,
        bbox[3] / image_height,
    ]
    return torch.tensor(bbox_normed, dtype=torch.float32)
