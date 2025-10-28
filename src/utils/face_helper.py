import cv2
import numpy as np
import torch

from torchvision.transforms.functional import normalize
from third_party.DreamO.dreamo.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


# @torch.no_grad()
# def get_align_face(face_helper, img):
#     # the face preprocessing code is same as PuLID
#     face_helper.clean_all()
#     image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     face_helper.read_image(image_bgr)
#     face_helper.get_face_landmarks_5(only_center_face=True)
#     face_helper.align_warp_face()
#     if len(face_helper.cropped_faces) == 0:
#         return None
#     align_face = face_helper.cropped_faces[0]

#     input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
#     input = input.to(face_helper.device)
#     parsing_out = face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
#     parsing_out = parsing_out.argmax(dim=1, keepdim=True)
#     bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
#     bg = sum(parsing_out == i for i in bg_label).bool()
#     white_image = torch.ones_like(input)
#     # only keep the face features
#     face_features_image = torch.where(bg, white_image, input)
#     face_features_image = tensor2img(face_features_image, rgb2bgr=False)

#     return face_features_image


@torch.no_grad()
def get_align_face(face_helper: FaceRestoreHelper, img, mask=None):
    """
    处理人脸对齐和裁剪，支持外部传入的透明度掩码 (mask)。

    Args:
        face_helper (FaceRestoreHelper): 人脸恢复辅助类实例。
        img (np.ndarray): 输入的原始 RGB 图像 (H, W, 3)。
        mask (np.ndarray, optional): 原始透明度掩码 (H, W) 或 (H, W, 1)，0=透明，255=不透明。
    """
    # 1. 人脸对齐和裁剪
    face_helper.clean_all()

    # 注意：输入 img 假设是 RGB
    # 转换为 BGR 给 face_helper.read_image()
    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face_helper.read_image(image_bgr)  # 此处 face_helper 仅处理 3 通道 BGR

    # 裁剪前，先处理输入的 mask
    original_alpha_np = None
    if mask is not None:
        if mask.ndim == 3 and mask.shape[2] == 1:
            original_alpha_np = mask.squeeze()  # 确保是 (H, W)
        elif mask.ndim == 2:
            original_alpha_np = mask
        else:
            # 如果 mask 格式不正确，可以选择忽略或抛出错误
            print("Warning: Input mask format is incorrect, ignoring it.")

    # 执行人脸检测、定位和仿射矩阵计算
    face_helper.get_face_landmarks_5(only_center_face=True)

    # 执行人脸对齐和裁剪。此时 affine_matrices 和 cropped_faces 被填充
    face_helper.align_warp_face()

    if len(face_helper.cropped_faces) == 0:
        return None

    align_face_bgr = face_helper.cropped_faces[0]

    # 提取对齐使用的仿射矩阵
    # 由于我们设置了 only_center_face=True，所以只处理第一个检测到的脸
    affine_matrix = face_helper.affine_matrices[0]
    target_size = face_helper.face_size  # 对齐后的目标尺寸 (W, H)

    # ** 关键步骤 1：对齐原始 Alpha **
    aligned_input_alpha = None
    if original_alpha_np is not None:
        # 对原始 Alpha 应用相同的仿射变换
        # 使用 cv2.INTER_NEAREST 避免模糊 Alpha 边缘
        aligned_input_alpha = cv2.warpAffine(
            original_alpha_np,
            affine_matrix,
            target_size,  # (W, H)
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,  # 边界设为 0 (完全透明)
        )
        # 确保 aligned_input_alpha 是 np.uint8, [0, 255]
        aligned_input_alpha = np.clip(aligned_input_alpha, 0, 255).astype(np.uint8)
    # ** 结束对齐原始 Alpha **

    # 转换为 RGB 并在 [0, 1] 归一化 (Input to Parsing Model)
    input_tensor = img2tensor(align_face_bgr, bgr2rgb=True).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(face_helper.device)

    # 获取对齐后的原始人脸的 NumPy 数组 (H, W, 3)
    align_face_rgb = tensor2img(input_tensor, rgb2bgr=False)

    # 2. 人脸解析
    parsing_out = face_helper.face_parse(normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
    parsing_out = parsing_out.argmax(dim=1, keepdim=True)

    # 3. 创建背景/非特征区域掩码
    bg_label = [0]
    bg_mask_tensor = sum(parsing_out == i for i in bg_label).bool()

    # 4. 创建 Alpha 通道
    bg_mask_np = bg_mask_tensor.squeeze().cpu().numpy()

    # 解析结果 Alpha 通道：背景区域 (True) 设置为 0 (完全透明)
    parsing_alpha_channel = np.ones_like(bg_mask_np, dtype=np.uint8) * 255
    parsing_alpha_channel[bg_mask_np] = 0

    # ** 关键步骤 2：合并原始 Alpha 和解析 Alpha **
    if aligned_input_alpha is not None:
        # 结合逻辑：取两者最小值，即“最透明”的那个。
        # 只有当原始输入不透明 (255) 且解析结果不透明 (255) 时，最终才不透明。
        alpha_channel = np.minimum(aligned_input_alpha, parsing_alpha_channel)
    else:
        # 如果没有输入 mask，则只使用解析结果
        alpha_channel = parsing_alpha_channel
    # ** 结束合并 Alpha **

    # 5. 合并 RGB 和 Alpha 通道，确保 RGBA 顺序
    # align_face_rgb: (H, W, 3) RGB, np.uint8
    # alpha_channel: (H, W) np.uint8
    alpha_channel = alpha_channel[:, :, np.newaxis]

    # 使用 np.concatenate 合并，得到 (H, W, 4) 的 RGBA 图像
    result_rgba = np.concatenate((align_face_rgb, alpha_channel), axis=2)

    # 返回结果数组 (RGBA 格式)
    return result_rgba
