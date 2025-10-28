# Please download all the required models and set their paths here.
# And configure other necessary environment variables.

# basic sub-dataset save path
DATASET_DIR = "/root/autodl-tmp/IMIG-Dataset/imig-basic"

# composite sub-dataset save path
DATASET_DIR_2 = "/root/autodl-tmp/IMIG-Dataset/imig-composite"

# https://huggingface.co/black-forest-labs/FLUX.1-dev
FLUX_PATH = "/root/autodl-tmp/ckpt/FLUX.1-dev"

# https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha
FLUX_TURBO_PATH = "/root/autodl-tmp/ckpt/FLUX.1-Turbo-Alpha"

# https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
FLUX_KONTEXT_PATH = "/root/autodl-tmp/ckpt/FLUX.1-Kontext-dev"

# https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev
NUNCHAKU_FLUX_KONTEXT_PATH = (
    "/root/autodl-tmp/ckpt/nunchaku-flux.1-kontext-dev/svdq-int4_r32-flux.1-kontext-dev.safetensors"
)

# https://huggingface.co/ByteDance/DreamO
DREAMO_MODEL_PATH = "/root/autodl-tmp/ckpt/DreamO"

# https://huggingface.co/ByteDance-FanQie/MOSAIC optional if you want to use MOSAIC model
MOSAIC_MODEL_PATH = "/root/autodl-tmp/ckpt/MOSAIC"

# https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
GROUNDINGDINO_CKPT_PATH = "/root/autodl-tmp/ckpt/GroundingDINO/groundingdino_swint_ogc.pth"

# https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
# please place the unzipped folder in the `FACE_MODEL_ROOT/'models'` directory
# like /root/autodl-tmp/ckpt/insightface/models/buffalo_l
FACE_MODEL_ROOT = "/root/autodl-tmp/ckpt/insightface"

# https://huggingface.co/facebook/dinov2-giant
DINO_MODEL_ROOT = "/root/autodl-tmp/ckpt/dinov2-giant"


# Your Deepseek API key
DEEPSEEK_API_KEY = ""
