from .nodes import Qwen3_VQA_Plus
from .util_nodes import ImageLoaderPlus, VideoLoaderPlus, VideoLoaderPathPlus
from .path_nodes import MultiplePathsInputPlus

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA_Plus": Qwen3_VQA_Plus,
    "ImageLoaderPlus": ImageLoaderPlus,
    "VideoLoaderPlus": VideoLoaderPlus,
    "VideoLoaderPathPlus": VideoLoaderPathPlus,
    "MultiplePathsInputPlus": MultiplePathsInputPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA_Plus": "Qwen3 VQA Plus",
    "ImageLoaderPlus": "Load Image Advanced Plus",
    "VideoLoaderPlus": "Load Video Advanced Plus",
    "VideoLoaderPathPlus": "Load Video Advanced (Path) Plus",
    "MultiplePathsInputPlus": "Multiple Paths Input Plus",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]