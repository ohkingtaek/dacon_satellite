from typing import List
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from swin import SwinTransformer


class SwinEncoder(torch.nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [32, 64, 128, 256, 512, 1024]
        self._depth: int = 5
        self._in_channels: int = 3
        kwargs.pop('depth')
        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return outs
    
    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict['model'], strict=False, **kwargs)


def register_encoder():
    smp.encoders.encoders["Swin_Transformer_Encoder"] = {
    "encoder": SwinEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "pretrain_img_size": 224,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        "window_size": 7,
        "drop_path_rate": 0.2,
    }
}

def register_encoder2():
    smp.encoders.encoders["Swin_Transformer_L_Encoder"] = {
    "encoder": SwinEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "pretrain_img_size": 384,
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        "window_size": 12,
        "drop_path_rate": 0.2,
    }
}