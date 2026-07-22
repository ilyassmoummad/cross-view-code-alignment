import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import AutoModel
import open_clip
from peft import LoraConfig, get_peft_model
from vit import vit_base
import re


def load_simdinov2(checkpoint_path):
    """
    Load SimDINOv2 ViT-B backbone.
    """
    model = vit_base(patch_size=16, img_size=224, init_values=0.1, block_chunks=0, num_register_tokens=4)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["teacher"]
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    state_dict = {remap_simdino_key(k): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    return model


MODEL_REGISTRY = {
    # DFN
    "dfn": {
        "loader": lambda ckpt: open_clip.create_model_from_pretrained(ckpt),
        "dim": 512,
        "ckpt": "hf-hub:apple/DFN2B-CLIP-ViT-B-16",
        "lora_targets": ["attn"],
    },
    # DeiT
    "deit": {
        "loader": lambda _: vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1),
        "dim": 768,
        "ckpt": None,
        "lora_targets": ["self_attention"],
    },
    # SimDINOv2
    "simdinov2": {
        "loader": load_simdinov2,
        "dim": 768,
        "ckpt": "./checkpoints/vitb16_reg4_SimDNIOv2_ep100.pth",
        "lora_targets": ["qkv"],
    },
    # SWAG
    "swag": {
        "loader": lambda _: vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
        "dim": 768,
        "ckpt": None,
        "lora_targets": ["self_attention"],
    },
    # DINOv2
    "dinov2": {
        "loader": lambda ckpt: AutoModel.from_pretrained(ckpt, device_map="auto"),
        "dim": 768,
        "ckpt": "facebook/dinov2-with-registers-base",
        "lora_targets": ["query", "value"],
    },
    # DINOv3
    "dinov3": {
        "loader": lambda ckpt: AutoModel.from_pretrained(ckpt, device_map="auto"),
        "dim": 768,
        "ckpt": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "lora_targets": ["q_proj", "v_proj"],
    },
}


def remap_simdino_key(key):
    """
    Adapt SimDINOv2 checkpoint naming to local ViT implementation.
    """
    return re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\1.", key)


def get_encoder(args):
    """
    Unified function to get any encoder by name, with optional LoRA.
    
    Args:
        args.encoder: str, encoder name in MODEL_REGISTRY
        args.device: str
        args.lora: bool
        args.lora_rank: int
        args.lora_dropout: float
    Returns:
        encoder, embedding_dim
    """
    key = args.encoder.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported encoder {args.encoder}. Available: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[key]
    ckpt = info["ckpt"]
    encoder_dim = info["dim"]

    # Load encoder
    encoder = info["loader"](ckpt)
    if encoder == 'dfn':
        encoder = encoder[0]

    # For torchvision ViTs, replace classifier head with Identity
    if hasattr(encoder, "heads"):
        encoder.heads = torch.nn.Identity()

    # Apply LoRA if requested
    if getattr(args, "lora", False):
        lora_config = LoraConfig(
            r=getattr(args, "lora_rank", 16),
            lora_alpha=getattr(args, "lora_rank", 16),
            target_modules=info["lora_targets"],
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            bias="none"
        )
        encoder = get_peft_model(encoder, lora_config)
        encoder.print_trainable_parameters()

    encoder = encoder.to(args.device)
    return encoder, encoder_dim


class HashCoder(nn.Module):
    """
    Flexible MLP hashing network with 'small' and 'large' variants.
    """
    def __init__(self, in_dim, bitdim, hashcoder="small"):
        super().__init__()
        self.hashcoder = hashcoder.lower()
        self.bitdim = bitdim
        self.in_dim = in_dim

        if self.hashcoder == "small":
            # 2-layer MLP, simple
            layers = [
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, bitdim),
                nn.BatchNorm1d(bitdim)
            ]
        elif self.hashcoder == "large":
            # 3-layer + bottleneck + GELU
            hidden_dim = max(in_dim * 2, 2048)
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bitdim),
                nn.BatchNorm1d(bitdim)
            ]
        else:
            raise ValueError(f"Unknown HashCoder variant: {hashcoder}. Choose ['small','large'].")

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def get_hashcoder(in_dim, args):
    """
    Returns a HashCoder MLP with proper device placement.
    Args:
        in_dim: int, input feature dimension from encoder
        args.bitdim: int, output hash/code dimension
        args.hashcoder: str, 'small' or 'large'
    """
    coder = HashCoder(in_dim=in_dim, bitdim=args.bitdim, hashcoder=args.hashcoder)
    return coder.to(args.device)
