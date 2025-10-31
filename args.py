import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Deep Hashing Training")

    # -----------------------
    # General
    # -----------------------
    parser.add_argument("--save", action="store_true", help="Save model checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default='', help="Directory to save model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")

    # -----------------------
    # Data
    # -----------------------
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "coco", "flickr25k", "nuswide", "imagenet100"],
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="", help="Directory of the dataset")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for training/evaluation")
    parser.add_argument("--resize_size", type=int, default=256, help="Resize size before center crop for evaluation")
    parser.add_argument("--scale_min", type=float, default=0.4, help="Minimum scale for random resized crop")

    # -----------------------
    # Model
    # -----------------------
    parser.add_argument("--encoder", type=str, default="deit", 
                        choices=["dfn", "dinov3", "deit"],
                        help="Encoder backbone")
    parser.add_argument("--hashcoder", type=str, default="small", 
                        choices=["small", "large"],
                        help="Hashcoder variant")
    parser.add_argument("--nlayers", type=int, default=3, help="Number of layers in HashCoder")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension in HashCoder")
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="Bottleneck dimension in HashCoder")

    # -----------------------
    # Training
    # -----------------------
    parser.add_argument("--n_views", type=int, default=2, help="Number of augmented views")
    parser.add_argument("--bitdim", type=int, default=16, help="Number of bits for projection")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA training")
    parser.add_argument("--frozenepochs", type=int, default=1, help="Epochs for head pretraining before LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (for r and alpha)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--coeff_inv", type=float, default=1.0, help="Weight for invariance (augmentation) loss")
    parser.add_argument("--coeff_reg", type=float, default=0.1, help="Weight for regularization loss")
    parser.add_argument("--supervised", action="store_true", help="Use labels for training (supervised mode)")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency (in epochs)")

    # -----------------------
    # Optimization
    # -----------------------
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Linear LR warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate after warmup")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate at the end of schedule")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--eps", type=float, default=0.05, help="Epsilon for total coding rate regularization")

    return parser.parse_args()


args = get_args()
