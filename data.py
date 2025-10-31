import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from util import download_file_and_fix_paths


# --- Base dataset for text splits ---
class BaseDataset(Dataset):
    """Generic dataset for image paths listed in a text file (with labels)."""
    def __init__(self, root: str, list_path: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                rel_path = parts[0].lstrip("./")
                full_path = os.path.join(root, rel_path)
                if not os.path.exists(full_path):
                    full_path = os.path.join(root, os.path.basename(rel_path))
                labels = torch.tensor([int(x) for x in parts[1:]], dtype=torch.float32)
                self.samples.append((full_path, labels))

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)


# --- Dataset registry for text-based datasets ---
DATASET_REGISTRY = {
    "flickr25k": {
        "class": BaseDataset,
        "base_url": "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/mirflickr/",
        "map_k": 5000
    },
    "nuswide": {
        "class": BaseDataset,
        "base_url": "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/nuswide_21/",
        "map_k": 5000
    },
    "coco": {
        "class": BaseDataset,
        "base_url": "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/coco/",
        "map_k": 5000
    },
    "imagenet100": {
        "class": BaseDataset,
        "base_url": "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/imagenet/",
        "map_k": 1000
    },
}


# --- Helper to create a DataLoader for a split ---
def _get_loader(root: str, split: str, transform, dataset_info, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool):
    """Download split file, create dataset, and return a DataLoader."""
    txt_file = download_file_and_fix_paths(f"{dataset_info['base_url']}{split}.txt", root)
    dataset = dataset_info["class"](root, txt_file, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)


# --- Unified dataloader creator ---
def get_dataloaders(args, train_transform=None, eval_transform=None, train_drop_last=False):
    """
    Create train, database, and query DataLoaders for supported datasets.
    Handles CIFAR10 and text-based datasets in a unified way.
    """
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transform, download=True)
        database_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=eval_transform, download=True)
        query_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=eval_transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                  num_workers=args.n_workers, pin_memory=False, drop_last=train_drop_last)
        database_loader = DataLoader(database_dataset, batch_size=args.bs, shuffle=False,
                                     num_workers=args.n_workers, pin_memory=False)
        query_loader = DataLoader(query_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=args.n_workers, pin_memory=False)
        map_k = 1000

    elif args.dataset in DATASET_REGISTRY:
        dataset_info = DATASET_REGISTRY[args.dataset]

        train_loader = _get_loader(args.data_dir, "train", train_transform, dataset_info,
                                   batch_size=args.bs, num_workers=args.n_workers, shuffle=True, drop_last=train_drop_last)
        database_loader = _get_loader(args.data_dir, "database", eval_transform, dataset_info,
                                      batch_size=args.bs, num_workers=args.n_workers, shuffle=False, drop_last=False)
        query_loader = _get_loader(args.data_dir, "test", eval_transform, dataset_info,
                                   batch_size=args.bs, num_workers=args.n_workers, shuffle=False, drop_last=False)
        map_k = dataset_info["map_k"]

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_loader, database_loader, query_loader, map_k