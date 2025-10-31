import requests
import torch
import os


def download_file_and_fix_paths(url: str, dest_dir: str) -> str:
    """
    Download a file from a URL into dest_dir.
    If file exists, reuse it.
    After downloading, fix any absolute or prefixed paths in the file by rewriting
    them as relative paths (relative to dest_dir).
    Returns the local filepath.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(filepath):
        print(f"[INFO] Downloading {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)

        _fix_paths_in_txt(filepath, dest_dir)
    else:
        print(f"[INFO] {filepath} already exists, skipping download")

    return filepath


def _fix_paths_in_txt(filepath: str, root_dir: str):
    """
    Fix absolute or prefixed paths inside a downloaded .txt split file by converting
    them to relative paths with respect to root_dir.
    This handles absolute paths and removes './' prefixes.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        path = parts[0]

        # Normalize path separators
        path = os.path.normpath(path)

        # Convert absolute paths to relative paths if possible
        if os.path.isabs(path):
            try:
                path = os.path.relpath(path, root_dir)
            except ValueError:
                # Different drive or can't relativize: fallback to basename
                path = os.path.basename(path)
        else:
            # Remove leading './' or '.\' if any
            while path.startswith(".{}" .format(os.sep)):
                path = path[2:]

        fixed_lines.append(" ".join([path] + parts[1:]) + "\n")

    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)


def save_checkpoint(encoder, hashcoder, args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    model_dict = {
        'encoder': encoder.state_dict(),
        'hashcoder': hashcoder.state_dict(),
    }
    model_name = f"{args.encoder}_{args.dataset}_{args.bitdim}.pth"
    if args.supervised:
        model_name = "supervised_" + model_name
    checkpoint_path = os.path.join(args.ckpt_dir, model_name)
    torch.save(model_dict, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")