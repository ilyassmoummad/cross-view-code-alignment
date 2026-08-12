import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from train import get_features
from typing import Tuple, List
from tqdm import tqdm


def extract_features(encoder, dataloader, device, model_name: str, desc: str = "Extracting Features") -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features and labels using the encoder with model-specific feature handling."""
    features, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc=desc):
            images = images.to(device)
            feat = get_features(encoder, images, model_name)
            features.append(feat.cpu())
            labels.append(lbls.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_map_at_k(retrieval_indices, retrieval_labels, query_labels, topk):
    """Compute Mean Average Precision at k (supports single-label & multi-label)."""
    num_queries = retrieval_indices.size(0)
    APs = []

    is_multilabel = (query_labels.dim() == 2)

    for i in range(num_queries):
        retrieved_indices = retrieval_indices[i, :topk]
        retrieved_labels = retrieval_labels[retrieved_indices]

        if is_multilabel:
            gt_label = query_labels[i].float()
            relevant = (retrieved_labels * gt_label).sum(dim=1) > 0
        else:
            gt_label = query_labels[i].item()
            relevant = (retrieved_labels == gt_label)

        relevant = relevant.float()
        num_relevant = relevant.sum()

        if num_relevant == 0:
            APs.append(0.0)
            continue

        precision_at_k = relevant.cumsum(dim=0) / torch.arange(1, topk + 1, device=retrieved_indices.device).float()
        AP = (precision_at_k * relevant).sum() / num_relevant
        APs.append(AP.item())

    return 100 * sum(APs) / num_queries


def compute_retrieval(query_features, query_logits, database_features, database_codes, map_k, args, retrieval_methods: List[str]) -> dict:
    """
    Compute retrieval results for selected methods.
    retrieval_methods can include: ['continuous', 'asymhamming']
    """
    results = {}

    if 'continuous' in retrieval_methods:
        # Cosine similarity
        dist = torch.mm(F.normalize(query_features.to(args.device), dim=-1),
                        F.normalize(database_features.to(args.device), dim=-1).t())
        _, topk_idx = dist.topk(map_k, largest=True, sorted=True)
        results['continuous'] = topk_idx.cpu()

    if 'asymhamming' in retrieval_methods:
        # Asymmetric Hamming: query logits vs database binary codes
        asymhamming_dist = torch.cdist(torch.sigmoid(query_logits).to(args.device),
                                       database_codes.to(args.device), p=1)
        _, topk_idx = asymhamming_dist.topk(map_k, largest=False, sorted=True)
        results['asymhamming'] = topk_idx.cpu()

    return results


def retrieval(encoder, buckethead, database_loader, query_loader, map_k, args):
    """
    Evaluate retrieval using cosine similarity and asymmetric Hamming only.
    """
    # --- Extract features for database/query ---
    database_features, database_labels = extract_features(
        encoder, database_loader, args.device, args.encoder, "Extracting DB Features"
    )
    query_features, query_labels = extract_features(
        encoder, query_loader, args.device, args.encoder, "Extracting Query Features"
    )

    # --- Compute database logits + codes ---
    database_dataset = TensorDataset(database_features)
    database_dataloader = DataLoader(database_dataset, batch_size=args.bs, shuffle=False)

    database_logits = []
    buckethead.eval()
    with torch.no_grad():
        for features, in database_dataloader:
            logit = buckethead(features.to(args.device))
            database_logits.append(logit.cpu())
    database_logits = torch.cat(database_logits, dim=0)
    database_codes = (database_logits > 0).to(dtype=database_features.dtype)

    # --- Eval loop ---
    retrievals = {k: [] for k in ['continuous', 'asymhamming']}
    query_labels_all = []

    query_dataset = TensorDataset(query_features, query_labels)
    query_dataloader = DataLoader(query_dataset, batch_size=args.bs, shuffle=False)

    for features, labels in tqdm(query_dataloader, desc="Retrieval Evaluation"):
        query_labels_all.append(labels)

        with torch.no_grad():
            query_logit = buckethead(features.to(args.device)).cpu()

        retrieval_results = compute_retrieval(features, query_logit, database_features, database_codes, map_k, args, retrieval_methods=['continuous', 'asymhamming'])
        for k, v in retrieval_results.items():
            retrievals[k].append(v)

    # --- Metrics ---
    query_labels_cat = torch.cat(query_labels_all, dim=0)
    retrievals = {k: torch.cat(v, dim=0) for k, v in retrievals.items()}

    metrics = {
        "map_cont": compute_map_at_k(retrievals['continuous'], database_labels, query_labels_cat, topk=map_k),
        "map_asymhamming": compute_map_at_k(retrievals['asymhamming'], database_labels, query_labels_cat, topk=map_k),
    }
    return metrics


def run_evaluation(encoder, hashcoder, database_loader, query_loader, map_k, args, epoch_metrics):
    metrics = retrieval(encoder, hashcoder, database_loader, query_loader, map_k, args)
    eval_keys = {"Continuous mAP": "map_cont", "AsymHamming mAP": "map_asymhamming"}
    for name, key in eval_keys.items():
        value = metrics[key]
        print(f"{name}@{map_k}: {value:.4f}")
        epoch_metrics[name] = value
    return epoch_metrics
