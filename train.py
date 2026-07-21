import torch
from tqdm import tqdm


def get_features(encoder, x, model_name: str):
    """Extract features depending on model type."""
    if "simdinov2" in model_name.lower():
        return encoder(x)["x_norm_clstoken"]
    elif "dinov2" in model_name.lower():
        return encoder(x).pooler_output
    elif "dinov3" in model_name.lower():
        return encoder(x).pooler_output
    elif "dfn" in model_name.lower():
        return encoder.encode_image(x)
    elif "deit" in model_name.lower() or "swag" in model_name.lower():
        return encoder(x)
    else:
        raise ValueError(f"Unknown model_name={model_name}, cannot extract features.")


def train_epoch(encoder, projector, train_loader, optimizer, criterion_inv, criterion_reg, scaler, scheduler, epoch, args):
    """
    Train for one epoch (supervised or unsupervised depending on flag).
    
    Args:
        encoder: Feature extractor.
        projector: Projection head.
        train_loader: DataLoader with batches of (images, labels).
        optimizer: Optimizer.
        criterion_inv: Loss for invariance (e.g., BCE).
        criterion_reg: Regularization loss.
        scaler: GradScaler for AMP.
        scheduler: Dict with "lr" (per-step learning rate schedule).
        epoch: Current epoch index.
        args: Namespace with training hyperparams (needs args.encoder).
    """
    freeze_encoder = epoch < args.frozenepochs

    encoder.eval() if freeze_encoder else encoder.train()
    projector.train()

    total_loss, total_ssl, total_reg = 0.0, 0.0, 0.0
    n_batches = 0

    for idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training Iteration")):
        # images is a list/tuple of multiple views
        views = [img.to(args.device) for img in images]

        # --- Scheduler update ---
        it = len(train_loader) * epoch + idx
        for g in optimizer.param_groups:
            g["lr"] = scheduler["lr"][it]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # --- Encode features ---
            if freeze_encoder:
                with torch.no_grad():
                    feats = [get_features(encoder, v, args.encoder) for v in views]
            else:
                feats = [get_features(encoder, v, args.encoder) for v in views]

            # --- Project ---
            projs = [projector(f) for f in feats]

            # --- Supervised or Unsupervised coding ---
            if args.supervised:
                # if labels.ndim > 1:  # one-hot → class indices
                #     labels = labels.argmax(dim=1)

                # all_proj = torch.cat(projs, dim=0)
                # all_labels = torch.cat([labels for _ in projs], dim=0)

                # # class means
                # class_means = {
                #     c.item(): all_proj[all_labels == c].mean(dim=0, keepdim=True)
                #     for c in all_labels.unique()
                # }

                # # each view uses codes from its class mean
                # all_codes = [
                #     torch.cat(
                #         [(class_means[c.item()] > 0).to(all_proj.dtype) for c in labels],
                #         dim=0,
                #     )
                #     for _ in projs
                # ]
                
                # labels:
                # mono-label: [B]
                # multi-label: [B, C]
                # if labels.ndim == 1:
                #     labels = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).float()
                # else:
                #     labels = labels.float()

                labels = labels.to(args.device)

                # Combine projections from all views
                all_proj = torch.cat(projs, dim=0) # [B*num_views, D]
                all_labels = torch.cat([labels for _ in projs],  dim=0) # [B*num_views, C]

                # ---- Compute label prototypes ----
                # [C,D] = [C,B] @ [B,D]
                label_means = all_labels.T @ all_proj

                # normalize by number of samples per label
                label_counts = all_labels.sum(dim=0, keepdim=True).T
                label_means = label_means / label_counts.clamp_min(1)

                # ---- Build image prototypes from active labels ----
                # [B,D] = [B,C] @ [C,D]
                sample_codes = labels @ label_means

                # average over active labels
                sample_label_counts = labels.sum(dim=1, keepdim=True)
                sample_codes = sample_codes / sample_label_counts.clamp_min(1)

                # binary codes
                sample_codes = (sample_codes > 0).to(all_proj.dtype)

                # one target per view
                all_codes = [sample_codes for _ in projs]

            else:  # unsupervised: threshold per-view projection
                all_codes = [(p > 0).to(p.dtype) for p in projs]

            # --- SSL loss (averaged across all pairs of views) ---
            ssl_losses = []
            for i, p in enumerate(projs):
                for j, codes in enumerate(all_codes):
                    if i != j:
                        ssl_losses.append(criterion_inv(p, codes))
            loss_inv = sum(ssl_losses) / len(ssl_losses)

            # --- Regularization loss (average over all views) ---
            loss_reg = sum(criterion_reg(p) for p in projs) / len(projs)

            # --- Total loss ---
            loss = args.coeff_inv * loss_inv + args.coeff_reg * loss_reg

        # --- Backward + Step ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- Logging accumulators ---
        total_loss += loss.item()
        total_ssl += loss_inv.item()
        total_reg += loss_reg.item()
        n_batches += 1

    # --- Averages ---
    avg_loss = total_loss / n_batches
    avg_ssl = total_ssl / n_batches
    avg_reg = total_reg / n_batches

    return avg_loss, avg_ssl, avg_reg