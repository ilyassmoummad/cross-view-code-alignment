import torch
import torch.nn as nn
from loss import MCR
import numpy as np


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def setup_optim(encoder, hashcoder, train_loader, args):
    """
    Create optimizer, scheduler, criterion, and scaler for training.
    """
    # --- Trainable params ---
    trainable_params = list(encoder.parameters()) + list(hashcoder.parameters())

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    # --- Learning rate scheduler ---
    lr_sched = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs, niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs)
    scheduler = {'lr': lr_sched}

    # --- Losses ---
    criterion_inv = nn.BCEWithLogitsLoss()
    criterion_reg = MCR(eps=args.eps)

    # --- Mixed precision ---
    scaler = torch.cuda.amp.GradScaler()

    return optimizer, scheduler, criterion_inv, criterion_reg, scaler