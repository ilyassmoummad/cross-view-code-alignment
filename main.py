from model import get_encoder, get_hashcoder
from transforms import setup_transforms
from data import get_dataloaders
from util import save_checkpoint
from eval import run_evaluation
from train import train_epoch
from optim import setup_optim
from args import args


def main(args):
    # --- Transforms ---
    train_transform, eval_transform = setup_transforms(args)

    # --- Data ---
    train_loader, database_loader, query_loader, map_k = get_dataloaders(args, train_transform, eval_transform, train_drop_last=True)

    # --- Models ---
    encoder, dim = get_encoder(args)
    hashcoder = get_hashcoder(dim, args)

    # --- Optim & Loss ---
    optimizer, scheduler, criterion_inv, criterion_reg, scaler = setup_optim(encoder, hashcoder, train_loader, args)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        train_loss, ssl_loss, reg_loss = train_epoch(encoder, hashcoder, train_loader, optimizer, criterion_inv, criterion_reg, scaler, scheduler, epoch, args)
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f} SSL Loss: {ssl_loss:.4f} TCR Loss: {reg_loss:.4f}")

        epoch_metrics = {"Train Loss": train_loss, "SSL Loss": ssl_loss, "TCR Loss": reg_loss}

        if epoch % args.eval_freq == 0:
            epoch_metrics = run_evaluation(encoder, hashcoder, database_loader, query_loader, map_k, args, epoch_metrics)

    # --- Save model ---
    if args.save:
        save_checkpoint(encoder, hashcoder, args)


if __name__ == "__main__":
    main(args)
