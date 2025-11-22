

import os, time, argparse, csv, json, random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt 


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    t0 = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return loss_sum / total, correct / total, time.time() - t0


# ============================================================
# CIFAR-100-C evaluation
# ============================================================

@torch.no_grad()
def evaluate_cifar100c(model, cifar100c_loaders, device):
    corr_err = {}
    for corr_name, sev_loaders in cifar100c_loaders.items():
        correct, total = 0, 0
        for loader in sev_loaders:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        corr_err[corr_name] = 1 - acc

    mCE = float(np.mean(list(corr_err.values())))
    return corr_err, mCE


# ============================================================
# Plotting
# ============================================================

def plot_curves(csv_path, out_dir):
    epochs, tl, ta, vl, va, et = [], [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs.append(int(r["epoch"]))
            tl.append(float(r["train_loss"]))
            ta.append(float(r["train_acc"]))
            vl.append(float(r["val_loss"]))
            va.append(float(r["val_acc"]))
            et.append(float(r["epoch_time_sec"]))

    # Loss curves
    plt.figure()
    plt.plot(epochs, tl, label="Train Loss")
    plt.plot(epochs, vl, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "loss_curves.png", dpi=200); plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, ta, label="Train Acc")
    plt.plot(epochs, va, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "acc_curves.png", dpi=200); plt.close()

    # Epoch time
    plt.figure()
    plt.plot(epochs, et, label="Seconds per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Seconds")
    plt.title("Epoch Runtime"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "epoch_time.png", dpi=200); plt.close()


# ============================================================
# Main
# ============================================================

def main():

    from dataset import get_cifar100_loaders, get_cifar100c_loaders
    from model import resnet18_cifar

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--cifar100c-dir", type=str, default="./cifar100-c")
    parser.add_argument("--out-dir", type=str, default="./runs/baseline_sgd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_loader, val_loader = get_cifar100_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=True
    )

    # Model
    model = resnet18_cifar().to(device)
    print("Device:", device)
    print("Model on:", next(model.parameters()).device)


    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CSV logger
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","train_acc","val_loss","val_acc","lr","epoch_time_sec"]
        )

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, sec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        with open(csv_path, "a") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, val_loss, val_acc, lr, sec])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, out_dir / "best.pt")

        print(f"Epoch {epoch:03d} | TL {tr_loss:.4f} TA {tr_acc:.4f} | "
              f"VL {val_loss:.4f} VA {val_acc:.4f} | {sec:.1f}s | lr {lr:.5f}")

    # Save last checkpoint
    torch.save({"epoch": args.epochs, "state_dict": model.state_dict()}, out_dir / "last.pt")

    # Plots
    plot_curves(csv_path, out_dir)

    # CIFAR-100-C robustness evaluation
    if os.path.exists(args.cifar100c_dir):
        loaders_c = get_cifar100c_loaders(
            args.cifar100c_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        corr_err, mCE = evaluate_cifar100c(model, loaders_c, device)

        with open(out_dir / "cifar100c_results.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(["corruption","error"])
            for c, e in corr_err.items():
                w.writerow([c, e])
            w.writerow(["mCE", mCE])

        print(f"\nCIFAR-100-C mCE: {mCE:.4f}")

    print(f"\nDone. Saved to {out_dir}")


if __name__ == "__main__":
    main()
