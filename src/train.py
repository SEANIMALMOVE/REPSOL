"""
EfficientNet training script

Training logic:
1. Forward pass: spectrogram -> model -> logits
2. Loss computation: compare logits with true labels
3. Backward pass: compute gradients
4. Optimizer step: update weights

Before training:
- logits are noisy
- predictions are random
- loss ~ log(num_classes)

After training:
- correct class logits increase
- wrong class logits decrease
- loss decreases
- accuracy improves

NOTE:
- Only model weights change
- Code and data stay the same
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore
from pathlib import Path
import os
import sys
from tqdm import tqdm # type: ignore

# Import modules robustly so `py -m src.train` and `py src/train.py` both work
try:
    from .model import get_model, get_weighted_criterion
    from .dataloaders import get_dataloaders
except Exception:
    try:
        from model import get_model, get_weighted_criterion
        from dataloaders import get_dataloaders
    except Exception:
        from src.model import get_model, get_weighted_criterion
        from src.dataloaders import get_dataloaders


# =========================
# Training class
# =========================

class Trainer:
    def __init__(
        self,
        spectrogram_dir: Path,
        checkpoint_path: Path,
        model_name: str = "efficientnet",
        batch_size: int = 64,
        max_epochs: int = 10,
        patience: int = 3,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name

        # -----------------------
        # Data
        # -----------------------
        # Avoid DataLoader worker hangs when running inside interactive
        # environments (notebooks) on Windows — fall back to num_workers=0.
        interactive = False
        try:
            # If IPython is available and active, assume interactive
            from IPython import get_ipython # type: ignore

            if get_ipython() is not None:
                interactive = True
        except Exception:
            interactive = False

        # sensible default: leave one core free
        try:
            import multiprocessing

            cores = multiprocessing.cpu_count()
        except Exception:
            cores = 2

        effective_num_workers = 0 if interactive else max(1, min(8, cores - 1))

        pin_memory = True if str(device).startswith("cuda") else False

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            spectrogram_dir,
            batch_size=batch_size,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(not interactive),
            prefetch_factor=2,
        )

        self.num_classes = len(self.train_loader.dataset.classes)

        # -----------------------
        # Model
        # -----------------------
        # Fine-tune the EfficientNet backbone
        freeze_backbone = False
        self.model = get_model(model_name=self.model_name, num_classes=self.num_classes, freeze_backbone=freeze_backbone).to(self.device)

        # -----------------------
        # Ensure checkpoint directory and per-model history file exist
        # -----------------------
        try:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Create a unique history file per checkpoint (e.g. baseline_best -> baseline_best_training_history.csv)
        self.history_path = Path(self.checkpoint_path).with_name(f"{Path(self.checkpoint_path).stem}_training_history.csv")
        # If a previous run left artifacts, remove them so this run starts fresh
        try:
            cp = Path(self.checkpoint_path)
            if cp.exists():
                try:
                    cp.unlink()
                except Exception:
                    pass
            if self.history_path.exists():
                try:
                    self.history_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if not self.history_path.exists() or self.history_path.stat().st_size == 0:
                with open(self.history_path, "w", encoding="utf-8") as fh:
                    fh.write("epoch,train_loss,val_loss,train_acc,val_acc,lr\n")
        except Exception:
            # non-fatal; continue without history if filesystem disallows it
            pass

        # Create an initial checkpoint file at start of run so the "best" path exists.
        try:
            # save the empty/initial model to the configured best path only
            self.save_checkpoint()
        except Exception:
            pass

        # -----------------------
        # Training components
        # -----------------------
        # compute class-balanced weights from training labels and use in CrossEntropyLoss
        try:
            train_labels = [label for (_path, label) in self.train_loader.dataset.samples]
            self.criterion = get_weighted_criterion(train_labels, num_classes=self.num_classes, device=self.device)
        except Exception:
            # fallback to unweighted loss if something goes wrong
            self.criterion = nn.CrossEntropyLoss()

        # Use AdamW with a small weight decay for better generalization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # LR scheduler based on validation loss
        # don't pass `verbose` for compatibility with older PyTorch versions
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

        # Use AMP when on CUDA for faster training and lower memory
        self.use_amp = True if (str(device).startswith("cuda") and torch.cuda.is_available()) else False
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # enable cudnn autotuner when input sizes are fixed
        try:
            if str(device).startswith("cuda"):
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # -----------------------
        # Early stopping state
        # -----------------------
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        print("Train samples:", len(self.train_loader.dataset), flush=True)
        print("Train batches:", len(self.train_loader), flush=True)


    # -----------------------
    # One training epoch
    # -----------------------
    def train_one_epoch(self, epoch: int) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        desc = f"Epoch {epoch}/{self.max_epochs} Training"
        pbar = tqdm(
            self.train_loader,
            desc=desc,
            leave=True,
            ncols=100,
            unit="batch",
        )

        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self.model(x)
                    loss = self.criterion(out, y)
                self.scaler.scale(loss).backward()
                # gradient clipping (after unscale)
                try:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                except Exception:
                    pass
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.train_loader)
        acc = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, acc
    # -----------------------
    # Validation
    # -----------------------
    def validate(self, epoch: int) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        desc = f"Epoch {epoch}/{self.max_epochs} Validation"
        pbar = tqdm(
            self.val_loader,
            desc=desc,
            leave=True,
            ncols=100,
            unit="batch",
        )

        with torch.no_grad():
            for x, y in pbar:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        out = self.model(x)
                        loss = self.criterion(out, y)
                else:
                    out = self.model(x)
                    loss = self.criterion(out, y)

                running_loss += loss.item()

                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.val_loader)
        acc = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, acc


    # -----------------------
    # Save checkpoint
    # -----------------------
    def save_checkpoint(self, epoch=None):
        """Save the current model state.

        - Always save to the configured `self.checkpoint_path` (keeps the best model).
        - No epoch-stamped checkpoints or per-epoch history files are created.
        """
        try:
            torch.save(self.model.state_dict(), self.checkpoint_path)
        except Exception as exc:
            print(f"Failed to save checkpoint to {self.checkpoint_path}: {exc}", flush=True)

    # -----------------------
    # Full training loop
    # -----------------------

    def fit(self):
        # use history file created during init; append each epoch
        history_path = getattr(self, "history_path", Path(self.checkpoint_path).with_name(f"{Path(self.checkpoint_path).stem}_training_history.csv"))

        for epoch in range(1, self.max_epochs + 1):

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            print(
                f"Epoch {epoch}/{self.max_epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"| Train Acc: {train_acc:.2f} "
                f"| Val Acc: {val_acc:.2f}",
                flush=True
            )

            # append metrics to CSV for later plotting
            try:
                with open(history_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f},{lr}\n")
            except Exception:
                # non-fatal; continue if filesystem disallows writing history
                pass

            # Scheduler step (ReduceLROnPlateau expects validation metric)
            try:
                self.scheduler.step(val_loss)
            except Exception:
                pass

            # Early stopping: save best checkpoint and stop when no improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                try:
                    self.save_checkpoint()
                    print(f"Saved improved checkpoint to: {self.checkpoint_path}", flush=True)
                except Exception as exc:
                    print(f"Failed to save checkpoint on improvement: {exc}", flush=True)
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs", flush=True)
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} epochs.", flush=True)
                    break

        print("Training finished.", flush=True)


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["efficientnet"], default="efficientnet")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SPECT_DIR = PROJECT_ROOT / "Data" / "Spectrograms"
    CHECKPOINT_PATH = PROJECT_ROOT / f"{args.model}_best.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        spectrogram_dir=SPECT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        model_name=args.model,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=DEVICE,
    )

    trainer.fit()