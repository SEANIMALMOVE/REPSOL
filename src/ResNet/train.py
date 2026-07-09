"""ResNet-50 training script — same structure as EfficientNet trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import sys
from tqdm import tqdm

try:
    from .model import get_model, get_weighted_criterion
    from ..dataloaders import get_dataloaders
except Exception:
    try:
        from ResNet.model import get_model, get_weighted_criterion
        from dataloaders import get_dataloaders
    except Exception:
        from src.ResNet.model import get_model, get_weighted_criterion
        from src.dataloaders import get_dataloaders


class Trainer:
    def __init__(
        self,
        spectrogram_dir: Path,
        checkpoint_path: Path,
        model_name: str = "resnet50",
        batch_size: int = 8,
        max_epochs: int = 15,
        patience: int = 4,
        lr: float = 1e-4,
        device: str = "cpu",
        freeze_backbone: bool = False,
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name

        interactive = False
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                interactive = True
        except Exception:
            pass

        try:
            import multiprocessing
            cores = multiprocessing.cpu_count()
        except Exception:
            cores = 2

        effective_num_workers = 0 if interactive else max(1, min(8, cores - 1))
        pin_memory = str(device).startswith("cuda")

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            spectrogram_dir,
            batch_size=batch_size,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(not interactive),
            prefetch_factor=2,
        )

        self.num_classes = len(self.train_loader.dataset.classes)

        self.model = get_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path = self.checkpoint_path.with_name(
            f"{self.checkpoint_path.stem}_training_history.csv"
        )

        # Clear any leftover artifacts from a previous run
        for p in (self.checkpoint_path, self.history_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        try:
            with open(self.history_path, "w", encoding="utf-8") as fh:
                fh.write("epoch,train_loss,val_loss,train_acc,val_acc,lr\n")
        except Exception:
            pass

        try:
            self.save_checkpoint()
        except Exception:
            pass

        try:
            train_labels = [label for (_, label) in self.train_loader.dataset.samples]
            self.criterion = get_weighted_criterion(
                train_labels, num_classes=self.num_classes, device=self.device
            )
        except Exception:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

        self.use_amp = str(device).startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if str(device).startswith("cuda"):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        print("Train samples:", len(self.train_loader.dataset), flush=True)
        print("Train batches:", len(self.train_loader), flush=True)

    def train_one_epoch(self, epoch: int) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.max_epochs} Training",
            leave=True, ncols=100, unit="batch",
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

        return running_loss / len(self.train_loader), 100.0 * correct / total if total > 0 else 0.0

    def validate(self, epoch: int) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.max_epochs} Validation",
            leave=True, ncols=100, unit="batch",
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

        return running_loss / len(self.val_loader), 100.0 * correct / total if total > 0 else 0.0

    def save_checkpoint(self, epoch=None):
        try:
            torch.save(self.model.state_dict(), self.checkpoint_path)
        except Exception as exc:
            print(f"Failed to save checkpoint: {exc}", flush=True)

    def fit(self):
        history_path = getattr(
            self, "history_path",
            self.checkpoint_path.with_name(f"{self.checkpoint_path.stem}_training_history.csv"),
        )

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            print(
                f"Epoch {epoch}/{self.max_epochs} "
                f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                f"| Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}",
                flush=True,
            )

            try:
                with open(history_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f},{lr}\n")
            except Exception:
                pass

            try:
                self.scheduler.step(val_loss)
            except Exception:
                pass

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                try:
                    self.save_checkpoint()
                    print(f"Saved improved checkpoint to: {self.checkpoint_path}", flush=True)
                except Exception as exc:
                    print(f"Failed to save checkpoint: {exc}", flush=True)
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs", flush=True)
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} epochs.", flush=True)
                    break

        print("Training finished.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SPECT_DIR = PROJECT_ROOT / "Data" / "Spectrograms"
    CHECKPOINT_PATH = PROJECT_ROOT / "Models_output" / "resnet50_best.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        spectrogram_dir=SPECT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        model_name="resnet50",
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=DEVICE,
    )
    trainer.fit()
