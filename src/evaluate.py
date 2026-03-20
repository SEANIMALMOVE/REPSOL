import torch # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, zero_division=0),
    }