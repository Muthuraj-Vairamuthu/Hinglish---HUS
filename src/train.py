"""Training script for hallucination classification model."""
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from .schema import LabeledExample

# Updated to match new label scheme: N, H-, H+
LABEL2ID = {"N": 0, "H-": 1, "H+": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


class MLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_input(ex: LabeledExample) -> str:
    """Build text input — include variant so model is aware of syntactic context."""
    return f"[TASK={ex.task_type}][VARIANT={ex.variant}]\nPrompt: {ex.prompt}\nOutput: {ex.response}"


def main():
    path = "data/labeled_PRESSURED.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        data = [LabeledExample.model_validate_json(line) for line in f]

    X_text = [build_input(x) for x in data]
    y = np.array([LABEL2ID[x.label] for x in data], dtype=np.int64)

    # Class distribution
    unique_classes = np.unique(y)
    print(f"\nDataset stats: {len(data)} examples")
    for cls in unique_classes:
        count = np.sum(y == cls)
        print(f"  {ID2LABEL[cls]}: {count} ({count/len(y)*100:.1f}%)")

    encoder = SentenceTransformer("all-mpnet-base-v2")
    X = encoder.encode(
        X_text, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    # Stratify only if viable
    min_class_count = min(np.sum(y == cls) for cls in unique_classes)
    can_stratify = len(unique_classes) >= 2 and min_class_count >= 2

    if not can_stratify:
        print("\nWarning: Not enough samples per class for stratification.")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42,
        stratify=y if can_stratify else None
    )

    unique_tmp    = np.unique(y_tmp)
    min_tmp_count = min(np.sum(y_tmp == cls) for cls in unique_tmp)
    can_stratify_tmp = len(unique_tmp) >= 2 and min_tmp_count >= 2

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42,
        stratify=y_tmp if can_stratify_tmp else None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MLP(in_dim=X.shape[1]).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(Xb, yb, train=True):
        model.train(train)
        xb = torch.tensor(Xb, dtype=torch.float32, device=device)
        yb = torch.tensor(yb, dtype=torch.long,    device=device)
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        return float(loss.detach().cpu()), preds

    best_val, best_state = 1e9, None

    for epoch in range(1, 21):
        tr_loss, _         = run_epoch(X_train, y_train, train=True)
        val_loss, val_preds = run_epoch(X_val,   y_val,   train=False)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    _, test_preds = run_epoch(X_test, y_test, train=False)

    test_classes      = np.unique(np.concatenate([y_test, test_preds]))
    test_target_names = [ID2LABEL[i] for i in test_classes]

    print("\nClassification report:")
    print(classification_report(
        y_test, test_preds,
        labels=test_classes, target_names=test_target_names, zero_division=0
    ))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_preds))

    torch.save(model.state_dict(), "data/hinglish_hus_mlp.pt")
    print("Saved: data/hinglish_hus_mlp.pt")


if __name__ == "__main__":
    main()