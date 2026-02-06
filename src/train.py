import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None
from torch.utils.data import DataLoader, Dataset, random_split

from src.data_processing import build_lit_tensor, load_fi2010_file, parse_fi2010_levels
from src.eval import compute_basic_metrics, plot_confusion_matrix
from src.labels import build_directional_labels, estimate_tick_size
from src.models import LitLSTMModel, LitTransModel
from src.utils import ensure_dir, load_yaml, save_json, set_seed


class LitDataset(Dataset):
    def __init__(self, lit_tensor: np.ndarray, labels: np.ndarray, seq_len: int = 64):
        self.x = torch.from_numpy(lit_tensor).float()
        self.labels = torch.from_numpy(labels).long()
        self.seq_len = seq_len
        self.length = len(self.x) - seq_len + 1
        if self.length <= 0:
            raise ValueError("seq_len is too large for dataset")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x_seq = self.x[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x_seq, y


def collate_batch(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


@dataclass
class TrainState:
    best_f1: float = -1.0
    best_epoch: int = 0
    patience_left: int = 0
    best_state: Dict[str, torch.Tensor] | None = None


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights * (num_classes / weights.sum())
    return weights


def train_one_epoch(model, loader, optimizer, device, criterion, use_ovr, decision_threshold, show_progress):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    iterable = loader
    if show_progress and tqdm is not None:
        iterable = tqdm(loader, desc="Train", leave=False)

    for xb, yb in iterable:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)

        if use_ovr:
            targets = F.one_hot(yb, num_classes=logits.size(1)).float()
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)
            preds = probs.argmax(dim=1)
            if decision_threshold is not None:
                max_prob, _ = probs.max(dim=1)
                preds = torch.where(max_prob < decision_threshold, torch.tensor(1, device=device), preds)
        else:
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (preds == yb).sum().item()
        total_samples += xb.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(model, loader, device, num_classes, criterion, use_ovr, decision_threshold, show_progress):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    iterable = loader
    if show_progress and tqdm is not None:
        iterable = tqdm(loader, desc="Eval", leave=False)

    with torch.no_grad():
        for xb, yb in iterable:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if use_ovr:
                targets = F.one_hot(yb, num_classes=num_classes).float()
                loss = criterion(logits, targets)
            else:
                loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if use_ovr:
        probs = torch.sigmoid(logits)
        preds = probs.argmax(dim=1)
        if decision_threshold is not None:
            max_prob, _ = probs.max(dim=1)
            preds = torch.where(max_prob < decision_threshold, torch.tensor(1), preds)
    else:
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

    metrics = compute_basic_metrics(labels.numpy(), preds.numpy())
    metrics["loss"] = total_loss / total_samples
    metrics["f1_macro"] = float(f1_score(labels.numpy(), preds.numpy(), average="macro"))
    return metrics, labels.numpy(), preds.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(str(config_path))
    project_root = config_path.parent.parent
    set_seed(cfg.get("seed", 42))

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    show_progress = bool(cfg.get("show_progress", True))

    train_path = Path(cfg["train_path"])
    if not train_path.is_absolute():
        train_path = (project_root / train_path).resolve()
    test_path = Path(cfg["val_path"])
    if not test_path.is_absolute():
        test_path = (project_root / test_path).resolve()

    x_train, _ = load_fi2010_file(str(train_path))
    x_test, _ = load_fi2010_file(str(test_path))

    parsed_train = parse_fi2010_levels(x_train, cfg["levels_per_side"], cfg["fi2010_order"])
    best_bid_train = parsed_train["bid_px"][:, 0]
    best_ask_train = parsed_train["ask_px"][:, 0]
    mid_price_train = (best_ask_train + best_bid_train) / 2.0

    tick_size, tick_source = estimate_tick_size(best_bid_train, best_ask_train, mid_price_train)
    label_matrix_train = build_directional_labels(
        mid_price_train,
        cfg["horizons"],
        tick_size,
        neutral_band_ticks=cfg["neutral_band_ticks"],
    )

    lit_train, _, scalers, _ = build_lit_tensor(
        x_train,
        levels_per_side=cfg["levels_per_side"],
        order=cfg["fi2010_order"],
        train_end=int(0.7 * len(x_train)),
        scalers=None,
        use_extra=cfg.get("c_in_extra", True),
    )

    parsed_test = parse_fi2010_levels(x_test, cfg["levels_per_side"], cfg["fi2010_order"])
    best_bid_test = parsed_test["bid_px"][:, 0]
    best_ask_test = parsed_test["ask_px"][:, 0]
    mid_price_test = (best_ask_test + best_bid_test) / 2.0

    label_matrix_test = build_directional_labels(
        mid_price_test,
        cfg["horizons"],
        tick_size,
        neutral_band_ticks=cfg["neutral_band_ticks"],
    )

    lit_test, _, _, _ = build_lit_tensor(
        x_test,
        levels_per_side=cfg["levels_per_side"],
        order=cfg["fi2010_order"],
        train_end=None,
        scalers=scalers,
        use_extra=cfg.get("c_in_extra", True),
    )

    horizon_idx = cfg["horizon_idx"]
    y_train = label_matrix_train[horizon_idx]
    y_test = label_matrix_test[horizon_idx]

    dataset = LitDataset(lit_train, y_train, seq_len=cfg["seq_len"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    test_dataset = LitDataset(lit_test, y_test, seq_len=cfg["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_batch)

    num_classes = 3
    c_in = lit_train.shape[2]
    ph = cfg["levels_per_side"]

    if cfg["model"] == "lstm":
        model = LitLSTMModel(
            c_in=c_in,
            d_model=cfg["d_model"],
            pw=4,
            ph=ph,
            conv_channels=cfg["conv_channels"],
            num_classes=num_classes,
            dropout=cfg["dropout"],
            use_ovr=cfg["use_ovr"],
        ).to(device)
    else:
        model = LitTransModel(
            c_in=c_in,
            d_model=cfg["d_model"],
            pw=4,
            ph=ph,
            conv_channels=cfg["conv_channels"],
            pooling=cfg["pooling"],
            num_classes=num_classes,
            n_heads=cfg["n_heads"],
            ff_dim=cfg["ff_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            use_ovr=cfg["use_ovr"],
        ).to(device)

    use_ovr = cfg["use_ovr"]
    decision_threshold = cfg.get("decision_threshold", None)

    if use_ovr:
        manual_pos_weight = cfg.get("ovr_pos_weight")
        if manual_pos_weight:
            class_weights = torch.tensor(manual_pos_weight, dtype=torch.float32, device=device)
        else:
            class_weights = compute_class_weights(torch.tensor(y_train), num_classes=num_classes).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        class_weights = compute_class_weights(torch.tensor(y_train), num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    state = TrainState(patience_left=cfg["patience"])

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            use_ovr,
            decision_threshold,
            show_progress,
        )
        val_metrics, _, _ = eval_epoch(
            model,
            val_loader,
            device,
            num_classes,
            criterion,
            use_ovr,
            decision_threshold,
            show_progress,
        )

        if val_metrics["f1_macro"] > state.best_f1:
            state.best_f1 = val_metrics["f1_macro"]
            state.best_epoch = epoch
            state.patience_left = cfg["patience"]
            state.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            state.patience_left -= 1
            if state.patience_left == 0:
                break

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

    if state.best_state is not None:
        model.load_state_dict(state.best_state)

    test_metrics, y_true, y_pred = eval_epoch(
        model,
        test_loader,
        device,
        num_classes,
        criterion,
        use_ovr,
        decision_threshold,
        show_progress,
    )

    run_name = cfg["run_name"]
    ensure_dir("results/metrics")
    ensure_dir("results/figures")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["down", "flat", "up"],
        output_dict=True,
        zero_division=0,
    )

    metrics_payload = {
        "run_name": run_name,
        "tick_size": tick_size,
        "tick_source": tick_source,
        "best_epoch": state.best_epoch,
        "val_best_f1": state.best_f1,
        "test": test_metrics,
        "confusion_matrix": cm.tolist(),
        "per_class": report,
    }
    save_json(f"results/metrics/{run_name}.json", metrics_payload)
    plot_confusion_matrix(y_true, y_pred, save_path=f"results/figures/{run_name}_cm.png")

    print("Saved metrics and confusion matrix.")


if __name__ == "__main__":
    main()
