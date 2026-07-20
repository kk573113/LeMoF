# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)

from config import N_CLASSES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cm_to_cols(cm, n_cls=3, prefix="CM"):
    cm = np.asarray(cm, dtype=int)
    out = {}
    for i in range(n_cls):
        for j in range(n_cls):
            out[f"{prefix}_{i}{j}"] = int(cm[i, j])
    return out


def get_clf_eval_mc(name, y_true, proba, elapsed_sec=0.0):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_true, pred)

    if N_CLASSES == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        y_onehot = np.eye(N_CLASSES)[y_true]
        auc = roc_auc_score(y_onehot, proba, average="macro", multi_class="ovr")
    bacc = balanced_accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, pred, labels=list(range(N_CLASSES)))
    row = {
        "Model": name,
        "Accuracy": float(acc),
        "Balanced_Acc": float(bacc),
        "Macro_Precision": float(prec),
        "Macro_Recall": float(rec),
        "Macro_F1": float(f1),
        "Macro_ROC_AUC(ovr)": float(auc),
        "Train_Time": float(elapsed_sec),
    }
    row.update(cm_to_cols(cm, n_cls=N_CLASSES, prefix="CM"))
    return row


def print_stage_header(title):
    print(f"\n{'='*20} Training {title} {'='*20}")


def print_repeat_line(repeat_idx, n_repeats):
    print(f"Repeat {repeat_idx}/{n_repeats}")


def print_result_dict(row, keys=None):
    if keys is None:
        keys = ["Model", "Accuracy", "Macro_F1", "Macro_ROC_AUC(ovr)"]
    out = {k: row.get(k, None) for k in keys}
    print(f"Result: {out}")


def print_stage_summary(rows, stage_name):
    # rows: list of dict
    if len(rows) == 0:
        return
    df = pd.DataFrame(rows)

    cols = []
    for c in ["Accuracy", "Balanced_Acc", "Macro_F1", "Macro_ROC_AUC(ovr)"]:
        if c in df.columns:
            cols.append(c)

    print(f"\n[{stage_name} Summary] (mean over repeats)")
    for c in cols:
        m = df[c].mean()
        s = df[c].std(ddof=0)
        print(f"- {c}: {m:.6f} ± {s:.6f}")
