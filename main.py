# -*- coding: utf-8 -*-
import os
import copy
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

import torch

from config import (
    device, MAX_LEN, SEED_BASE, N_REPEATS, ENCODER_EPOCHS, FUSION_EPOCHS,
    BATCH_SIZE, N_CLASSES, ECG_BASE, CSV_PATH, OUT_DIR, CAT_COLS,
    ECG_MODEL_NAMES, TAB_MODEL_NAMES,
)
from utils import set_seed, get_clf_eval_mc, print_result_dict
from dataset import get_loader
from fusion_model import CoAttention
from train_utils import train_model_early_stop, predict_heads_proba, predict_fusion_proba
from model_builders import build_ecg_model, build_tab_model

warnings.filterwarnings("ignore")
print(f"Using Device: {device}")


if __name__ == "__main__":
    set_seed(SEED_BASE)

    # 1) Load Data
    df = pd.read_csv(CSV_PATH)
    # df = df[df['los_class'].isin([0, 1, 2])].reset_index(drop=True)
    df['icu_los_class'] = df['icu_los_days'].apply(lambda x: 1 if x > 3 else 0)
    # target_col = 'los_class'
    target_col = 'icu_los_class'

    ignore_cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'ecg_path', 'icu_mortality',
                   'hosp_mortality', 'readmission_30d', 'sapsii', 'icu_los_days', 'icu_los_class',
                   'anchor_year', 'hosp_los_days', 'admittime', 'dischtime']

    available_cols = [c for c in df.columns if c not in ignore_cols + [target_col]]
    cat_features = [c for c in CAT_COLS if c in available_cols]
    num_features = [c for c in available_cols if c not in cat_features]

    cat_dims = []
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_dims.append(len(le.classes_))
    cat_dims = tuple(cat_dims)

    X_num = df[num_features].values.astype(np.float32)
    X_cat = df[cat_features].values.astype(np.float32)
    X_tab_all = np.hstack([X_num, X_cat])

    x_ecg_all, valid_indices = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading ECG"):
        try:
            path = os.path.join(ECG_BASE, row["ecg_path"])
            if os.path.exists(path + ".dat"):
                sig, _ = wfdb.rdsamp(path)
                if len(sig) < MAX_LEN:
                    sig = np.pad(sig, ((0, MAX_LEN - len(sig)), (0, 0)), "constant")
                else:
                    sig = sig[:MAX_LEN]
                if not np.isnan(sig).any():
                    x_ecg_all.append(sig.astype(np.float32))
                    valid_indices.append(idx)
        except Exception:
            continue

    x_ecg_all = np.array(x_ecg_all)
    y_all = df.loc[valid_indices, target_col].values.astype(int)
    X_tab_all = X_tab_all[valid_indices]

    # 2) Split & Scale
    x_tab_train, x_tab_temp, x_ecg_train, x_ecg_temp, y_train, y_temp = train_test_split(
        X_tab_all, x_ecg_all, y_all, test_size=0.4, stratify=y_all, random_state=SEED_BASE
    )
    x_tab_val, x_tab_test, x_ecg_val, x_ecg_test, y_val, y_test = train_test_split(
        x_tab_temp, x_ecg_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED_BASE
    )

    num_cnt = len(num_features)
    scaler = StandardScaler()
    x_tab_train[:, :num_cnt] = scaler.fit_transform(x_tab_train[:, :num_cnt])
    x_tab_val[:, :num_cnt] = scaler.transform(x_tab_val[:, :num_cnt])
    x_tab_test[:, :num_cnt] = scaler.transform(x_tab_test[:, :num_cnt])

    scaler_ecg = StandardScaler()
    N_tr, L, C = x_ecg_train.shape
    x_ecg_train = scaler_ecg.fit_transform(x_ecg_train.reshape(-1, C)).reshape(N_tr, L, C)
    x_ecg_val = scaler_ecg.transform(x_ecg_val.reshape(-1, C)).reshape(x_ecg_val.shape[0], L, C)
    x_ecg_test = scaler_ecg.transform(x_ecg_test.reshape(-1, C)).reshape(x_ecg_test.shape[0], L, C)

    dl_train = get_loader(x_ecg_train, x_tab_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = get_loader(x_ecg_val, x_tab_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    dl_test = get_loader(x_ecg_test, x_tab_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    counts = np.bincount(y_train, minlength=N_CLASSES)
    class_weights = (len(y_train) / (N_CLASSES * np.maximum(counts, 1))).tolist()

    results = {"single": [], "m1": [], "m2": [], "final": []}

    tab_dim = x_tab_train.shape[1]

    # =========================================================
    # repeat
    # =========================================================
    for repeat in range(N_REPEATS):
        set_seed(SEED_BASE + repeat)
        print(f"\n==================== REPEAT {repeat+1}/{N_REPEATS} ====================")

        ecg_cache = {}
        tab_cache = {}

        # -----------------------------
        # (A) ECG
        # -----------------------------
        for ecg_name in ECG_MODEL_NAMES:
            set_seed(SEED_BASE + repeat)
            print(f"[Train ECG] {ecg_name}")

            ecg_model = build_ecg_model(ecg_name)

            t_start = time.time()
            ecg_model = train_model_early_stop(
                ecg_model, dl_train, dl_val, epochs=ENCODER_EPOCHS, class_weights=class_weights
            )
            train_time = time.time() - t_start

            ecg_val_p = predict_heads_proba(ecg_model, dl_val, True)   # (Nval, 3, C)
            ecg_test_p = predict_heads_proba(ecg_model, dl_test, True)  # (Ntest, 3, C)

            ecg_val_flat = ecg_val_p.reshape(len(y_val), -1)
            ecg_test_flat = ecg_test_p.reshape(len(y_test), -1)

            meta_ecg = LogisticRegression(max_iter=1000).fit(ecg_val_flat, y_val)

            explainer = shap.LinearExplainer(meta_ecg, ecg_val_flat)
            shap_values = explainer.shap_values(ecg_val_flat)

            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            head_importance = mean_abs.reshape(3, N_CLASSES).sum(axis=1)
            ecg_best_idx = int(np.argmax(head_importance))

            ecg_val_stack_p = meta_ecg.predict_proba(ecg_val_flat)
            ecg_test_stack_p = meta_ecg.predict_proba(ecg_test_flat)

            row_single_ecg = get_clf_eval_mc(f"ECG_{ecg_name}", y_test, ecg_test_stack_p, train_time)
            row_single_ecg["Repeat"] = repeat
            row_single_ecg["Modality"] = "ECG"
            results["single"].append(row_single_ecg)
            print_result_dict(
                {"Model": f"ECG_{ecg_name}", "Accuracy": row_single_ecg["Accuracy"],
                 "Macro_F1": row_single_ecg["Macro_F1"], "Macro_ROC_AUC(ovr)": row_single_ecg["Macro_ROC_AUC(ovr)"]},
                keys=["Model", "Accuracy", "Macro_F1", "Macro_ROC_AUC(ovr)"]
            )

            ecg_cache[ecg_name] = {
                "state_dict": copy.deepcopy(ecg_model.state_dict()),
                "best_head": int(ecg_best_idx),
                "val_heads": ecg_val_p,      # (Nval, 3, C)
                "test_heads": ecg_test_p,    # (Ntest, 3, C)
                "val_stack": ecg_val_stack_p,
                "test_stack": ecg_test_stack_p,
            }

            del ecg_model
            torch.cuda.empty_cache()

        # -----------------------------
        # (B) Tab
        # -----------------------------
        for tab_name in TAB_MODEL_NAMES:
            set_seed(SEED_BASE + repeat)
            print(f"[Train Tab] {tab_name}")

            tab_model = build_tab_model(tab_name, tab_dim, num_cnt, cat_dims)

            t_start = time.time()
            tab_model = train_model_early_stop(
                tab_model, dl_train, dl_val, epochs=ENCODER_EPOCHS, class_weights=class_weights
            )
            train_time = time.time() - t_start

            tab_val_p = predict_heads_proba(tab_model, dl_val, False)
            tab_test_p = predict_heads_proba(tab_model, dl_test, False)

            tab_val_flat = tab_val_p.reshape(len(y_val), -1)
            tab_test_flat = tab_test_p.reshape(len(y_test), -1)

            meta_tab = LogisticRegression(max_iter=1000).fit(tab_val_flat, y_val)
            tab_val_stack_p = meta_tab.predict_proba(tab_val_flat)
            tab_test_stack_p = meta_tab.predict_proba(tab_test_flat)

            explainer = shap.LinearExplainer(meta_tab, tab_val_flat)
            shap_values = explainer.shap_values(tab_val_flat)

            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            head_importance = mean_abs.reshape(3, N_CLASSES).sum(axis=1)
            tab_best_idx = int(np.argmax(head_importance))

            row_single_tab = get_clf_eval_mc(f"Tab_{tab_name}", y_test, tab_test_stack_p, train_time)
            row_single_tab["Repeat"] = repeat
            row_single_tab["Modality"] = "Tab"
            results["single"].append(row_single_tab)
            print_result_dict(
                {"Model": f"Tab_{tab_name}", "Accuracy": row_single_tab["Accuracy"],
                 "Macro_F1": row_single_tab["Macro_F1"], "Macro_ROC_AUC(ovr)": row_single_tab["Macro_ROC_AUC(ovr)"]},
                keys=["Model", "Accuracy", "Macro_F1", "Macro_ROC_AUC(ovr)"]
            )

            tab_cache[tab_name] = {
                "state_dict": copy.deepcopy(tab_model.state_dict()),
                "best_head": int(tab_best_idx),
                "val_heads": tab_val_p,
                "test_heads": tab_test_p,
                "val_stack": tab_val_stack_p,
                "test_stack": tab_test_stack_p,
            }

            del tab_model
            torch.cuda.empty_cache()

        # -----------------------------
        # (C) M1 / M2 / Final
        # -----------------------------
        for ecg_name in ECG_MODEL_NAMES:
            for tab_name in TAB_MODEL_NAMES:
                set_seed(SEED_BASE + repeat)

                print(f"\n[Combo] {ecg_name} + {tab_name} (Repeat {repeat+1}/{N_REPEATS})")

                E = ecg_cache[ecg_name]
                T = tab_cache[tab_name]

                ecg_best_idx = E["best_head"]
                tab_best_idx = T["best_head"]

                m1_train = np.hstack([
                    E["val_stack"],
                    E["val_heads"][:, ecg_best_idx, :],
                    T["val_stack"],
                    T["val_heads"][:, tab_best_idx, :],
                ])
                m1_test = np.hstack([
                    E["test_stack"],
                    E["test_heads"][:, ecg_best_idx, :],
                    T["test_stack"],
                    T["test_heads"][:, tab_best_idx, :],
                ])

                lr_m1 = LogisticRegression(max_iter=1000).fit(m1_train, y_val)
                m1_pred = lr_m1.predict_proba(m1_test)

                row_m1 = get_clf_eval_mc(f"M1_{ecg_name}_{tab_name}", y_test, m1_pred, 0.0)
                row_m1["Repeat"] = repeat
                results["m1"].append(row_m1)
                print_result_dict(
                    {"Model": f"M1_{ecg_name}_{tab_name}", "Accuracy": row_m1["Accuracy"],
                     "Macro_F1": row_m1["Macro_F1"], "Macro_ROC_AUC(ovr)": row_m1["Macro_ROC_AUC(ovr)"]},
                    keys=["Model", "Accuracy", "Macro_F1", "Macro_ROC_AUC(ovr)"]
                )
                # M2 Fusion
                ecg_model = build_ecg_model(ecg_name)
                tab_model = build_tab_model(tab_name, tab_dim, num_cnt, cat_dims)

                ecg_model.load_state_dict(E["state_dict"])
                tab_model.load_state_dict(T["state_dict"])

                fusion_model = CoAttention(ecg_model, tab_model, ecg_best_idx, tab_best_idx).to(device)

                # freeze
                fusion_model.ecg_encoder.freeze_by_index(ecg_best_idx)
                fusion_model.tab_encoder.freeze_by_index(tab_best_idx)

                fusion_model = train_model_early_stop(
                    fusion_model, dl_train, dl_val,
                    epochs=FUSION_EPOCHS, is_fusion=True,
                    class_weights=class_weights
                )

                m2_pred = predict_fusion_proba(fusion_model, dl_test)
                m2_val_pred = predict_fusion_proba(fusion_model, dl_val)

                row_m2 = get_clf_eval_mc(f"M2_{ecg_name}_{tab_name}", y_test, m2_pred, 0.0)
                row_m2["Repeat"] = repeat
                results["m2"].append(row_m2)
                print_result_dict(
                    {"Model": f"M2_{ecg_name}_{tab_name}", "Accuracy": row_m2["Accuracy"],
                     "Macro_F1": row_m2["Macro_F1"], "Macro_ROC_AUC(ovr)": row_m2["Macro_ROC_AUC(ovr)"]},
                    keys=["Model", "Accuracy", "Macro_F1", "Macro_ROC_AUC(ovr)"]
                )

                # Final stacking
                final_train = np.hstack([lr_m1.predict_proba(m1_train), m2_val_pred])
                final_test = np.hstack([m1_pred, m2_pred])

                lr_final = LogisticRegression(max_iter=1000).fit(final_train, y_val)
                final_pred = lr_final.predict_proba(final_test)

                row_final = get_clf_eval_mc(f"Final_{ecg_name}_{tab_name}", y_test, final_pred, 0.0)
                row_final["Repeat"] = repeat
                results["final"].append(row_final)
                print_result_dict(
                    {"Model": f"Final_{ecg_name}_{tab_name}", "Accuracy": row_final["Accuracy"],
                     "Macro_F1": row_final["Macro_F1"], "Macro_ROC_AUC(ovr)": row_final["Macro_ROC_AUC(ovr)"]},
                    keys=["Model", "Accuracy", "Macro_F1", "Macro_ROC_AUC(ovr)"]
                )
                del ecg_model, tab_model, fusion_model
                torch.cuda.empty_cache()

    # =========================================
    # Save Results
    # =========================================
    pd.DataFrame(results["single"]).to_csv(os.path.join(OUT_DIR, "1_single_modality.csv"), index=False)
    pd.DataFrame(results["m1"]).to_csv(os.path.join(OUT_DIR, "2_m1_ensemble.csv"), index=False)
    pd.DataFrame(results["m2"]).to_csv(os.path.join(OUT_DIR, "3_m2_fusion.csv"), index=False)
    pd.DataFrame(results["final"]).to_csv(os.path.join(OUT_DIR, "4_final_stacked.csv"), index=False)

    print("All tasks completed.")
