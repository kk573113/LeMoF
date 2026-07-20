# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import device


def train_model_early_stop(model, train_loader, val_loader, epochs=50, is_fusion=False, patience=7, class_weights=None):
    model = model.to(device)

    if class_weights is not None:
        cw = torch.FloatTensor(class_weights).to(device)
        crit = nn.CrossEntropyLoss(weight=cw)
    else:
        crit = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_loss = float('inf')
    best_state = None
    counter = 0

    for ep in range(epochs):
        model.train()
        for x_e, x_t, y in train_loader:
            x_e, x_t, y = x_e.to(device), x_t.to(device), y.to(device)
            opt.zero_grad()

            if is_fusion:
                logits = model(x_e, x_t)
                loss = crit(logits, y)
            else:
                if hasattr(model, 'start_conv') or hasattr(model, 'first_layer') or hasattr(model, 'lstm1') or hasattr(model, 'c1'):
                    outs = model(x_e)
                else:
                    outs = model(x_t)
                loss = sum([crit(z, y) for z in outs[:3]])

            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_e, x_t, y in val_loader:
                x_e, x_t, y = x_e.to(device), x_t.to(device), y.to(device)
                if is_fusion:
                    logits = model(x_e, x_t)
                    loss = crit(logits, y)
                else:
                    if hasattr(model, 'start_conv') or hasattr(model, 'first_layer') or hasattr(model, 'lstm1') or hasattr(model, 'c1'):
                        outs = model(x_e)
                    else:
                        outs = model(x_t)
                    loss = sum([crit(z, y) for z in outs[:3]])
                val_loss += loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_heads_proba(model, loader, is_ecg_model):
    model.eval()
    probs = []
    for x_e, x_t, _ in loader:
        x_e, x_t = x_e.to(device), x_t.to(device)
        outs = model(x_e) if is_ecg_model else model(x_t)
        p_heads = [F.softmax(z, dim=1).cpu().numpy() for z in outs[:3]]
        probs.append(np.stack(p_heads, axis=1))  # (B, 3, C)
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def predict_fusion_proba(model, loader):
    model.eval()
    probs = []
    for x_e, x_t, _ in loader:
        x_e, x_t = x_e.to(device), x_t.to(device)
        logits = model(x_e, x_t)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)
