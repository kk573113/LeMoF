# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from config import N_CLASSES


class CoAttention(nn.Module):
    def __init__(self, ecg_encoder, tab_encoder, ecg_idx, tab_idx, ecg_dim=64, tab_dim=32, d_model=64, num_heads=4):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.tab_encoder = tab_encoder
        self.ecg_idx = ecg_idx
        self.tab_idx = tab_idx

        # Projection (64 -> d_model)
        self.proj_ecg = nn.Linear(ecg_dim, d_model)
        self.proj_tab = nn.Linear(tab_dim, d_model)  # HeadBlock output is 64 dim

        self.attn_tab_to_ecg = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.attn_ecg_to_tab = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_CLASSES)
        )

    def forward(self, x_ecg, x_tab):
        # Encoders return list: [z1, z2, z3, f1, f2, f3, ...]
        e_outs = self.ecg_encoder(x_ecg)
        t_outs = self.tab_encoder(x_tab)

        # Feature indices in the return list (0,1,2 are logits, 3,4 are head1 feats, etc.)
        # HeadBlock returns: [feat_seq, feat_pool, logits] -> but Encoder wrapper returns flattened list
        # Encoder returns: [z1, z2, z3, (f_seq1, f_pool1), (f_seq2, f_pool2), ...]
        # Let's adjust index based on how we implemented models.
        # Impl: return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]

        # indices for f_seq:
        # Head 1 (idx 0): index 3
        # Head 2 (idx 1): index 5
        # Head 3 (idx 2): index 7

        real_ecg_seq_idx = 3 + (self.ecg_idx * 2)
        real_tab_seq_idx = 3 + (self.tab_idx * 2)

        # ECG Feat: (Batch, Dim, Length) -> Permute to (Batch, Length, Dim) for Attention
        ecg_feat = e_outs[real_ecg_seq_idx].permute(0, 2, 1)

        # Tab Feat
        tab_feat = t_outs[real_tab_seq_idx]
        if tab_feat.dim() == 2:
            tab_feat = tab_feat.unsqueeze(1)  # (Batch, 1, Dim)
        elif tab_feat.dim() == 3:
            tab_feat = tab_feat.permute(0, 2, 1)  # (Batch, Length, Dim)

        H_ecg = self.proj_ecg(ecg_feat)
        H_tab = self.proj_tab(tab_feat)

        # Cross Attention
        out_tab_guided, _ = self.attn_tab_to_ecg(query=H_tab, key=H_ecg, value=H_ecg)
        out_ecg_guided, _ = self.attn_ecg_to_tab(query=H_ecg, key=H_tab, value=H_tab)

        # Global Average Pooling
        ctx_tab = out_tab_guided.mean(dim=1)
        ctx_ecg = out_ecg_guided.mean(dim=1)

        combined = torch.cat([ctx_tab, ctx_ecg], dim=-1)
        return self.fc_out(combined)
