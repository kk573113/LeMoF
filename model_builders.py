# -*- coding: utf-8 -*-
from config import N_CLASSES, TPCConfig

from models_ecg.ecg_baseline import ECG_Baseline
from models_ecg.ecg_resnet import ResNet1d
from models_ecg.wavenet import WaveNet
from models_ecg.lstm import ECG_LSTM

from models_ehr.ehr_baseline import Tabular_Baseline
from models_ehr.tpc import TempPointConv
from models_ehr.tabnet import TabNet
from models_ehr.tabtransformer import TabTransformer
from models_ehr.fttransformer import FTTransformer


def build_ecg_model(ecg_name):
    if ecg_name == 'Baseline':
        return ECG_Baseline(num_classes=N_CLASSES)
    elif ecg_name == 'ResNet':
        return ResNet1d(input_channels=12, num_classes=N_CLASSES)
    elif ecg_name == 'WaveNet':
        return WaveNet(input_channels=12, num_classes=N_CLASSES)
    elif ecg_name == 'LSTM':
        return ECG_LSTM(input_channels=12, num_classes=N_CLASSES)
    else:
        raise ValueError(ecg_name)


def build_tab_model(tab_name, tab_dim, num_cnt, cat_dims):
    if tab_name == 'Baseline':
        return Tabular_Baseline(in_dim=tab_dim, num_classes=N_CLASSES)
    elif tab_name == 'TPC':
        tpc_layers = 9
        tpc_conf = TPCConfig(n_layers=tpc_layers, temp_kernels=[2] * tpc_layers, point_sizes=[64] * tpc_layers)
        return TempPointConv(config=tpc_conf, F=tab_dim, D=0, no_flat_features=0, num_classes=N_CLASSES)
    if tab_name == 'TabNet':
        return TabNet(
            input_dim=tab_dim, num_classes=N_CLASSES,
            cat_idxs=list(range(num_cnt, tab_dim)),
            cat_dims=list(cat_dims), cat_emb_dim=2
        )
    elif tab_name == 'TabTransformer':
        return TabTransformer(
            categories=cat_dims, num_continuous=num_cnt,
            dim=32, depth=6, heads=8, num_classes=N_CLASSES
        )
    elif tab_name == 'FTTransformer':
        return FTTransformer(
            categories=cat_dims, num_continuous=num_cnt,
            dim=32, depth=3, heads=8, num_classes=N_CLASSES
        )
    else:
        raise ValueError(tab_name)
