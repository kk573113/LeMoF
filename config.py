# -*- coding: utf-8 -*-
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================
# Configuration
# =========================================
MAX_LEN = 5000
SEED_BASE = 42
N_REPEATS = 5
ENCODER_EPOCHS = 50
FUSION_EPOCHS = 80
BATCH_SIZE = 64
N_CLASSES = 2

ECG_BASE = r"/home/c/Public/bio/dataset/mimic-iv-ecg-diagnostic"
CSV_PATH = r"/home/c/Public/bio/dataset/csv/integrated_dataset_with_ecg.csv"

OUT_DIR = r"/home/cbnu/Public/bio/result/los/[260410]integrated"
os.makedirs(OUT_DIR, exist_ok=True)

CAT_COLS = ['gender', 'ventilation', 'vasopressor', 'anticoag', 'beta1']

ECG_MODEL_NAMES = ['Baseline', 'WaveNet', 'LSTM', 'ResNet']
TAB_MODEL_NAMES = ['Baseline', 'TPC', 'TabNet', 'TabTransformer', 'FTTransformer']


class TPCConfig:
    def __init__(self, n_layers=9, temp_kernels=[2] * 9, point_sizes=[64] * 9):
        self.task = 'classification'
        self.n_layers = n_layers
        self.model_type = 'tpc'
        self.diagnosis_size = 64
        self.main_dropout_rate = 0.3
        self.temp_dropout_rate = 0.1
        self.kernel_size = 3
        self.temp_kernels = temp_kernels
        self.point_sizes = point_sizes
        self.batchnorm = 'batchnorm'
        self.no_diag = True
        self.no_mask = True
        self.no_skip_connections = False
        self.no_exp = True
