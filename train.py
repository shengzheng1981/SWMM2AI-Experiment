import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from dataset import SWMMDataset
from mlp import SimpleMLP
from lstm import SimpleLSTM, MultiScaleLSTM
from gru import SimpleGRU
from attention import AttentionLSTM, CausalAttentionLSTM
from residual import ResidualLSTM
from seq2seq import Seq2SeqLSTM
# from transformer import Transformer
from model import Trainer
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

if __name__ == "__main__":
    try:
        trainer = Trainer(
            model_type='SimpleMLP', 
            model_params={
                'input_size': 1, 
                'num_layers': 2,
                'output_size': 1
                }, 
            model_path='simple_mlp_model.pth'
        )
        model, dataset = trainer.train()
        print("\n=== 程序运行成功 ===")
    except Exception as e:
        print(f"\n=== 程序运行出错: {e} ===")