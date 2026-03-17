import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from dataset import SWMMDataset

from lstm import SimpleLSTM
from gru import SimpleGRU
from attention import AttentionLSTM, CausalAttentionLSTM
from model import Trainer
import matplotlib.pyplot as plt

from swmm.simulator import SWMMSimulator
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

if __name__ == "__main__":
    try:
        trainer = Trainer(
            model_type='SimpleLSTM', 
            model_params={
                'input_size': 1, 
                'num_layers': 2,
                'output_size': 1
                }, 
            model_path='simple_lstm_model.pth'
        )
        swmm_simulator = SWMMSimulator(template_inp_path='template.inp', output_element='SN_001', output_type='node', output_variable='depth')
        model, dataset = trainer.train(swmm_simulator=swmm_simulator)
        print("\n=== 程序运行成功 ===")
    except Exception as e:
        print(f"\n=== 程序运行出错: {e} ===")