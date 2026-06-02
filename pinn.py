import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader 
from dataset import SWMMDataset
from registry import register_model, create_model

@register_model("PINNLSTM")
class PINNLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.3):
        super().__init__()
        # LSTM部分与之前相同
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        # 可学习的物理参数，用 softplus 确保正数
        self.log_a = nn.Parameter(torch.tensor(0.0))
        self.log_b = nn.Parameter(torch.tensor(0.0))

    @property
    def a(self):
        return torch.nn.functional.softplus(self.log_a)

    @property
    def b(self):
        return torch.nn.functional.softplus(self.log_b)

    def forward(self, x):
        # LSTM前向
        lstm_out, _ = self.lstm(x)
        water_level = self.fc(lstm_out)   # (batch, seq_len, 1)
        return water_level

    def physics_residual(self, rain, water_level):
        """
        计算物理残差
        rain: (batch, seq_len, 1)
        water_level: (batch, seq_len, 1)
        """
        h_t = water_level[:, :-1, 0]      # (batch, seq_len-1)
        h_t1 = water_level[:, 1:, 0]       # (batch, seq_len-1)
        P_t = rain[:, :-1, 0]               # (batch, seq_len-1)
        # 期望的下一时刻水位
        expected = (1 - self.b) * h_t + self.a * P_t
        residual = h_t1 - expected
        return residual



class PINNTrainer:
    def __init__(self, model_type='PINNLSTM', 
                 model_params={'input_size': 1, 'output_size': 1}, 
                 model_path='pinn_lstm_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model_type = model_type
        self.model_params = model_params
        self.model_path = model_path

    def train(self, n_events=100, seq_length=288, time_step_min=5, swmm_simulator=None, epochs=200, lr=0.001):
        """主程序：训练模型"""
        print("=== 水位预测模型训练 ===")
        print(f"序列长度: {seq_length} 个时间步 ({seq_length * time_step_min/60:.1f}小时)")
        print(f"时间分辨率: {time_step_min}分钟")
        
        # 创建数据集
        print("\n创建数据集...")
        dataset = SWMMDataset(
            n_events=n_events,
            seq_length=seq_length,
            time_step_min=time_step_min,
            swmm_simulator=swmm_simulator
        )
        
        # 划分数据集
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_dataset)} 个样本")
        print(f"  验证集: {len(val_dataset)} 个样本")
        print(f"  测试集: {len(test_dataset)} 个样本")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 初始化模型
        print(f"\n初始化{self.model_type}模型...")

        # model = SimpleLSTM(
        #     input_size=input_size,
        #     output_size=output_size,
        #     dropout=0.3
        # )
        model = create_model(self.model_type, **self.model_params)
        
        # 训练模型
        """训练序列到序列模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        train_losses, val_losses = [], []
        lambda_physics = 0.1   # 可调整
        print(f"开始训练{self.model_type}模型...")
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)  # 输出形状: (batch_size, seq_length, 1)
                loss_data = criterion(output, target)  # target形状: (batch_size, seq_length, 1)
                
                # 物理损失（用预测水位计算残差）
                residual = model.physics_residual(data, output)
                loss_physics = torch.mean(residual ** 2)
                # 总损失
                loss = loss_data + lambda_physics * loss_physics

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss_data = criterion(output, target) 
                    # 物理损失（用预测水位计算残差）
                    residual = model.physics_residual(data, output)
                    loss_physics = torch.mean(residual ** 2)
                     # 总损失
                    loss = loss_data + lambda_physics * loss_physics
                    val_loss += loss.item()
            
            # 计算平均损失
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
        # 绘制训练过程
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('模型训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 测试模型
        print(f"\n测试{self.model_type}模型...")
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 在测试集上评估
        test_loss = 0
        criterion = nn.MSELoss()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss_data = criterion(output, target) 
                # 物理损失（用预测水位计算残差）
                residual = model.physics_residual(data, output)
                loss_physics = torch.mean(residual ** 2)
                  # 总损失
                loss = loss_data + lambda_physics * loss_physics
                test_loss += loss.item()
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        test_loss /= len(test_loader)
        print(f'测试损失: {test_loss:.6f}')
        
        # 保存模型
        print(f"\n保存{self.model_type}模型...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'rain_scaler': dataset.rain_scaler,
            'water_scaler': dataset.water_scaler,
            'seq_length': dataset.seq_length,
            'n_events': dataset.n_events,
            'time_step_min': dataset.time_step_min,
            'model_type': self.model_type,
            'model_params': self.model_params
        }, self.model_path)
        
        print("模型训练完成！")
        
        return model, dataset

from swmm.simulator import SWMMSimulator
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

if __name__ == "__main__":
    try:
        trainer = PINNTrainer(
            model_type='PINNLSTM', 
            model_params={
                'input_size': 1, 
                'num_layers': 2,
                'output_size': 1
                }, 
            model_path='pinn_lstm_model.pth'
        )
        swmm_simulator = SWMMSimulator(template_inp_path='template.inp', output_element='SN_001', output_type='node', output_variable='depth')
        model, dataset = trainer.train(swmm_simulator=swmm_simulator)
        print("\n=== 程序运行成功 ===")
    except Exception as e:
        print(f"\n=== 程序运行出错: {e} ===")