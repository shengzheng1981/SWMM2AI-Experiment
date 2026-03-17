import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from dataset import SWMMDataset
from swmm.rainfall.generator import RainfallGenerator
from swmm.simulator import SWMMSimulator

class ChebConv(nn.Module):
    """
    Chebyshev Graph Convolution (NO EINSUM)
    - Explicit matrix multiplication (stable & readable)
    - Adapted for directed drainage pipe networks with edge features
    """
    def __init__(self, in_channels, out_channels, K, edge_dim=0):
        super().__init__()
        self.K = K  # Chebyshev polynomial order (recommend K=2 for pipe networks)
        self.edge_dim = edge_dim
        
        # Chebyshev weights: K layers × in_channels × out_channels
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_channels, out_channels)) 
            for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Edge feature projection (pipe diameter/slope → adjacency weight)
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)
        if self.edge_dim > 0:
            nn.init.xavier_uniform_(self.edge_proj.weight)
            nn.init.zeros_(self.edge_proj.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: Node features → (B, N, in_channels) or (N, in_channels)
        edge_index: Directed edges → (2, E) (source → target)
        edge_attr: Edge features (pipe diameter/slope) → (E, edge_dim)
        """
        # Step 1: Handle batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, in_channels)
        B, N, C_in = x.shape
        device = x.device

        # Step 2: Build weighted adjacency matrix (N, N) for pipe network
        adj = torch.zeros(N, N, device=device)
        src, dst = edge_index[0], edge_index[1]
        
        # Add edge feature weights (pipe properties)
        if edge_attr is not None and self.edge_dim > 0:
            edge_weights = self.edge_proj(edge_attr).squeeze(-1)  # (E,)
            adj[dst, src] = edge_weights  # Directed: dst ← src (upstream → downstream)
        else:
            adj[dst, src] = 1.0  # Unweighted adjacency

        # Step 3: Build graph Laplacian (normalized)
        degree = torch.sum(adj, dim=1)  # Degree matrix (N,)
        degree_sqrt_inv = torch.where(degree > 0, 1.0 / torch.sqrt(degree), torch.zeros_like(degree))
        laplacian = torch.eye(N, device=device) - (
            degree_sqrt_inv.unsqueeze(1) * adj * degree_sqrt_inv.unsqueeze(0)
        )

        # Step 4: Chebyshev polynomial expansion (NO EINSUM)
        # T0 = x, T1 = L·x, Tk = 2L·Tk-1 - Tk-2
        cheb_feat = [x]  # T0
        if self.K > 1:
            T1 = torch.bmm(laplacian.unsqueeze(0).expand(B, -1, -1), x)  # (B, N, C_in)
            cheb_feat.append(T1)
        
        for k in range(2, self.K):
            Tk = 2 * torch.bmm(laplacian.unsqueeze(0).expand(B, -1, -1), cheb_feat[-1]) - cheb_feat[-2]
            cheb_feat.append(Tk)

        # Step 5: Apply Chebyshev weights (matrix multiplication)
        out = torch.zeros(B, N, self.weights[0].shape[1], device=device)
        for k in range(self.K):
            # (B, N, C_in) × (C_in, C_out) → (B, N, C_out)
            out += torch.matmul(cheb_feat[k], self.weights[k])

        # Step 6: Add bias & activation
        out = out + self.bias.unsqueeze(0).unsqueeze(0)  # Broadcast bias
        out = F.relu(out)

        # Remove batch dim if input had no batch
        if B == 1 and x.dim() == 2:
            out = out.squeeze(0)
        
        return out

# Main ChebNet-LSTM Model (Drainage Pipe Network Adapted)
class ChebNetLSTM(nn.Module):
    def __init__(self, num_nodes, node_static_dim=0, edge_dim=0, K=2,
                 hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_static_dim = node_static_dim
        self.edge_dim = edge_dim
        
        # Input dimension: 1 (rainfall) + static node features (elevation/area)
        self.node_input_dim = 1 + node_static_dim

        # Chebyshev GCN (spatial feature extraction for pipe network)
        self.chebnet = ChebConv(
            in_channels=self.node_input_dim,
            out_channels=hidden_size,
            K=K,
            edge_dim=edge_dim
        )

        # LSTM (temporal feature extraction for rainfall/water level)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output head (predict water level/depth)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, rain, edge_index, edge_attr, node_static_feat=None):
        """
        Args:
            rain: (batch, seq_len, 1) → Global rainfall time series
            edge_index: (2, num_edges) → Directed pipe network topology
            edge_attr: (num_edges, edge_dim) → Pipe features (diameter/slope/roughness)
            node_static_feat: (num_nodes, node_static_dim) → Node features (elevation/catchment area)
        Returns:
            output: (batch, num_nodes, seq_len, output_size) → Predicted water level
        """
        batch_size, seq_len, _ = rain.shape
        device = rain.device

        # Step 1: Broadcast rainfall to all nodes (rainfall drives pipe network)
        rain_expanded = rain.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)  # (B, N, T, 1)

        # Step 2: Concatenate static node features (elevation/catchment area)
        if node_static_feat is not None:
            static_expanded = node_static_feat.unsqueeze(0).unsqueeze(2).expand(
                batch_size, -1, seq_len, -1
            )  # (B, N, T, static_dim)
            node_feat = torch.cat([rain_expanded, static_expanded], dim=-1)  # (B, N, T, 1+static_dim)
        else:
            node_feat = rain_expanded  # Only rainfall

        # Step 3: Spatial ChebNet (process all time steps in parallel)
        # Reshape: (B, N, T, C) → (B×T, N, C)
        node_feat_flat = node_feat.permute(0, 2, 1, 3).reshape(batch_size * seq_len, self.num_nodes, -1)
        cheb_out = self.chebnet(node_feat_flat, edge_index, edge_attr)  # (B×T, N, hidden_size)

        # Reshape back: (B×T, N, H) → (B, N, T, H)
        cheb_out = cheb_out.reshape(batch_size, seq_len, self.num_nodes, -1).permute(0, 2, 1, 3)

        # Step 4: Temporal LSTM (shared across all nodes)
        # Reshape: (B, N, T, H) → (B×N, T, H)
        lstm_in = cheb_out.reshape(batch_size * self.num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)  # (B×N, T, hidden_size)

        # Step 5: Predict water level/depth
        fc_out = self.fc(lstm_out)  # (B×N, T, output_size)

        # Reshape to final output: (B, N, T, output_size)
        output = fc_out.reshape(batch_size, self.num_nodes, seq_len, -1)
        
        return output
 
# ---------- 生成具有物理意义的模拟数据 ----------
def generate_realistic_data(num_nodes, num_edges, seq_len, num_samples,
                            node_static_dim, edge_dim, output_size=1,
                            device='cpu'):
    """
    生成模拟数据，使水位与降雨相关，并保证水位为正。
    生成规则：
      - 每个节点的水位 = 过去3个时刻降雨的加权和（权重递减） + 邻居影响的线性组合 + 噪声
      - 邻居影响：节点水位受其上游节点当前时刻水位的一定比例影响（模拟水流传播）
    为了简化，我们先生成降雨序列，然后通过一个简单的传递函数计算水位。
    注意：这里为了演示，我们假设节点之间是独立计算的（不考虑上游对下游的延迟），
          但可以通过边特征和GNN学习这种关系。
    """
    # 固定边索引（随机，但避免自环）
    edge_index = []
    while len(edge_index) < num_edges * 2:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v:
            edge_index.extend([u, v])
    edge_index = torch.tensor(edge_index).reshape(2, num_edges).long().to(device)

    # 边特征：随机正态，模拟管径、坡度等
    edge_attr = torch.randn(num_edges, edge_dim, device=device)

    # 节点静态特征：随机正态
    node_static = torch.randn(num_nodes, node_static_dim, device=device)

    # 生成降雨序列 (num_samples, seq_len, 1)  [正值，对数正态分布]
    rain = torch.exp(0.5 * torch.randn(num_samples, seq_len, 1, device=device))

    # 生成目标水位
    # 简单物理模型：每个时刻的水位 = 过去3个时刻降雨的加权和 + 0.1 * 上游节点水位（通过图传播） + 静态偏置
    # 为了简化，我们使用一个循环生成，并加入噪声
    target = torch.zeros(num_samples, num_nodes, seq_len, output_size, device=device)

    # 为每个节点生成一个基础偏置（来自静态特征）
    base_bias = node_static @ torch.randn(node_static_dim, 1, device=device)  # (N,1)

    # 上游影响矩阵（根据边索引）
    upstream_mask = torch.zeros(num_nodes, num_nodes, device=device)
    for i in range(num_edges):
        u, v = edge_index[0, i], edge_index[1, i]
        upstream_mask[v, u] = 1.0  # v 受 u 影响

    # 时间步循环（从第3步开始）
    for b in range(num_samples):
        for t in range(seq_len):
            # 降雨贡献：过去3个时刻（如果t>=2）
            rain_contrib = 0.0
            for k in range(max(0, t-2), t+1):
                rain_contrib += 0.5 ** (t - k) * rain[b, k, 0]  # 权重递减
            # 加入基础偏置
            node_level = base_bias.squeeze() + rain_contrib  # (N,)
            # 加入上游节点影响（假设当前时刻上游水位已知，这里简单用前一时刻的上游水位？为了简化，我们使用同时间步的上游水位，但会造成循环依赖，所以这里我们使用前一时刻的上游水位）
            if t > 0:
                upstream_influence = upstream_mask @ target[b, :, t-1, 0]  # (N,)
                node_level = node_level + 0.1 * upstream_influence
            # 加入噪声，并确保为正（通过softplus或绝对值，这里使用指数变换保证正）
            noise = 0.05 * torch.randn(num_nodes, device=device)
            target[b, :, t, 0] = torch.exp(0.1 * (node_level + noise))  # 指数保证正

    return rain, node_static, edge_index, edge_attr, target



def read_swmm_inp(filepath):
    """
    从 SWMM .inp 文件中提取管网拓扑和静态属性。

    参数:
        filepath: .inp 文件路径

    返回:
        dict: 包含以下键值:
            - node_names: 列表，节点ID字符串
            - node_invert: numpy数组，每个节点的井底高程 (m)
            - node_depth: numpy数组，每个节点的井深 (m)
            - edge_from: 列表，每条边的上游节点索引 (0-based)
            - edge_to: 列表，每条边的下游节点索引 (0-based)
            - edge_length: numpy数组，每条边的长度 (m)
            - edge_roughness: numpy数组，每条边的曼宁系数
            - edge_diameter: numpy数组，每条边的管径 (m)
            - num_nodes: 节点数量
            - num_edges: 管道数量
    """
    # 初始化存储
    junctions = OrderedDict()      # 节点名称 -> (invert, depth)
    conduits = []                  # 列表，每个元素为 (from, to, length, roughness)
    xsections = {}                 # 管道名称 -> 直径 (圆形管道)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 状态变量
    current_section = None
    in_junctions = False
    in_conduits = False
    in_xsections = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):  # 空行或注释
            continue

        # 检测节开始
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].upper()
            in_junctions = (current_section == 'JUNCTIONS')
            in_conduits = (current_section == 'CONDUITS')
            in_xsections = (current_section == 'XSECTIONS')
            continue

        # 根据当前节解析数据
        if in_junctions:
            # 格式: Name        Elevation   MaxDepth   InitDepth  SurchargeDepth  PondedArea
            # 有些列可能缺失，但前三个通常存在。我们取 Elevation 作为井底高程，MaxDepth 作为井深。
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    invert = float(parts[1])      # 井底高程
                    depth = float(parts[2])       # 井深
                    junctions[name] = (invert, depth)
                except ValueError:
                    continue

        elif in_conduits:
            # 格式: Name   FromNode   ToNode   Length   Roughness   ... (后面还有 InOffset, OutOffset 等，但我们只需前5个)
            parts = line.split()
            if len(parts) >= 5:
                name = parts[0]
                from_node = parts[1]
                to_node = parts[2]
                try:
                    length = float(parts[3])
                    roughness = float(parts[4])
                    conduits.append((name, from_node, to_node, length, roughness))
                except ValueError:
                    continue

        elif in_xsections:
            # 格式: Link        Shape      Geom1     Geom2     Geom3     Geom4
            # 对于圆形管道，Shape='CIRCULAR', Geom1 为直径
            parts = line.split()
            if len(parts) >= 3 and parts[1].upper() == 'CIRCULAR':
                link_name = parts[0]
                try:
                    diameter = float(parts[2])     # 直径
                    xsections[link_name] = diameter
                except ValueError:
                    continue

    # 构建节点索引映射
    node_names = list(junctions.keys())
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    num_nodes = len(node_names)

    # 提取节点属性
    node_invert = np.zeros(num_nodes, dtype=np.float32)
    node_depth = np.zeros(num_nodes, dtype=np.float32)
    for i, name in enumerate(node_names):
        invert, depth = junctions[name]
        node_invert[i] = invert
        node_depth[i] = depth

    # 提取边信息
    edge_from = []
    edge_to = []
    edge_length = []
    edge_roughness = []
    edge_diameter = []

    for conduit in conduits:
        name, from_node, to_node, length, roughness = conduit
        # 确保节点存在
        if from_node not in node_to_idx or to_node not in node_to_idx:
            continue
        # 确保有管径信息
        if name not in xsections:
            continue
        edge_from.append(node_to_idx[from_node])
        edge_to.append(node_to_idx[to_node])
        edge_length.append(length)
        edge_roughness.append(roughness)
        edge_diameter.append(xsections[name])

    num_edges = len(edge_from)

    # 转换为numpy数组
    edge_length = np.array(edge_length, dtype=np.float32)
    edge_roughness = np.array(edge_roughness, dtype=np.float32)
    edge_diameter = np.array(edge_diameter, dtype=np.float32)

    # 返回结果字典
    return {
        'node_names': node_names,
        'node_invert': node_invert,
        'node_depth': node_depth,
        'edge_from': edge_from,
        'edge_to': edge_to,
        'edge_length': edge_length,
        'edge_roughness': edge_roughness,
        'edge_diameter': edge_diameter,
        'num_nodes': num_nodes,
        'num_edges': num_edges
    }

# ---------- 训练设置 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
num_nodes = 10
num_edges = 25
seq_length = 288
num_samples = 100          # 总样本数
batch_size = 16
node_static_dim = 2
edge_dim = 3
output_size = 1
hidden_size = 64
num_layers = 2
dropout = 0.2
learning_rate = 0.001
epochs = 100
patience = 10              # 早停耐心值
target_node_name = 'SN_001'

data = read_swmm_inp('template.inp')
target_node_idx = data['node_names'].index(target_node_name)
num_nodes = data['num_nodes']
num_edges = data['num_edges']
node_static = torch.tensor(np.column_stack([data['node_invert'], data['node_depth']]), dtype=torch.float32)
edge_index = torch.tensor([data['edge_from'], data['edge_to']], dtype=torch.long)
edge_attr = torch.tensor(np.column_stack([data['edge_diameter'], data['edge_length'], data['edge_roughness']]), dtype=torch.float32)


def train():
    # 生成固定数据集
    print("生成模拟数据...")
    simulator = SWMMSimulator(template_inp_path='template.inp', output_element=target_node_name, output_type='node', output_variable='depth')
    dataset = SWMMDataset(
                n_events=num_samples,
                seq_length=seq_length,
                time_step_min=5,
                swmm_simulator=simulator
            )
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # 实例化模型
    model = ChebNetLSTM(
        num_nodes=num_nodes,
        node_static_dim=node_static_dim,
        edge_dim=edge_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # ---------- 训练循环（带早停） ----------
    print("开始训练...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_rain, batch_target in train_loader:
            batch_rain = batch_rain.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_rain, edge_index, edge_attr, node_static)
            pred_target = output[:, target_node_idx, :, :]    
            loss = criterion(pred_target, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_rain.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_rain, batch_target in val_loader:
                batch_rain = batch_rain.to(device)
                batch_target = batch_target.to(device)
                output = model(batch_rain, edge_index, edge_attr, node_static)
                pred_target = output[:, target_node_idx, :, :]  
                loss = criterion(pred_target, batch_target)
                val_loss += loss.item() * batch_rain.size(0)
        val_loss /= len(val_loader.dataset)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), 'chebnet_model.pth')
    # 加载最佳模型
    model.load_state_dict(torch.load('chebnet_model.pth'))

    # ---------- 测试 ----------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_rain, batch_target in test_loader:
            batch_rain = batch_rain.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_rain, edge_index, edge_attr, node_static)
            pred_target = output[:, target_node_idx, :, :]  
            loss = criterion(pred_target, batch_target)
            test_loss += loss.item() * batch_rain.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"\n测试集损失: {test_loss:.6f}")

def predict():
    # 1. 实例化模型并加载
    model = ChebNetLSTM(
        num_nodes=num_nodes,
        node_static_dim=node_static_dim,
        edge_dim=edge_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load('chebnet_model.pth'))
    model.eval()
    # 2. 生成测试降雨序列
    rg = RainfallGenerator(time_step_min=5)
    
    test_rainfall = rg.generate_random_rainfall_event(
        seq_length=288,  # 必须与训练时相同的长度
        min_duration=2,
        max_duration=4,
        rain_type='chicago'
    )

    # 3. 创建SWMM模拟
    simulator = SWMMSimulator(template_inp_path='template.inp', output_element=target_node_name, output_type='node', output_variable='depth')
    swmm_results = simulator.run_swmm_simulation(
                    rainfall_mm_h=test_rainfall
                )
    swmm_water_sequence = swmm_results['values']
    # 4. 进行预测 
    # 或者使用 reshape
    rain_tensor = torch.tensor(test_rainfall, dtype=torch.float32).view(1, -1, 1)
    rain_tensor = rain_tensor.to(device)
    with torch.no_grad():
        output = model(rain_tensor, edge_index, edge_attr, node_static)  # (1, N, T, 1)

    # 5. 提取目标节点 SN_001 的预测 
    pred_target = output[0, target_node_idx, :, 0].cpu().numpy()  # 形状 (T,)

    # 可选：绘制一个节点的预测曲线
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,4))
    plt.plot(swmm_water_sequence, label='True')
    plt.plot(pred_target, label='Predicted')
    plt.xlabel('Time step')
    plt.ylabel('Water level')
    plt.title(f'Node {target_node_name} prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    predict()

