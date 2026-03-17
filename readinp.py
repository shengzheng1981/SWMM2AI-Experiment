import re
import numpy as np
from collections import OrderedDict

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

# 使用示例
if __name__ == '__main__':
    data = read_swmm_inp('template.inp')
    print("节点数:", data['num_nodes'])
    print("管道数:", data['num_edges'])
    print("前5个节点井底高程:", data['node_invert'][:5])
    print("前5条管道管径:", data['edge_diameter'][:5])