import numpy as np
import scipy.io as sio
import torch

def load_and_denoise_data(mat_file_path):
    """
    读取 .mat 格式的脑电数据，并执行极简且高效的物理去噪。
    输入数据形状预期为: (Trials, 30, 250)
    """
    # ================= 1. 读取原始数据 =================
    mat = sio.loadmat(mat_file_path)
    all_data = np.ascontiguousarray(mat['data'], dtype=np.float32)
    all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

    # 此时 all_data 的形状为 (N, 30, 250)
    
    # ================= 2. 核心去噪模块 =================
    
    # 【步骤 A：幅值截断 (剔除极端的眼电/肌电突刺)】
    # 计算当前被试全局数据的标准差
    std_val = np.std(all_data)
    # 采用 3-Sigma 原则，认为超过 3 倍标准差的信号是肌肉或设备伪影
    threshold = 3 * std_val
    # np.clip 会把超过阈值的波峰“削平”，保留正常的脑电波动
    all_data = np.clip(all_data, -threshold, threshold)


    # 【步骤 B：CAR 空间滤波 (共模平均参考，去除全局环境噪声)】
    # 沿着通道维度 (axis=1) 求平均，计算出此刻覆盖在整个头皮上的“共模噪声”
    # keepdims=True 保证形状变为 (N, 1, 250)，以便利用广播机制相减
    common_mode_noise = np.mean(all_data, axis=1, keepdims=True)
    # 让每一个通道都减去这个共模噪声，凸显出各个脑区局部的真实放电差异
    all_data = all_data - common_mode_noise


    mu = all_data.mean()
    std = all_data.std() + 1e-8
    all_data = (all_data - mu) / std
    

    # (N, 30, 250) -> (N, 1, 30, 250) 适配你的 PatchEmbedding 的 Conv2d
    all_data = np.ascontiguousarray(all_data[:, np.newaxis, :, :], dtype=np.float32)

    return torch.tensor(all_data, dtype=torch.float32), torch.tensor(all_label, dtype=torch.long)