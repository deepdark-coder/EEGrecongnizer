import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import collections

def cluster_subjects_by_baseline(data_dir: str, subject_ids: list, n_clusters: int = 3, fs: int = 250, show_plot: bool = True):
    """
    根据受试者的生理基线特征进行空间聚类划分，并提供可视化接口。
    
    参数:
        data_dir: 数据集目录
        subject_ids: 参与划分的受试者 ID 列表
        n_clusters: 期望划分的簇(孤岛)数量
        fs: 采样率
        show_plot: 是否弹出 2D 聚类可视化散点图
        
    返回:
        cluster_dict: 字典结构 {簇ID: [受试者ID列表]}
    """
    print(f"\n[聚类引擎] 正在提取 {len(subject_ids)} 个受试者的生理基线特征...")
    
    baselines = []
    valid_subject_ids = []
    
    for sid in subject_ids:
        mat_file = os.path.join(data_dir, f'HC{sid}_1s.mat')
        if not os.path.exists(mat_file):
            continue
            
        mat = scipy.io.loadmat(mat_file)
        data = np.ascontiguousarray(mat['data'], dtype=np.float32)
        
        # 1. 提取频域特征 (与模型预处理保持一致)
        trials, channels, timepoints = data.shape
        fft_data = np.fft.fft(data, axis=2)
        power_spectrum = (np.abs(fft_data) ** 2) / timepoints
        freqs = np.fft.fftfreq(timepoints, 1/fs)
        
        idx = np.where((freqs >= 1) & (freqs <= 50))[0]
        psd_features = power_spectrum[:, :, idx]
        log_psd = np.log10(psd_features + 1e-8)
        
        # 2. 提取生理基线：沿着 Trials 维度 (axis=0) 求平均
        subject_baseline = np.mean(log_psd, axis=0)
        baselines.append(subject_baseline.flatten())
        valid_subject_ids.append(sid)
        
    X = np.stack(baselines)
    
    # 3. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. PCA 降维去噪
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"[聚类引擎] PCA 降维后特征维度: {X_pca.shape[1]}")
    
    # 5. K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    # ================= 新增：可视化接口 =================
    if show_plot:
        print("[聚类引擎] 正在计算 t-SNE 2D 投影用于可视化...")
        # 动态调整 perplexity，防止受试者数量较少时报错
        perplexity = min(30, len(valid_subject_ids) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        
        plt.figure(figsize=(10, 8))
        # 获取离散的颜色映射表
        cmap = plt.get_cmap('tab10')
        
        for cluster_id in range(n_clusters):
            # 找到属于当前簇的受试者索引
            idx = np.where(labels == cluster_id)[0]
            
            # 画散点
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                        c=[cmap(cluster_id)], label=f'Cluster {cluster_id}', 
                        alpha=0.7, s=200, edgecolors='white', linewidth=1.5)
            
            # 在中心打上受试者的 ID 数字
            for i in idx:
                plt.annotate(str(valid_subject_ids[i]), 
                             (X_tsne[i, 0], X_tsne[i, 1]),
                             fontsize=10, ha='center', va='center', color='black', weight='bold')

        plt.title(f'Subject Baseline Clustering (K-Means, k={n_clusters})', fontsize=16, pad=15)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(title="Assigned Clusters", fontsize=10, title_fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300)
        
        # 弹出窗口展示
        print("fig saved")
        plt.show()
    # ==================================================

    # 6. 构建返回字典
    cluster_dict = collections.defaultdict(list)
    for sid, label in zip(valid_subject_ids, labels):
        cluster_dict[label].append(sid)
        
    print("\n[划分结果摘要]")
    for cluster_id, sids in cluster_dict.items():
        print(f"  ▶ 簇 {cluster_id} 包含 {len(sids)} 个受试者: {sids}")
        
    return dict(cluster_dict)