import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================= 配置区 =================
# 替换为你实际的数据目录
DATA_DIR = "./EEG-Conformer/data/processed_normal/"
# 为了可视化清晰，抽取前 5 个被试的数据
SUBJECT_LIMIT = 5 
# ==========================================

def extract_log_psd_for_clustering(data, fs=250):
    """提取 1-50Hz 的 Log-PSD 特征"""
    trials, channels, timepoints = data.shape
    fft_data = np.fft.fft(data, axis=2)
    power_spectrum = (np.abs(fft_data) ** 2) / timepoints
    freqs = np.fft.fftfreq(timepoints, 1/fs)
    
    idx = np.where((freqs >= 1) & (freqs <= 50))[0]
    psd_features = power_spectrum[:, :, idx]
    return np.log10(psd_features + 1e-8)

def load_multi_subject_data(data_dir, limit):
    all_features = []
    all_labels = []
    all_subjects = []  # 记录样本属于哪个被试
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, 'HC*_1s.mat')))[:limit]
    
    if not mat_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到数据文件。")
        
    print(f"正在加载 {len(mat_files)} 个被试的数据进行特征诊断...")
    
    for sid, mat_file in enumerate(mat_files):
        mat = sio.loadmat(mat_file)
        data = np.ascontiguousarray(mat['data'], dtype=np.float32)
        label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)
        
        # 提取频域特征
        features = extract_log_psd_for_clustering(data)
        
        # 将 (Trials, 30, 50) 展平为 (Trials, 1500) 用于经典机器学习算法
        features = features.reshape(features.shape[0], -1)
        
        all_features.append(features)
        all_labels.append(label)
        all_subjects.append(np.full(label.shape[0], sid))
        
    return np.concatenate(all_features), np.concatenate(all_labels), np.concatenate(all_subjects)

def main():
    # 1. 加载数据
    X, y_true, y_subject = load_multi_subject_data(DATA_DIR, SUBJECT_LIMIT)
    print(f"提取的总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

    # 2. 数据标准化 (极其重要，消除绝对功率的个体差异)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA 降维去噪 (保留 95% 的方差，通常能将 1500 维降到 100 维左右)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA 降维后特征维度: {X_pca.shape[1]}")

    # 4. t-SNE 降维到 2 维用于可视化
    print("正在执行 t-SNE 流形映射 (可能需要一分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    # 5. K-Means 无监督聚类 (强制分为两类)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_pca) # 在 PCA 空间聚类更准

    # 6. 绘图对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图 1：按真实情绪标签着色
    scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='coolwarm', alpha=0.6, s=15)
    axes[0].set_title('Ground Truth (Emotion Labels)')
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Emotion")
    axes[0].add_artist(legend1)

    # 子图 2：按 K-Means 聚类结果着色
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, s=15)
    axes[1].set_title('K-Means Unsupervised Clustering')
    legend2 = axes[1].legend(*scatter2.legend_elements(), title="Cluster")
    axes[1].add_artist(legend2)

    # 子图 3：按被试 ID 着色 (排查最大元凶)
    scatter3 = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subject, cmap='tab10', alpha=0.6, s=15)
    axes[2].set_title('Subject Identity (Domain Shift Check)')
    legend3 = axes[2].legend(*scatter3.legend_elements(), title="Subject ID")
    axes[2].add_artist(legend3)

    plt.tight_layout()
    plt.show()
    plt.savefig('./EEG-Conformer/infer_results/feature_diagnostic.png', dpi=300)

if __name__ == "__main__":
    main()