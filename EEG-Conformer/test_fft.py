import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def extract_log_psd_for_plot(data, fs=250):
    trials, channels, timepoints = data.shape
    fft_data = np.fft.fft(data, axis=2)
    power_spectrum = (np.abs(fft_data) ** 2) / timepoints
    freqs = np.fft.fftfreq(timepoints, 1/fs)
    
    idx = np.where((freqs >= 1) & (freqs <= 50))[0]
    psd_features = power_spectrum[:, :, idx]
    return np.log10(psd_features + 1e-8), freqs[idx]

# 请替换为你实际的测试文件路径
mat_path = './EEG-Conformer/data/processed_normal/HC1021_1s.mat'
mat = sio.loadmat(mat_path)
data = np.ascontiguousarray(mat['data'], dtype=np.float32)
label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

# 提取特征与对应频率
log_psd, freqs = extract_log_psd_for_plot(data)

# 计算两类情绪的全局平均功率
neutral_psd = log_psd[label == 0].mean(axis=(0, 1))
positive_psd = log_psd[label == 1].mean(axis=(0, 1))

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(freqs, neutral_psd, label='Neutral (0)', linewidth=2, color='#1f77b4')
plt.plot(freqs, positive_psd, label='Positive (1)', linewidth=2, color='#ff7f0e')

# 划分出经典的脑电频带区域，方便观察
plt.axvspan(1, 4, alpha=0.1, color='gray', label='Delta')
plt.axvspan(4, 8, alpha=0.1, color='blue', label='Theta')
plt.axvspan(8, 13, alpha=0.1, color='green', label='Alpha')
plt.axvspan(13, 30, alpha=0.1, color='orange', label='Beta')
plt.axvspan(30, 50, alpha=0.1, color='red', label='Gamma')

plt.title('Log-PSD (1-50Hz): Neutral vs Positive', fontsize=14)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Log Power', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.savefig('./EEG-Conformer/infer_results/psd_comparison.png', dpi=300)