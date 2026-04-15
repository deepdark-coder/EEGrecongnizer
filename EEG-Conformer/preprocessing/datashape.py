import h5py
import numpy as np

file_path = './EEG-Conformer/data/normal/HC1003timedata.mat'

# 查看文件内容
with h5py.File(file_path, 'r') as f:
    print("文件中的所有 keys:")
    for key in f.keys():
        print(f"  {key}: 形状 {f[key].shape}, 类型 {f[key].dtype}")
    
    # 读取一个数据看看
    sample_data = np.array(f['EEG_data']).T
    print(f"\n读取的数据形状: {sample_data.shape}")
    print(f"数据范围: [{sample_data.min():.3f}, {sample_data.max():.3f}]")