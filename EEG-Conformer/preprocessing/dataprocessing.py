import h5py
import numpy as np
import os
import re

def process_task_data():
    raw_data_dir = './EEG-Conformer/data/normal/'
    save_dir = './EEG-Conformer/data/processed_normal/'
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有 .mat 文件
    mat_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('timedata.mat')])
    
    print(f"找到 {len(mat_files)} 个数据文件")
    
    for idx, mat_file in enumerate(mat_files, 1):
        file_path = os.path.join(raw_data_dir, mat_file)
        subject_id = re.search(r'(\d+)', mat_file).group(1)
        
        print(f"[{idx}/{len(mat_files)}] 正在处理 {mat_file} (ID: {subject_id}) ...")
        
        try:
            # 使用 h5py 读取 MATLAB v7.3 文件
            with h5py.File(file_path, 'r') as mat_data:
                # 打印所有可用的 keys
                print(f"    数据包含的 keys: {list(mat_data.keys())}")
                
                
                # 假设数据形状为 (30, 50000)
                eeg_neu = np.array(mat_data['EEG_data_neu']).T  # (50000, 30) -> (30, 50000)
                eeg_pos = np.array(mat_data['EEG_data_pos']).T  # (50000, 30) -> (30, 50000)
                
                # 1. 滑动窗口切片
                neu_sliced = eeg_neu.reshape(eeg_neu.shape[0], 200, 250).transpose(1, 0, 2)
                pos_sliced = eeg_pos.reshape(eeg_pos.shape[0], 200, 250).transpose(1, 0, 2)
                
                # 2. 生成标签
                neu_labels = np.zeros(200, dtype=np.int64)
                pos_labels = np.ones(200, dtype=np.int64)
                
                # 3. 合并数据
                sub_data = np.concatenate((neu_sliced, pos_sliced), axis=0)
                sub_labels = np.concatenate((neu_labels, pos_labels), axis=0)

                # ── 新增：预先打乱（固定seed保证可复现）──────────────────
                rng        = np.random.default_rng(seed=42)
                perm       = rng.permutation(len(sub_data))
                sub_data   = sub_data[perm]
                sub_labels = sub_labels[perm]
                
                # 4. 保存（还是用 scipy 保存为 v7.2 格式）
                import scipy.io as sio
                output_file = os.path.join(save_dir, f'HC{subject_id}_1s.mat')
                sio.savemat(output_file, {'data': sub_data, 'label': sub_labels})
                
                print(f"    ✓ 保存完成: {output_file}")
                print(f"    数据形状: {sub_data.shape}, 标签形状: {sub_labels.shape}\n")
                
        except Exception as e:
            print(f"    ✗ 处理失败: {str(e)}\n")
            continue

if __name__ == '__main__':
    process_task_data()