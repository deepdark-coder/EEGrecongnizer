import os
import torch
import numpy as np
import scipy.io as sio
import glob
import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 【关键】从你之前的训练脚本中导入 ViT 模型结构
# 假设你的训练脚本命名为 conformer_train.py，请根据实际文件名修改
from conformer import ViT, ExGAN 

def load_and_preprocess_data(data_path):
    """
    加载数据并执行与训练时完全一致的预处理（Z-score标准化 + 增加通道维）
    """
    print(f"=正在加载验证数据: {data_path}")
    mat = sio.loadmat(data_path)
    
    data = np.ascontiguousarray(mat['data'], dtype=np.float32)
    label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)
    
    # 必须执行与训练集相同的标准化操作
    mu = data.mean()
    std = data.std() + 1e-8
    data = (data - mu) / std
    
    # 增加通道维度: (N, 30, 250) -> (N, 1, 30, 250)
    data = np.ascontiguousarray(data[:, np.newaxis, :, :], dtype=np.float32)
    
    return torch.tensor(data,dtype=torch.float32), torch.tensor(label,dtype=torch.long)

def plot_results(y_true, y_pred, y_prob, save_dir):
    # 仅计算准确率：正确预测的样本数 / 总样本数
    accuracy = (y_true == y_pred).mean()
    
    # 极简输出，只 print 精度
    print("\n" + "="*45)
    print(f"最终预测准确率 (Accuracy): {accuracy * 100:.2f}%")
    print("="*45 + "\n")



def main():
    # ================= 参数配置区 =================
    MODEL_WEIGHTS_PATH = './EEG-Conformer/last_params/D2_H4_S40_best1.pth'
    TEST_DATA_DIR = './EEG-Conformer/data/processed_normal/'
    # ==============================================

    # 【替换部分】：使用 os.listdir 替代 glob 获取所有 .mat 文件
    if not os.path.exists(TEST_DATA_DIR):
        print(f"错误：测试数据文件夹 {TEST_DATA_DIR} 不存在！请检查路径。")
        return
        
    test_files = [
        os.path.join(TEST_DATA_DIR, f) 
        for f in os.listdir(TEST_DATA_DIR) 
        if f.endswith('.mat')
    ]
    
    # 强制排序，让输出按被试编号整齐排列 (可选)
    test_files.sort()

    if len(test_files) == 0:
        print(f"错误：在 {TEST_DATA_DIR} 下没有找到任何 .mat 文件！")
        return

    print(f"找到 {len(test_files)} 个测试文件，开始全局连续推理...\n")

    # 初始化模型
    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=40)
    model = ViT(emb_size=40, depth=2, n_classes=2, n_channels=30, seq_len=seq_len).cuda()
    
    # 加载权重
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location='cuda')
    state_dict = checkpoint.get('model_state', checkpoint) 
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("权重加载成功！\n")

    # 定义全局统计变量
    total_correct = 0
    total_samples = 0

    print("开始逐个文件推理计算...")
    
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        
        try:
            x_test, y_true = load_and_preprocess_data(file_path)
        except Exception as e:
            print(f"读取 {file_name} 失败，已跳过。报错信息: {e}")
            continue
        
        dataset = torch.utils.data.TensorDataset(x_test, y_true)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        file_correct = 0
        file_total = len(y_true)

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                _, outputs = model(imgs)
                
                preds = outputs.argmax(dim=1)
                file_correct += (preds == labels).sum().item()

        file_acc = file_correct / file_total if file_total > 0 else 0
        print(f"  ▶ {file_name}: 样本 {file_total} 个 | 命中 {file_correct} 个 | 准确率 {file_acc * 100:.2f}%")

        total_correct += file_correct
        total_samples += file_total

    # 汇总计算
    if total_samples > 0:
        overall_acc = total_correct / total_samples
        print("\n" + "="*50)
        print("全局推理执行完毕！")
        print(f"总测试样本数: {total_samples}")
        print(f"总命中样本数: {total_correct}")
        print(f"最终总预测准确率 (Overall Accuracy): {overall_acc * 100:.2f}%")
        print("="*50 + "\n")
    else:
        print("\n最终没有成功加载任何有效样本，无法计算精度。")

if __name__ == '__main__':
    main()

