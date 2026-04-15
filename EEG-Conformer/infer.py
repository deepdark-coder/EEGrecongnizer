import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 【关键】从你之前的训练脚本中导入 ViT 模型结构
# 假设你的训练脚本命名为 conformer_train.py，请根据实际文件名修改
from conformer_pro import ViT, ExGAN 

def load_and_preprocess_data(data_path):
    """
    加载数据并执行与训练时完全一致的预处理（Z-score标准化 + 增加通道维）
    """
    print(f"📥 正在加载验证数据: {data_path}")
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
    print(f"🎯 最终预测准确率 (Accuracy): {accuracy * 100:.2f}%")
    print("="*45 + "\n")



def main():
    # ================= 参数配置区 =================
    # 1. 填入你想要测试的模型权重路径 (例如取某个被试 fold1 的最佳权重)
    MODEL_WEIGHTS_PATH = './EEG-Conformer/params/pretrain_best.pth'
    
    # 2. 填入测试数据的路径 (可以是用来验证的某个 held-out 被试数据)
    TEST_DATA_PATH = './EEG-Conformer/data/processed_normal/HC1068_1s.mat'
    
    # 3. 结果图表保存路径
    SAVE_DIR = './EEG-Conformer/infer_results/'
    # ==============================================

    # 初始化模型
    print("🚀 初始化 EEG-Conformer 模型...")
    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=40)
    model = ViT(emb_size=40, depth=2, n_classes=2, n_channels=30, seq_len=seq_len).cuda()
    
    # 加载权重
    # 注意：如果训练时用了 nn.DataParallel，权重字典的 key 会带有 'module.' 前缀
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location='cuda')
    # 兼容处理：检查保存的是单纯的 state_dict 还是包含 epoch 等信息的字典
    state_dict = checkpoint.get('model_state', checkpoint) 
    
    # 如果推理时没有用 DataParallel，需要手动剥离 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("✅ 权重加载成功！")

    # 加载数据
    x_test, y_true = load_and_preprocess_data(TEST_DATA_PATH)
    
    # 构建 DataLoader 以防止显存溢出 (OOM)
    dataset = torch.utils.data.TensorDataset(x_test, y_true)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    print("🧠 开始前向推理计算...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda()
            _, outputs = model(imgs)
            
            # 获取预测的类别 (0 或 1)
            preds = outputs.argmax(dim=1).cpu().tolist()
            
            # 获取预测为正类(1)的概率，用于画 ROC 曲线
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().tolist()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 打印终端文本分类报告 (Precision, Recall, F1-score)
    print("\n" + "="*50)
    print("="*50 + "\n")

    # 绘制并保存图表
    plot_results(all_labels, all_preds, all_probs, SAVE_DIR)

if __name__ == '__main__':
    main()