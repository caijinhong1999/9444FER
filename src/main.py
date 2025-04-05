import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import model_cdnn  # 你提供的模型文件
import model_vgg
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#数据集处理使用单独的.py文件
'''def create_softlabels(fer2013, ferplus, output_dir)
    emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness',
                    'anger', 'disgust', 'fear', 'contempt']
    ferplus["pixels"] = fer2013["pixels"]
    # 过滤掉 NF ≠ 0 或 unknown > 1 的行
    ferplus = ferplus[(ferplus["NF"] == 0) & (ferplus["unknown"] <= 1)].copy()
    # 将得票数转换为概率分布（soft labels）
    ferplus[emotion_cols] = ferplus[emotion_cols].div(ferplus[emotion_cols].sum(axis=1), axis=0)
    ferplus.to_csv(os.path.join(output_dir,"fer_softlabels.csv"), index=False)

def create_onehot(fer2013, ferplus, output_dir):
    emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness',
                    'anger', 'disgust', 'fear', 'contempt']
    ferplus["pixels"] = fer2013["pixels"]
    # 过滤掉 NF ≠ 0 或 unknown > 1 的行
    ferplus = ferplus[(ferplus["NF"] == 0) & (ferplus["unknown"] <= 1)].copy()
    # 找到每行得票最多的情绪类别
    majority_emotion = ferplus[emotion_cols].idxmax(axis=1)
    # 将情绪标签列修改为 one-hot 编码（只保留得票最多的类别为1，其余为0）
    for col in emotion_cols:
        ferplus[col] = (majority_emotion == col).astype(int)
    ferplus.to_csv(os.path.join(output_dir,"fer_onehot.csv"), index=False)'''

# 自定义数据集类
class FERPlusDataset(Dataset):
    def __init__(self, csv_file, usage='Training', transform=None):
        self.data = pd.read_csv(csv_file)
        self.data.columns = self.data.columns.str.strip()  # 清除列名空格
        self.data = self.data[self.data['Usage'] == usage]

        # 排除 unknown 和 NF 分布不为 0 的行（可选）
        self.data = self.data[(self.data['NF'] == 0) & (self.data['unknown'] <= 1)]

        self.label_keys = ['neutral', 'happiness', 'surprise', 'sadness',
                           'anger', 'disgust', 'fear', 'contempt']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = np.fromstring(row['pixels'], dtype=int, sep=' ').astype(np.uint8).reshape(48, 48)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)

        label = row[self.label_keys].values.astype(np.float32)
        label = label / label.sum()  # soft label 归一化
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label


# 混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

# 评估模型
def evaluate_model(model, data_loader, device, name="Validation"):
    model.eval()
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    mse_loss_fn = nn.MSELoss(reduction='mean')

    total_kl = 0.0
    total_expected_acc = 0.0
    total_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            log_probs = torch.log_softmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            # KL散度
            kl = kl_loss_fn(log_probs, labels)
            # Expected Accuracy（分布点积平均值）
            expected_acc = torch.sum(probs * labels, dim=1).mean()
            # 均方误差
            mse = mse_loss_fn(probs, labels)

            batch_size = images.size(0)
            total_kl += kl.item() * batch_size
            total_expected_acc += expected_acc.item() * batch_size
            total_mse += mse.item() * batch_size
            num_samples += batch_size

    avg_kl = total_kl / num_samples
    avg_expected_acc = total_expected_acc / num_samples
    avg_mse = total_mse / num_samples

    print(f"\n{name} Evaluation Metrics:")
    print(f"  KL Divergence:      {avg_kl:.4f}")
    print(f"  Expected Accuracy:  {avg_expected_acc:.4f}")
    print(f"  Mean Squared Error: {avg_mse:.4f}\n")

    return avg_kl, avg_expected_acc, avg_mse

def plot_training_curves(train_losses, val_kls, val_expected_accuracies, val_mses):
    epochs = list(range(1, len(train_losses)+1))

    plt.figure(figsize=(14, 10))

    # 1. 训练损失（KL Loss）
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train KL Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (KL)")
    plt.grid(True)

    # 2. 验证 KL Divergence
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_kls, label='Val KL Divergence', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("Validation KL Divergence")
    plt.grid(True)

    # 3. 验证 Expected Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_expected_accuracies, label='Val Expected Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Expected Accuracy")
    plt.title("Validation Expected Accuracy")
    plt.grid(True)

    # 4. 验证 MSE
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_mses, label='Val MSE', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Validation MSE")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    fer_path = '../Data/fer2013_softlabel.csv'  # 包含soft labels的csv
    train_dataset = FERPlusDataset(fer_path, usage='Training')
    val_dataset = FERPlusDataset(fer_path, usage='PublicTest')
    test_dataset = FERPlusDataset(fer_path, usage='PrivateTest')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_vgg.VGG().to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    softmax = nn.LogSoftmax(dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_kls = []
    val_expected_accuracies = []
    val_mses = []

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(softmax(outputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_kl, val_acc, val_mse = evaluate_model(model, val_loader, device, name="Validation")
        val_kls.append(val_kl)
        val_expected_accuracies.append(val_acc)
        val_mses.append(val_mse)

        #用每轮平均loss表示loss
        print(f"Epoch [{epoch + 1}/{num_epochs}] Avg KL Loss: {avg_loss:.4f}")
        #print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}")

        # 每轮都在验证集上评估
        evaluate_model(model, val_loader, device, name="Validation")

    # 训练完成后在测试集评估
    evaluate_model(model, test_loader, device, name="Test")

    plot_training_curves(train_losses, val_kls, val_expected_accuracies, val_mses)

if __name__ == "__main__":
    main()