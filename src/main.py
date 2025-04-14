from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import model_efficientnet as model_class
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import model_cdnn as model_class # 你提供的模型文件
# import model_vgg  as model_class
# import model_resnet  as model_class

# 将 fer2013.csv 转换为图片保存
def save_images_as_png(fer_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(fer_df)):
        pixel_str = fer_df['pixels'][i]
        pixel_list = [int(pixel) for pixel in pixel_str.split()]
        image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(image)
        img_path = os.path.join(output_dir, f"image_{i}.png")
        img.save(img_path)

# 自定义数据集类
class FERPlusDataset(Dataset):
    def __init__(self, csv_file, usage='Training', transform=None):
        self.data = pd.read_csv(csv_file)
        self.data.columns = self.data.columns.str.strip()  # 清除列名空格
        self.data = self.data[self.data['Usage'] == usage]

        # 排除 unknown 和 NF 分布不为 0 的行（可选）
        self.data = self.data[(self.data['NF'] == 0) & (self.data['unknown'] <= 1)]

        self.label_keys = ['neutral', 'happiness', 'surprise', 'sadness',
                           'anger', 'disgust', 'fear', 'contempt', 'unknown']   #9种class
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

            # 归一化的Expected Accuracy（分布点积平均值）
            numerator = torch.sum(probs * labels, dim=1)
            denominator = torch.sum(labels * labels, dim=1) + 1e-10  # 避免除零
            expected_acc = (numerator / denominator).mean()

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

# train模型可视化
def plot_training_curves(train_losses,
                         train_kls, val_kls,
                         train_expected_accuracies, val_expected_accuracies,
                         train_mses, val_mses):
    epochs = list(range(1, len(train_losses)+1))

    # 1. Train KL Loss vs Val KL Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_kls, label='Train KL', color='blue')
    plt.plot(epochs, val_kls, label='Val KL', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence (Train vs Val)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Expected Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_expected_accuracies, label='Train Expected Accuracy', color='blue')
    plt.plot(epochs, val_expected_accuracies, label='Val Expected Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Expected Accuracy")
    plt.title("Expected Accuracy (Train vs Val)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. MSE
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_mses, label='Train MSE', color='blue')
    plt.plot(epochs, val_mses, label='Val MSE', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE (Train vs Val)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Raw training KL loss (if different from train_kl)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Raw Training KL Loss (from optimizer)', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("Training KL Loss (from backprop)")
    plt.legend()
    plt.grid(True)
    plt.show()

# test模型可视化
def plot_test_curve(test_kl, test_acc, test_mse):
    metrics = ['KL Divergence', 'Expected Accuracy', 'MSE']
    values = [test_kl, test_acc, test_mse]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom')

    plt.title("Test Set Evaluation Metrics")
    plt.ylim(0, max(values)*1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# 训练模型
def train_model(model, train_loader, val_loader, optimizer, criterion, softmax, device, num_epochs):
    train_losses = []
    train_kls = []
    val_kls = []
    train_expected_accuracies = []
    val_expected_accuracies = []
    train_mses = []
    val_mses = []

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
        print(f"Epoch [{epoch + 1}/{num_epochs}] Avg Training KL Loss: {avg_loss:.4f}")

        train_kl, train_acc, train_mse = evaluate_model(model, train_loader, device, name="Train")
        val_kl, val_acc, val_mse = evaluate_model(model, val_loader, device, name="Validation")

        train_losses.append(avg_loss)
        train_kls.append(train_kl)
        val_kls.append(val_kl)
        train_expected_accuracies.append(train_acc)
        val_expected_accuracies.append(val_acc)
        train_mses.append(train_mse)
        val_mses.append(val_mse)

    return model, train_losses, train_kls, val_kls, train_expected_accuracies, val_expected_accuracies, train_mses, val_mses

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def main():
    fer_path = '../data/fer2013_softlabel.csv'
    train_dataset = FERPlusDataset(fer_path, usage='Training')
    val_dataset = FERPlusDataset(fer_path, usage='PublicTest')
    test_dataset = FERPlusDataset(fer_path, usage='PrivateTest')

    batch_size = model_class.VGG13_PyTorch().batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 01. 调用cdnn9层神经网络
    # model = model_cdnn.NineLayerCNN().to(device)
    # 02. 调用cdnn12层神经网络
    # model = model_cdnn.TwelveLayerCNN().to(device)
    # 03. 手动构建vgg
    model = model_class.VGG13_PyTorch().to(device)

    # 04. vgg
    # model = model_vgg.VGG().to(device)
    # 05. resnet
    # model = model_resnet.ResNet().to(device)


    model.apply(init_weights)

    softmax = nn.LogSoftmax(dim=1)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=model_class.VGG13_PyTorch().lr)
    num_epochs = model_class.VGG13_PyTorch().epoch

    # 抽取后的训练过程调用
    model, train_losses, train_kls, val_kls, train_expected_accuracies, val_expected_accuracies, train_mses, val_mses = \
        train_model(model, train_loader, val_loader, optimizer, criterion, softmax, device, num_epochs)

    # 测试集评估 & 可视化
    test_kl, test_acc, test_mse = evaluate_model(model, test_loader, device, name="Test")
    plot_training_curves(train_losses, train_kls, val_kls,
                         train_expected_accuracies, val_expected_accuracies,
                         train_mses, val_mses)
    plot_test_curve(test_kl, test_acc, test_mse)


if __name__ == "__main__":
    main()