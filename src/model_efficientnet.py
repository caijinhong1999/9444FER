import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm


# EfficientNet 模型结构
class EfficientNetSoftLabel(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetSoftLabel, self).__init__()
        self.base = efficientnet_b0(weights='DEFAULT')
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base.classifier[1].in_features, num_classes),
            nn.LogSoftmax(dim=1)  # 用于 KLDivLoss
        )

    def forward(self, x):
        return self.base(x)

# FER+ Dataset 类（带 soft label）
class FERPlusSoftLabelDataset(Dataset):
    def __init__(self, csv_path, usage, transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df['Usage'] == usage].reset_index(drop=True)
        self.transform = transform
        print(f"[{usage}] subset loaded: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            pixel_seq = self.data.loc[idx, 'pixels']
            image = np.array([int(p) for p in pixel_seq.split()], dtype=np.uint8).reshape(48, 48)
            image = Image.fromarray(image).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = self.data.iloc[idx, 2:10].astype(float).values
            label = torch.tensor(label, dtype=torch.float32)
            return image, label
        except Exception as e:
            print(f"❌ 样本 idx={idx} 加载失败：{e}")
            return self.__getitem__((idx + 1) % len(self.data))  # 尝试下一个样本

# 训练与验证主流程
def train_model():
    val_accuracies = []
    val_losses = []

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("💻 当前设备:", device)
    print("GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = FERPlusSoftLabelDataset("fer2013_softlabel.csv", "Training", transform)
    val_dataset = FERPlusSoftLabelDataset("fer2013_softlabel.csv", "PrivateTest", transform)
    df = pd.read_csv("fer2013_softlabel.csv")
    print(df["Usage"].value_counts())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    print(f"[VAL CHECK] 批次数: {len(val_loader)}")

    model = EfficientNetSoftLabel(num_classes=8).to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    per_class_acc = None
    per_class_loss = None
    epoch_losses = []  # 记录每轮训练 loss

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 🟡 每轮都跑验证（含 per-class acc/loss）→ 重复以下块
        model.eval()
        correct = 0
        total = 0
        num_classes = 8
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        class_loss_sum = [0.0] * num_classes

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                true = torch.argmax(labels, dim=1)
                correct += (preds == true).sum().item()
                total += labels.size(0)
                loss_batch = nn.functional.kl_div(outputs, labels, reduction='none').sum(dim=1)

                for i in range(len(true)):
                    label = true[i].item()
                    class_total[label] += 1
                    if preds[i].item() == label:
                        class_correct[label] += 1
                    class_loss_sum[label] += loss_batch[i].item()

        if total == 0:
            print("⚠️ 验证集为空或未被正确遍历，跳过此轮准确率计算。")
            continue

        acc = correct / total
        print(f"✅ Epoch {epoch + 1} Validation Accuracy: {acc:.4f}")
        val_accuracies.append(acc)

        # 计算整个验证集的平均 KL loss（注意不是 per-class，是总值）
        total_kl_loss = sum(class_loss_sum)
        total_kl_loss /= sum(class_total)  # 相当于平均每个样本的 loss
        val_losses.append(total_kl_loss)

        if epoch == 9:  # ✅ 最后一个 epoch 才保存绘图数据
            per_class_acc = [c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
            per_class_loss = [l / t if t > 0 else 0 for l, t in zip(class_loss_sum, class_total)]

    epochs = range(1, len(val_accuracies) + 1)

    # Training Loss 折线图
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_losses, marker='o', color='blue')
    plt.title("Training KL Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss (Training)")
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig("training_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Accuracy 折线图
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accuracies, marker='o', color='green')
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig("validation_accuracy_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 验证 Loss 折线图
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_losses, marker='o', color='orange')
    plt.title("Validation KL Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig("validation_kl_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    class_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]
    num_classes = 8
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    angles += angles[:1]

    # 🎯 雷达图 1：Accuracy
    stats_acc = per_class_acc + [per_class_acc[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats_acc, 'o-', linewidth=2, label='Per-class Accuracy')
    ax.fill(angles, stats_acc, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), class_names)
    ax.set_title("Per-class Accuracy Radar Chart")
    ax.grid(True)
    plt.legend(loc='upper right')
    plt.savefig("radar_per_class_accuracy.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 📉 雷达图 2：KL Loss（越低越好）
    stats_loss = per_class_loss + [per_class_loss[0]]

    fig, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax2.plot(angles, stats_loss, 'o-', linewidth=2, color='red', label='Per-class KL Loss')
    ax2.fill(angles, stats_loss, alpha=0.25, color='red')
    ax2.set_thetagrids(np.degrees(angles[:-1]), class_names)
    ax2.set_title("Per-class KL Divergence Radar Chart")
    ax2.grid(True)
    plt.legend(loc='upper right')
    plt.savefig("radar_per_class_kl_loss.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train_model()
