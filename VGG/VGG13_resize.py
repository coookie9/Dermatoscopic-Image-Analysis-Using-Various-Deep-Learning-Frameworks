import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist.dataset import DermaMNIST
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # 直接进行 Resize 操作
])

train_dataset = DermaMNIST(root="./", split="train", transform=transform)
test_dataset = DermaMNIST(root="./", split="test", transform=transform)

# 显示部分经过预处理的图像
sample_images, _ = next(iter(DataLoader(train_dataset, batch_size=5, shuffle=True)))
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img = transforms.ToPILImage()(sample_images[i])
    axes[i].imshow(img)
    axes[i].axis('off')
plt.show()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化预训练的VGG13模型
vgg_model = models.vgg13(pretrained=True)

# 更改最后的全连接层以适应我们的分类任务
num_classes = 7  # DermMNIST有7个类别
vgg_model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)

# 将模型移到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = vgg_model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 150
for epoch in range(num_epochs):
    vgg_model.train()
    all_true_labels = []
    all_predicted_probs = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.squeeze().to(device)
        optimizer.zero_grad()
        outputs = vgg_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    vgg_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.squeeze().to(device)
            outputs = vgg_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 添加真实标签和预测概率
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    # 计算 ACC和AUC
    accuracy = correct / total
    auc = roc_auc_score(all_true_labels, all_predicted_probs, multi_class='ovr')
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {100 * accuracy:.2f}%, Test AUC: {auc:.4f}')

# 清理模型
del vgg_model
torch.cuda.empty_cache()
