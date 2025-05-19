import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitRecognizer
import os

def train(model, device, train_loader, optimizer, epoch):
    """
    训练一个epoch
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    """
    测试模型性能
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    
    # 检查是否可以使用CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转±10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.RandomPerspective(0.1),  # 随机透视变换
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, transform=test_transform)

    # 使用更大的batch size和更多的worker
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, num_workers=4)

    # 创建模型实例
    model = DigitRecognizer().to(device)
    
    # 使用Adam优化器，添加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    # 训练参数
    epochs = 20  # 增加训练轮数
    best_accuracy = 0

    # 训练循环
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        
        # 更新学习率
        scheduler.step(accuracy)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'model.pth')
            print(f"模型已保存，准确率: {accuracy:.2f}%")

if __name__ == '__main__':
    main()