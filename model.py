import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # 第一个卷积层：输入通道1（灰度图），输出通道32，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层：输入通道32，输出通道64，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别（0-9）

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))  # 28x28 -> 28x28
        x = F.max_pool2d(x, 2)     # 28x28 -> 14x14

        # 第二个卷积块
        x = F.relu(self.conv2(x))  # 14x14 -> 14x14
        x = F.max_pool2d(x, 2)     # 14x14 -> 7x7

        # 展平操作
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # 输出层使用log_softmax
        return F.log_softmax(x, dim=1)

def load_model(model_path):
    """
    加载预训练模型
    """
    model = DigitRecognizer()
    # 使用weights_only=True参数，提高安全性
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model