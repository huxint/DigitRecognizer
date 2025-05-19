<<<<<<< HEAD
# DigitRecognizer
DigitRecognizerApp
=======
# 手写数字识别系统

这是一个基于Python实现的手写数字识别系统，使用MNIST数据集训练的模型来识别用户手写的数字。

## 项目原理

1. **数据预处理**：
   - 使用MNIST数据集进行模型训练
   - 图像预处理包括归一化、调整大小等操作

2. **模型架构**：
   - 使用卷积神经网络(CNN)进行特征提取
   - 包含多个卷积层、池化层和全连接层
   - 使用Softmax激活函数输出0-9的概率分布

3. **交互界面**：
   - 使用tkinter创建图形用户界面
   - 提供画板功能用于手写输入
   - 实时显示识别结果
   - 支持保存手写图片

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- Pillow
- tkinter (Python标准库)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行主程序：
```bash
python main.py
```

2. 界面操作说明：
   - 在画板上用鼠标手写数字
   - 点击"预测"按钮进行识别
   - 点击"清除"按钮重新书写
   - 点击"保存"按钮保存当前手写图片

## 项目结构

```
├── README.md
├── requirements.txt
├── main.py              # 主程序入口
├── model.py             # 模型定义
├── train.py             # 模型训练脚本
└── utils.py             # 工具函数
```
>>>>>>> e280ba3 (第一次提交)
