import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image):
    """
    预处理图像用于模型输入
    """
    # 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    # 调整大小为28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组
    image_array = np.array(image)
    
    # 反转颜色（因为MNIST数据集是白底黑字，而我们的画板是黑底白字）
    image_array = 255 - image_array
    
    # 转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 添加批次维度
    image_tensor = transform(Image.fromarray(image_array)).unsqueeze(0)
    return image_tensor

def predict_digit(model, image_tensor):
    """
    使用模型预测数字
    """
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item()

def save_image(image, filename):
    """
    保存图像到文件
    """
    image.save(filename)

def clear_canvas(canvas):
    """
    清除画布内容
    """
    canvas.delete("all")