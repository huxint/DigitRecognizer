import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import torch
from model import load_model
from utils import preprocess_image, predict_digit, save_image, clear_canvas

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别系统")

        # 创建画布
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)

        # 创建按钮框架
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        # 创建按钮
        self.predict_button = tk.Button(button_frame, text="预测", command=self.predict)
        self.clear_button = tk.Button(button_frame, text="清除", command=self.clear)
        self.save_button = tk.Button(button_frame, text="保存", command=self.save)

        # 放置按钮
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # 创建结果显示标签
        self.result_label = tk.Label(root, text="预测结果: ", font=('Arial', 14))
        self.result_label.pack(pady=10)

        # 初始化绘图变量
        self.last_x = None
        self.last_y = None
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # 绑定鼠标事件
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        # 加载模型
        try:
            self.model = load_model('model.pth')
        except:
            messagebox.showerror("错误", "模型文件未找到，请确保model.pth文件存在！")
            root.destroy()

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.last_x and self.last_y:
            # 使用更细的线条
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=15, fill='black', capstyle=tk.ROUND,
                                  smooth=tk.TRUE, splinesteps=36)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill='black', width=15)
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def predict(self):
        # 预处理图像
        image_tensor = preprocess_image(self.image)

        # 预测数字
        prediction = predict_digit(self.model, image_tensor)

        # 更新结果显示
        self.result_label.config(text=f"预测结果: {prediction}")

    def clear(self):
        clear_canvas(self.canvas)
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="预测结果: ")

    def save(self):
        try:
            self.image.save('digit.png')
            messagebox.showinfo("成功", "图片已保存为 digit.png")
        except:
            messagebox.showerror("错误", "保存图片失败！")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()