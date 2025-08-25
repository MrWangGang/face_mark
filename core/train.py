import torch
from ultralytics import YOLO

# 1. 加载预训练的YOLOv8模型
model = YOLO("./yolo_model/yolov8n.pt")
# 检查GPU是否可用
if torch.cuda.is_available():
    print("GPU is available!")
    device = '0'  # 使用第一个GPU，如果有多个GPU，可以使用 '0,1,2' 等指定多个GPU
else:
    print("GPU is not available, using CPU instead.")
    device = 'cpu'
# 2. 设置训练参数
# 这里是训练模型的配置
# 例如：指定数据集路径、训练批量大小、学习率、epoch 数量等
# 定义训练参数
# 调用模型的 train 方法开始训练
# 开始训练
# 调用模型的 train 方法开始训练，并将训练结果存储在 result 变量中
# 2. 训练配置
result = model.train(
    data='./datasets/dataset.yaml',  # 指定数据集
    cache='disk',  # 启用磁盘缓存
    imgsz=640,  # 统一输入图像大小
    epochs=60,  # 增加训练轮次，确保充分训练
    batch=8,  # 增加 batch size，提高稳定性
    close_mosaic=50,  # 关闭 mosaic 数据增强的 epoch
    workers=8,  # 使用 8 线程加载数据
    optimizer='AdamW',  # 使用 AdamW 优化器
    lr0=0.002,  # 适当调整初始学习率
    lrf=0.05,  # 适当降低最终学习率
    momentum=0.95,  # 提高动量，加速收敛
    weight_decay=2e-4,  # 降低权重衰减，提高召回率
    degrees=5,  # 轻微旋转数据增强
    translate=0.1,  # 轻微平移数据增强
    scale=0.15,  # 适当缩放
    shear=2,  # 适当剪切
    fliplr=0.2,  # 适当水平翻转
    flipud=0.1,  # 适当垂直翻转
    pretrained=True,  # 使用预训练权重
    patience=30,  # 增加耐心值，防止过早停止
    cos_lr=True,  # 采用余弦退火学习率
    save_period=5,  # 更频繁地保存模型
    amp=False,  # 关闭/启用自动混合精度加速训练
    iou=0.4  # 降低 IoU 阈值，提高召回率
)