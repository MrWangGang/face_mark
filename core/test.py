from ultralytics import YOLO
import cv2
import os

# 1. 加载训练好的 YOLOv8 模型
model = YOLO("./yolo_model/best_train.pt")  # 你的 best_train.pt 路径

# 2. 读取要检测的图片
image_path = "./input.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)

# 确保图片成功读取
if image is None:
    print(f"错误：无法读取图片 {image_path}，请检查路径是否正确！")
    exit(1)

# 3. 进行目标检测
results = model(image)

# 4. 解析检测结果
total_detections = 0  # 计数变量

for result in results:
    num_boxes = len(result.boxes)  # 当前图片的检测目标数量
    total_detections += num_boxes

    # 解析检测框并绘制
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
        confidence = box.conf[0]  # 置信度
        class_id = int(box.cls[0])  # 类别 ID
        label = f"{model.names[class_id]} {confidence:.2f}"  # 生成标签

        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 7. 保存标注后的图片
output_path = "./output.jpg"

# 确保目录存在
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 尝试保存图片
success = cv2.imwrite(output_path, image)

if success:
    print(f"图片已成功保存: {output_path}")
else:
    print("图片保存失败，请检查路径或权限！")

# 5. 显示检测总数
print(f"总共检测到 {total_detections} 个目标")

# 6. 显示检测结果
cv2.imshow("YOLOv8 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()





