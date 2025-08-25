import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO("./yolo_model/best_train.pt")

    # 设置Streamlit应用的标题，并使其字体更小
    col1, col2 = st.columns([3, 1])  # 划分两个列，col1占3份，col2占1份
    with col1:
        st.markdown("<h4 style='text-align: left; color: black;'>人脸检测与标注</h4>", unsafe_allow_html=True)

    # 上传图片
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])
    st.markdown("<p style='color: grey;'>鼠标悬停图片，右上角可点击放大！</p>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # 读取上传的文件为 NumPy 数组
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # 以 BGR 方式读取图片

        if img_array is None:
            st.error("无法加载图像，请尝试其他格式。")
            return

        # 保存原图
        original_img = img_array.copy()

        # 运行YOLO模型进行预测
        results = model(img_array)

        # 解析检测结果
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
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 更新右侧显示的人脸数量
        with col2:
            st.write(f"检测到的人脸数量: {total_detections}")  # 更新检测到的总数量

        # 将 BGR 转换为 RGB
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # 并排展示原图和标注后的图片
        col3, col4 = st.columns(2)
        with col3:
            st.image(original_img, caption="原图", use_container_width=True)
        with col4:
            st.image(img_array, caption="标注图片", use_container_width=True)

if __name__ == "__main__":
    main()
