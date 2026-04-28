import cv2
import numpy as np
import json
import os


def load_image(image_path):
    """加载图片，返回BGR格式图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")
    return image


def save_image(image_path, image):
    """保存图片"""
    cv2.imwrite(image_path, image)
    print(f"图片已保存: {image_path}")


def load_images_from_paths(image_paths):
    """从路径列表加载多张图片"""
    images = []
    for path in image_paths:
        images.append(load_image(path))
    return images


def draw_bounding_box(image, box, color=(0, 0, 255), thickness=2):
    """在图像上绘制矩形框
    box: (x, y, w, h) 左上角坐标和宽高
    """
    x, y, w, h = box
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return image


def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, 
              color=(0, 0, 255), thickness=2):
    """在图像上绘制文本"""
    cv2.putText(image, text, (int(position[0]), int(position[1])), font, font_scale, color, thickness)
    return image


def draw_detections(image, detections, draw_labels=True):
    """在图像上绘制所有检测结果
    detections: 列表，每个元素包含 {'box': (x,y,w,h), 'similarity': float, 'source_img': int}
    """
    for det in detections:
        box = det['box']
        similarity = det['similarity']
        
        image = draw_bounding_box(image, box, color=(0, 0, 255), thickness=2)
        
        if draw_labels:
            text = f"{similarity:.2f}"
            text_position = (box[0], box[1] - 10)
            image = draw_text(image, text, text_position, color=(0, 0, 255))
    
    return image


def save_detections_to_json(detections, json_path):
    """保存检测结果到JSON文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"检测结果已保存: {json_path}")


def load_detections_from_json(json_path):
    """从JSON文件加载检测结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """调整图像大小"""
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def convert_to_gray(image):
    """转换为灰度图"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_image(image):
    """归一化图像到0-255范围"""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def show_image(image, window_name="Image"):
    """显示图像（调试用）"""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
