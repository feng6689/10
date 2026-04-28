import os
import sys

from utils import (
    load_images_from_paths,
    save_image,
    save_detections_to_json
)
from stitching import create_panorama
from detection import detect_similar_objects


def main():
    """
    主函数：
    1. 加载5张原图（1.png ~ 5.png，从左到右顺序）
    2. 检测每张原图中的同类几何物体（可能在边缘处不完整）
       - 用红色矩形框选
       - 标注0~1的相似度
       - 收集坐标、相似度、来源图片编号等信息存入1.json
    3. 使用SIFT/ORB特征匹配处理重叠区域，拼接成全景图panorama.jpg
    """
    print("=" * 60)
    print("图片全景拼接与几何物体检测系统")
    print("=" * 60)
    
    image_paths = [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = [os.path.join(base_dir, path) for path in image_paths]
    
    print("\n[步骤1] 加载原图...")
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"错误: 图片不存在: {path}")
            sys.exit(1)
        print(f"  图片 {i+1}: {os.path.basename(path)}")
    
    images = load_images_from_paths(image_paths)
    print(f"成功加载 {len(images)} 张图片")
    
    print("\n[步骤2] 检测原图中的同类几何物体...")
    print("  说明：")
    print("    - 从所有5张原图中提取所有几何物体")
    print("    - 聚类找出最常见的同类几何物体（最多的一类）")
    print("    - 在每张原图中检测这类物体（包括边缘处可能不完整的）")
    print("    - 用红色矩形框选，标注0~1的相似度")
    
    try:
        all_detections, annotated_images, reference_cluster = detect_similar_objects(
            images=images,
            similarity_threshold=0.3,
            min_area=50,
            max_area=100000
        )
        
    except Exception as e:
        print(f"错误: 检测过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        all_detections = []
        annotated_images = []
        reference_cluster = []
    
    print("\n[步骤3] 保存检测结果...")
    
    formatted_detections = []
    for i, det in enumerate(all_detections):
        formatted_det = {
            "id": i,
            "box": det["box"],
            "similarity": det["similarity"],
            "source_img": det["source_img"] + 1,
            "contour_area": det.get("contour_area", 0)
        }
        formatted_detections.append(formatted_det)
    
    json_path = os.path.join(base_dir, "1.json")
    save_detections_to_json(formatted_detections, json_path)
    
    print("\n[步骤4] 保存标注后的原图（带红色矩形框和相似度）...")
    for i, annotated in enumerate(annotated_images):
        annotated_path = os.path.join(base_dir, f"annotated_{i+1}.jpg")
        save_image(annotated_path, annotated)
        print(f"  已保存: annotated_{i+1}.jpg")
    
    print("\n[步骤5] 使用SIFT/ORB特征匹配拼接全景图...")
    print("  说明：")
    print("    - 检测5张图片之间的重叠区域")
    print("    - 使用SIFT/ORB特征点匹配")
    print("    - 估计单应性矩阵进行图像变换")
    print("    - 融合重叠区域，生成无缝全景图")
    print("    - 自动去除黑边")
    
    try:
        panorama = create_panorama(
            images=images,
            detector='SIFT',
            use_opencv_stitcher=False,
            remove_borders=True
        )
        
        if panorama is None:
            print("\n警告: 手动拼接失败，尝试使用OpenCV内置Stitcher...")
            panorama = create_panorama(
                images=images,
                detector='SIFT',
                use_opencv_stitcher=True,
                remove_borders=True
            )
        
        if panorama is None:
            print("错误: 图片拼接失败")
        else:
            panorama_path = os.path.join(base_dir, "panorama.jpg")
            save_image(panorama_path, panorama)
            print(f"  全景图已保存: panorama.jpg")
            print(f"  全景图尺寸: {panorama.shape[1]} x {panorama.shape[0]}")
        
    except Exception as e:
        print(f"错误: 拼接过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  1. panorama.jpg - 特征匹配拼接后的全景图")
    print(f"  2. 1.json - 检测结果数据（坐标、相似度、来源图片编号等）")
    print(f"  3. annotated_1.jpg ~ annotated_5.jpg - 标注后的原图")
    print(f"     (红色矩形框 + 0~1相似度标注)")
    
    print(f"\n检测统计:")
    print(f"  - 总检测数: {len(formatted_detections)} 个同类几何物体")
    
    source_counts = {}
    for det in formatted_detections:
        src = det["source_img"]
        source_counts[src] = source_counts.get(src, 0) + 1
    
    for src in range(1, 6):
        count = source_counts.get(src, 0)
        print(f"  - 图片 {src}: {count} 个目标")
    
    if formatted_detections:
        similarities = [d["similarity"] for d in formatted_detections]
        print(f"\n相似度统计 (0~1，越接近1越相似):")
        print(f"  - 最高相似度: {max(similarities):.4f}")
        print(f"  - 最低相似度: {min(similarities):.4f}")
        print(f"  - 平均相似度: {sum(similarities)/len(similarities):.4f}")
    
    print(f"\n1.json 数据格式:")
    print(f"  [")
    print(f"    {{")
    print(f"      'id': 0,                    // 检测编号")
    print(f"      'box': [x, y, w, h],       // 矩形框: 左上角x,y + 宽w + 高h")
    print(f"      'similarity': 0.85,        // 相似度 (0~1)")
    print(f"      'source_img': 1,           // 来源图片编号 (1~5)")
    print(f"      'contour_area': 1500.0     // 轮廓面积")
    print(f"    }}")
    print(f"  ]")
    
    print(f"\n参数说明 (可根据需要调整):")
    print(f"  - similarity_threshold: 0.3 (相似度阈值，越低检测越多)")
    print(f"  - min_area: 50 (最小检测面积)")
    print(f"  - max_area: 100000 (最大检测面积)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
