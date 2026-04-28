import cv2
import numpy as np
from utils import convert_to_gray


class ImprovedGeometricDetector:
    def __init__(self, min_area=100, max_area=50000):
        """
        初始化改进的几何物体检测器
        :param min_area: 最小检测面积
        :param max_area: 最大检测面积
        """
        self.min_area = min_area
        self.max_area = max_area
        self.sift = self._create_sift()
    
    def _create_sift(self):
        """创建SIFT检测器"""
        try:
            return cv2.SIFT_create(nfeatures=2000)
        except AttributeError:
            return cv2.ORB_create(nfeatures=2000)
    
    def preprocess_image(self, image):
        """
        预处理图像，使用更稳定的方法
        """
        gray = convert_to_gray(image)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def find_contours(self, image):
        """
        查找轮廓，使用更稳定的方法
        """
        edges = self.preprocess_image(image)
        
        contours, hierarchy = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        valid_contours = []
        h, w = image.shape[:2]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, cnt_w, cnt_h = cv2.boundingRect(cnt)
            
            if cnt_w < 10 or cnt_h < 10:
                continue
            
            is_edge = (x <= 2 or y <= 2 or x + cnt_w >= w - 2 or y + cnt_h >= h - 2)
            
            valid_contours.append({
                'contour': cnt,
                'area': area,
                'bbox': (x, y, cnt_w, cnt_h),
                'is_edge': is_edge
            })
        
        valid_contours = self._remove_duplicate_contours(valid_contours)
        
        return valid_contours
    
    def _remove_duplicate_contours(self, contours, iou_threshold=0.6):
        """
        去除重复的轮廓
        """
        if len(contours) <= 1:
            return contours
        
        sorted_contours = sorted(contours, key=lambda x: x['area'], reverse=True)
        
        keep = []
        while sorted_contours:
            current = sorted_contours.pop(0)
            keep.append(current)
            
            remaining = []
            for cnt in sorted_contours:
                iou = self._compute_bbox_iou(current['bbox'], cnt['bbox'])
                if iou < iou_threshold:
                    remaining.append(cnt)
            sorted_contours = remaining
        
        return keep
    
    def _compute_bbox_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi
        
        if wi <= 0 or hi <= 0:
            return 0.0
        
        inter_area = wi * hi
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_shape_features(self, contour):
        """
        提取形状特征
        """
        features = []
        
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments)
        
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if len(contour) >= 5:
            try:
                (x_ell, y_ell), (MA, ma), angle = cv2.fitEllipse(contour)
                eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) > 0 else 0
            except:
                eccentricity = 0.5
        else:
            eccentricity = 0.5
        
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        features.extend([extent, circularity, eccentricity, aspect_ratio, solidity])
        
        return np.array(features)
    
    def get_color_features(self, image, contour):
        """
        提取颜色特征
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        if len(image.shape) == 3:
            mean_color = cv2.mean(image, mask=mask)[:3]
            std_color = cv2.meanStdDev(image, mask=mask)[1].flatten()[:3]
        else:
            mean_val = cv2.mean(image, mask=mask)[0]
            mean_color = (mean_val, mean_val, mean_val)
            std_color = (0, 0, 0)
        
        return np.array(list(mean_color) + list(std_color))
    
    def compute_shape_similarity(self, feat1, feat2):
        """
        计算形状相似度
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        if len(feat1) != len(feat2):
            return 0.0
        
        distance = np.linalg.norm(feat1 - feat2)
        
        sigma = 2.0
        similarity = np.exp(-distance ** 2 / (2 * sigma ** 2))
        
        return max(0.0, min(1.0, similarity))
    
    def compute_color_similarity(self, color1, color2):
        """
        计算颜色相似度
        """
        if color1 is None or color2 is None:
            return 0.5
        
        distance = np.linalg.norm(color1 - color2)
        max_distance = np.sqrt(3 * (255 ** 2) + 3 * (128 ** 2))
        
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def compute_overall_similarity(self, shape_sim, color_sim,
                                     shape_weight=0.7, color_weight=0.3):
        """
        计算综合相似度
        """
        total_weight = shape_weight + color_weight
        similarity = (shape_sim * shape_weight + color_sim * color_weight) / total_weight
        
        return max(0.0, min(1.0, similarity))
    
    def extract_objects_from_images(self, images):
        """
        从所有图片中提取所有几何物体
        """
        all_objects = []
        
        for img_idx, image in enumerate(images):
            print(f"\n  处理图片 {img_idx + 1}...")
            
            contour_info_list = self.find_contours(image)
            
            print(f"  找到 {len(contour_info_list)} 个有效轮廓")
            
            for info in contour_info_list:
                cnt = info['contour']
                area = info['area']
                
                shape_features = self.get_shape_features(cnt)
                color_features = self.get_color_features(image, cnt)
                
                obj = {
                    'shape_features': shape_features,
                    'color_features': color_features,
                    'bbox': info['bbox'],
                    'area': area,
                    'source_img': img_idx,
                    'contour': cnt,
                    'is_edge': info['is_edge']
                }
                all_objects.append(obj)
        
        print(f"\n从 {len(images)} 张图片中提取了 {len(all_objects)} 个几何物体")
        return all_objects
    
    def cluster_objects(self, objects, similarity_threshold=0.7):
        """
        对物体进行聚类，找出同类几何物体
        """
        if len(objects) == 0:
            return []
        
        clusters = []
        
        for i, obj in enumerate(objects):
            matched = False
            
            for cluster in clusters:
                avg_shape = np.mean([o['shape_features'] for o in cluster], axis=0)
                avg_color = np.mean([o['color_features'] for o in cluster], axis=0)
                
                shape_sim = self.compute_shape_similarity(obj['shape_features'], avg_shape)
                color_sim = self.compute_color_similarity(obj['color_features'], avg_color)
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim)
                
                if overall_sim >= similarity_threshold:
                    cluster.append(obj)
                    matched = True
                    break
            
            if not matched:
                clusters.append([obj])
        
        clusters = [c for c in clusters if len(c) >= 2]
        clusters.sort(key=lambda x: len(x), reverse=True)
        
        print(f"\n聚类结果:")
        print(f"  有效聚类数 (>=2个物体): {len(clusters)}")
        for i, cluster in enumerate(clusters[:5]):
            print(f"  聚类 {i+1}: {len(cluster)} 个物体")
        
        if clusters:
            return clusters[0]
        return []
    
    def detect_objects_in_image(self, image, source_img_idx, reference_objects, similarity_threshold=0.6):
        """
        在单张图片中检测与参考物体相似的几何物体
        """
        detections = []
        
        contour_info_list = self.find_contours(image)
        
        for info in contour_info_list:
            cnt = info['contour']
            area = info['area']
            
            shape_features = self.get_shape_features(cnt)
            color_features = self.get_color_features(image, cnt)
            bbox = info['bbox']
            
            max_similarity = 0.0
            best_match_idx = -1
            
            for ref_idx, ref_obj in enumerate(reference_objects):
                shape_sim = self.compute_shape_similarity(shape_features, ref_obj['shape_features'])
                color_sim = self.compute_color_similarity(color_features, ref_obj['color_features'])
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim)
                
                if overall_sim > max_similarity:
                    max_similarity = overall_sim
                    best_match_idx = ref_idx
            
            if max_similarity >= similarity_threshold:
                detection = {
                    'box': bbox,
                    'similarity': float(max_similarity),
                    'source_img': int(source_img_idx),
                    'contour_area': float(area),
                    'best_match_ref': best_match_idx,
                    'is_edge': info['is_edge']
                }
                detections.append(detection)
        
        detections = self._remove_duplicate_detections(detections)
        
        return detections
    
    def _remove_duplicate_detections(self, detections, iou_threshold=0.5):
        """
        去除重复的检测结果
        """
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x['similarity'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            remaining = []
            for det in detections:
                iou = self._compute_bbox_iou(best['box'], det['box'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def annotate_image(self, image, detections):
        """
        在图像上标注检测结果（仅使用英文，避免乱码）
        """
        annotated = image.copy()
        
        for det in detections:
            box = det['box']
            similarity = det['similarity']
            
            x, y, w, h = box
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            text = f"{similarity:.2f}"
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(annotated, text, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated


def detect_similar_objects(images, similarity_threshold=0.6, min_area=100, max_area=50000):
    """
    检测所有图片中的同类几何物体
    :param images: 原始图片列表
    :param similarity_threshold: 相似度阈值（提高以减少误检）
    :param min_area: 最小检测面积
    :param max_area: 最大检测面积
    :return: (所有检测结果列表, 标注后的图片列表, 参考物体列表)
    """
    detector = ImprovedGeometricDetector(min_area=min_area, max_area=max_area)
    
    print("\n" + "=" * 60)
    print("几何物体检测")
    print("=" * 60)
    
    print("\n[步骤1] 从所有图片中提取几何物体...")
    print(f"  参数: min_area={min_area}, max_area={max_area}")
    all_objects = detector.extract_objects_from_images(images)
    
    if len(all_objects) == 0:
        print("警告: 未检测到任何几何物体")
        return [], [], []
    
    print("\n[步骤2] 聚类找出同类几何物体...")
    print(f"  聚类相似度阈值: 0.7")
    reference_cluster = detector.cluster_objects(all_objects, similarity_threshold=0.7)
    
    if len(reference_cluster) == 0:
        print("警告: 未找到同类几何物体（需要至少2个相似物体）")
        return [], [], []
    
    print(f"\n[步骤3] 以最大聚类（{len(reference_cluster)}个物体）为参考，检测所有图片...")
    print(f"  检测相似度阈值: {similarity_threshold} (值越高检测越严格)")
    
    all_detections = []
    annotated_images = []
    
    for img_idx, image in enumerate(images):
        detections = detector.detect_objects_in_image(
            image, 
            img_idx, 
            reference_cluster, 
            similarity_threshold
        )
        
        annotated = detector.annotate_image(image, detections)
        annotated_images.append(annotated)
        
        all_detections.extend(detections)
        print(f"  图片 {img_idx+1}: 检测到 {len(detections)} 个同类物体")
    
    return all_detections, annotated_images, reference_cluster
