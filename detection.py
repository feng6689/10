import cv2
import numpy as np
from utils import convert_to_gray


class ImprovedGeometricDetector:
    def __init__(self, min_area=50, max_area=100000):
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
    
    def preprocess_multi_level(self, image):
        """
        多级预处理，提高检测率
        返回多种预处理结果
        """
        gray = convert_to_gray(image)
        
        results = []
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges1 = cv2.Canny(blurred, 30, 100)
        results.append(edges1)
        
        edges2 = cv2.Canny(blurred, 50, 150)
        results.append(edges2)
        
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        edges3 = cv2.Canny(thresh1, 30, 100)
        results.append(edges3)
        
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges4 = cv2.Canny(thresh2, 30, 100)
        results.append(edges4)
        
        adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        edges5 = cv2.Canny(adapt_thresh, 30, 100)
        results.append(edges5)
        
        return results
    
    def find_all_contours(self, image):
        """
        使用多种预处理方法查找所有轮廓
        """
        all_contours = []
        all_heirarchies = []
        
        preprocessed_list = self.preprocess_multi_level(image)
        
        for preprocessed in preprocessed_list:
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(preprocessed, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            contours, hierarchy = cv2.findContours(
                eroded, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > self.min_area:
                    is_duplicate = False
                    for existing_cnt in all_contours:
                        if self._are_contours_similar(cnt, existing_cnt):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        all_contours.append(cnt)
        
        return all_contours
    
    def _are_contours_similar(self, cnt1, cnt2, iou_threshold=0.7):
        """检查两个轮廓是否相似（基于IoU）"""
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi
        
        if wi <= 0 or hi <= 0:
            return False
        
        inter_area = wi * hi
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou > iou_threshold
    
    def get_shape_features(self, contour):
        """
        提取多种形状特征
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
            (x_ell, y_ell), (MA, ma), angle = cv2.fitEllipse(contour)
            eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) > 0 else 0
        else:
            eccentricity = 0.5
        
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        features.extend([extent, circularity, eccentricity, aspect_ratio])
        
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
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else image
        if len(hsv.shape) == 3:
            mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        else:
            mean_hsv = (0, 0, 0)
        
        return np.array(list(mean_color) + list(std_color) + list(mean_hsv))
    
    def get_sift_features(self, image, contour):
        """
        提取SIFT特征
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return np.zeros(128)
        
        gray_roi = convert_to_gray(roi)
        
        kp, desc = self.sift.detectAndCompute(gray_roi, None)
        
        if desc is None or len(desc) == 0:
            return np.zeros(128)
        
        return np.mean(desc, axis=0)
    
    def get_bounding_box(self, contour):
        """获取边界框"""
        x, y, w, h = cv2.boundingRect(contour)
        return (int(x), int(y), int(w), int(h))
    
    def compute_shape_similarity(self, feat1, feat2):
        """
        计算形状相似度（归一化到0~1）
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        if len(feat1) != len(feat2):
            return 0.0
        
        distance = np.linalg.norm(feat1 - feat2)
        
        sigma = 5.0
        similarity = np.exp(-distance ** 2 / (2 * sigma ** 2))
        
        return max(0.0, min(1.0, similarity))
    
    def compute_color_similarity(self, color1, color2):
        """
        计算颜色相似度
        """
        if color1 is None or color2 is None:
            return 0.5
        
        distance = np.linalg.norm(color1 - color2)
        max_distance = np.sqrt(3 * (255 ** 2) + 3 * (128 ** 2) + 3 * (255 ** 2))
        
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def compute_sift_similarity(self, desc1, desc2):
        """
        计算SIFT描述符相似度
        """
        if desc1 is None or desc2 is None:
            return 0.5
        
        if np.all(desc1 == 0) or np.all(desc2 == 0):
            return 0.5
        
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.5
        
        cosine = np.dot(desc1, desc2) / (norm1 * norm2)
        
        similarity = (cosine + 1) / 2
        
        return max(0.0, min(1.0, similarity))
    
    def compute_overall_similarity(self, shape_sim, color_sim, sift_sim,
                                     shape_weight=0.5, color_weight=0.3, sift_weight=0.2):
        """
        计算综合相似度
        """
        total_weight = shape_weight + color_weight + sift_weight
        similarity = (shape_sim * shape_weight + color_sim * color_weight + sift_sim * sift_weight) / total_weight
        
        return max(0.0, min(1.0, similarity))
    
    def extract_objects_from_images(self, images):
        """
        从所有图片中提取所有几何物体
        """
        all_objects = []
        
        for img_idx, image in enumerate(images):
            contours = self.find_all_contours(image)
            
            for cnt_idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                
                if area < self.min_area or area > self.max_area:
                    continue
                
                shape_features = self.get_shape_features(cnt)
                color_features = self.get_color_features(image, cnt)
                sift_features = self.get_sift_features(image, cnt)
                bbox = self.get_bounding_box(cnt)
                
                obj = {
                    'shape_features': shape_features,
                    'color_features': color_features,
                    'sift_features': sift_features,
                    'bbox': bbox,
                    'area': area,
                    'source_img': img_idx,
                    'contour': cnt
                }
                all_objects.append(obj)
        
        print(f"从 {len(images)} 张图片中提取了 {len(all_objects)} 个几何物体")
        return all_objects
    
    def cluster_objects(self, objects, similarity_threshold=0.6):
        """
        对物体进行聚类，找出同类几何物体
        """
        if len(objects) == 0:
            return []
        
        clusters = []
        
        for i, obj in enumerate(objects):
            matched = False
            
            for cluster in clusters:
                ref_obj = cluster[0]
                
                shape_sim = self.compute_shape_similarity(
                    obj['shape_features'], 
                    ref_obj['shape_features']
                )
                color_sim = self.compute_color_similarity(
                    obj['color_features'], 
                    ref_obj['color_features']
                )
                sift_sim = self.compute_sift_similarity(
                    obj['sift_features'], 
                    ref_obj['sift_features']
                )
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim, sift_sim)
                
                if overall_sim >= similarity_threshold:
                    cluster.append(obj)
                    matched = True
                    break
            
            if not matched:
                clusters.append([obj])
        
        clusters.sort(key=lambda x: len(x), reverse=True)
        
        print(f"\n聚类结果:")
        print(f"  总聚类数: {len(clusters)}")
        for i, cluster in enumerate(clusters[:5]):
            print(f"  聚类 {i+1}: {len(cluster)} 个物体")
        
        if clusters:
            return clusters[0]
        return []
    
    def detect_objects_in_image(self, image, source_img_idx, reference_objects, similarity_threshold=0.3):
        """
        在单张图片中检测与参考物体相似的几何物体
        包括检测边缘处可能不完整的物体
        """
        detections = []
        
        contours = self.find_all_contours(image)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            shape_features = self.get_shape_features(cnt)
            color_features = self.get_color_features(image, cnt)
            sift_features = self.get_sift_features(image, cnt)
            bbox = self.get_bounding_box(cnt)
            
            max_similarity = 0.0
            best_match_idx = -1
            
            for ref_idx, ref_obj in enumerate(reference_objects):
                shape_sim = self.compute_shape_similarity(shape_features, ref_obj['shape_features'])
                color_sim = self.compute_color_similarity(color_features, ref_obj['color_features'])
                sift_sim = self.compute_sift_similarity(sift_features, ref_obj['sift_features'])
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim, sift_sim)
                
                if overall_sim > max_similarity:
                    max_similarity = overall_sim
                    best_match_idx = ref_idx
            
            if max_similarity >= similarity_threshold:
                detection = {
                    'box': bbox,
                    'similarity': float(max_similarity),
                    'source_img': int(source_img_idx),
                    'contour_area': float(area),
                    'best_match_ref': best_match_idx
                }
                detections.append(detection)
        
        detections = self._remove_duplicate_detections(detections)
        
        return detections
    
    def _remove_duplicate_detections(self, detections, iou_threshold=0.5):
        """
        去除重复的检测结果（非极大值抑制）
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
                iou = self._compute_iou(best['box'], det['box'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _compute_iou(self, box1, box2):
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
    
    def annotate_image(self, image, detections):
        """
        在图像上标注检测结果
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


def detect_similar_objects(images, similarity_threshold=0.3, min_area=50, max_area=100000):
    """
    检测所有图片中的同类几何物体
    :param images: 原始图片列表
    :param similarity_threshold: 相似度阈值（降低阈值以检测更多物体）
    :param min_area: 最小检测面积
    :param max_area: 最大检测面积
    :return: (所有检测结果列表, 标注后的图片列表, 参考物体列表)
    """
    detector = ImprovedGeometricDetector(min_area=min_area, max_area=max_area)
    
    print("\n" + "=" * 60)
    print("几何物体检测")
    print("=" * 60)
    
    print("\n[步骤1] 从所有图片中提取几何物体...")
    all_objects = detector.extract_objects_from_images(images)
    
    if len(all_objects) == 0:
        print("警告: 未检测到任何几何物体")
        return [], [], []
    
    print("\n[步骤2] 聚类找出同类几何物体...")
    reference_cluster = detector.cluster_objects(all_objects, similarity_threshold=0.5)
    
    if len(reference_cluster) == 0:
        print("警告: 未找到同类几何物体")
        return [], [], []
    
    print(f"\n[步骤3] 以最大聚类（{len(reference_cluster)}个物体）为参考，检测所有图片...")
    print(f"       相似度阈值: {similarity_threshold} (值越低检测越多)")
    
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
