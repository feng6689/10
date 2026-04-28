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
    
    def preprocess_image_multi_level(self, image):
        """
        多级预处理，使用多种方法提高检测率
        """
        gray = convert_to_gray(image)
        
        preprocessed_list = []
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges1 = cv2.Canny(blurred, 20, 80)
        preprocessed_list.append(('canny_low', edges1))
        
        edges2 = cv2.Canny(blurred, 50, 150)
        preprocessed_list.append(('canny_medium', edges2))
        
        edges3 = cv2.Canny(blurred, 100, 200)
        preprocessed_list.append(('canny_high', edges3))
        
        _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        edges4 = cv2.Canny(thresh1, 20, 80)
        preprocessed_list.append(('binary_canny', edges4))
        
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges5 = cv2.Canny(thresh2, 20, 80)
        preprocessed_list.append(('otsu_canny', edges5))
        
        adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        edges6 = cv2.Canny(adapt_thresh, 20, 80)
        preprocessed_list.append(('adaptive_canny', edges6))
        
        return preprocessed_list
    
    def find_contours_multi_level(self, image):
        """
        多级轮廓检测，使用多种预处理方法
        """
        all_contours = []
        h, w = image.shape[:2]
        
        preprocessed_list = self.preprocess_image_multi_level(image)
        
        for method_name, preprocessed in preprocessed_list:
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(preprocessed, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            contours, hierarchy = cv2.findContours(
                eroded, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if area < self.min_area or area > self.max_area:
                    continue
                
                x, y, cnt_w, cnt_h = cv2.boundingRect(cnt)
                
                if cnt_w < 8 or cnt_h < 8:
                    continue
                
                is_edge = (x <= 5 or y <= 5 or x + cnt_w >= w - 5 or y + cnt_h >= h - 5)
                
                contour_info = {
                    'contour': cnt,
                    'area': area,
                    'bbox': (x, y, cnt_w, cnt_h),
                    'is_edge': is_edge,
                    'method': method_name
                }
                
                is_duplicate = False
                for existing in all_contours:
                    iou = self._compute_bbox_iou(contour_info['bbox'], existing['bbox'])
                    if iou > 0.5:
                        is_duplicate = True
                        if contour_info['area'] > existing['area']:
                            existing.update(contour_info)
                        break
                
                if not is_duplicate:
                    all_contours.append(contour_info)
        
        print(f"  从 {len(preprocessed_list)} 种预处理方法中找到 {len(all_contours)} 个唯一轮廓")
        
        return all_contours
    
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
        
        sigma = 3.0
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
                                     shape_weight=0.6, color_weight=0.4):
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
            
            contour_info_list = self.find_contours_multi_level(image)
            
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
    
    def cluster_objects(self, objects, similarity_threshold=0.5):
        """
        对物体进行聚类，找出同类几何物体
        降低阈值以增加聚类数量
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
    
    def detect_objects_in_image(self, image, source_img_idx, reference_objects, similarity_threshold=0.4):
        """
        在单张图片中检测与参考物体相似的几何物体
        降低阈值以增加检测数量
        """
        detections = []
        
        contour_info_list = self.find_contours_multi_level(image)
        
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
                
                if info['is_edge']:
                    overall_sim = overall_sim * 0.95
                
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
    
    def _remove_duplicate_detections(self, detections, iou_threshold=0.4):
        """
        去除重复的检测结果
        降低IoU阈值以保留更多检测
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


def detect_similar_objects(images, similarity_threshold=0.4, min_area=50, max_area=100000):
    """
    检测所有图片中的同类几何物体
    降低参数以增加检测数量
    :param images: 原始图片列表
    :param similarity_threshold: 相似度阈值（降低以增加检测）
    :param min_area: 最小检测面积（降低以检测更小的物体）
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
    print(f"  聚类相似度阈值: 0.5")
    reference_cluster = detector.cluster_objects(all_objects, similarity_threshold=0.5)
    
    if len(reference_cluster) == 0:
        print("警告: 未找到同类几何物体（需要至少2个相似物体）")
        return [], [], []
    
    print(f"\n[步骤3] 以最大聚类（{len(reference_cluster)}个物体）为参考，检测所有图片...")
    print(f"  检测相似度阈值: {similarity_threshold} (值越低检测越多)")
    
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
