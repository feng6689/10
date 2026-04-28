import cv2
import numpy as np
from utils import convert_to_gray


class PanoramaStitcher:
    def __init__(self, detector='SIFT'):
        """
        初始化全景图拼接器
        :param detector: 特征检测器类型 'SIFT' 或 'ORB'
        """
        self.detector_type = detector.upper()
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
    
    def _create_detector(self):
        """创建特征检测器"""
        if self.detector_type == 'SIFT':
            try:
                return cv2.SIFT_create(nfeatures=5000)
            except AttributeError:
                print("警告: SIFT不可用，使用ORB代替")
                return cv2.ORB_create(nfeatures=5000)
        elif self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_FAST_SCORE)
        else:
            raise ValueError(f"不支持的检测器: {self.detector_type}")
    
    def _create_matcher(self):
        """创建特征匹配器"""
        if self.detector_type == 'SIFT':
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_features(self, image):
        """检测特征点和描述符"""
        gray = convert_to_gray(image)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio_thresh=0.75):
        """
        特征匹配，使用Lowe's ratio test
        """
        if desc1 is None or desc2 is None:
            return []
        
        if self.detector_type == 'SIFT':
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
        
        return good_matches
    
    def estimate_homography(self, kp1, kp2, matches, ransac_thresh=5.0):
        """
        估计单应性矩阵
        """
        if len(matches) < 4:
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        
        return H, mask
    
    def get_warped_size(self, H, h, w):
        """
        计算变换后的图像尺寸和偏移
        """
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        corners_warped = cv2.perspectiveTransform(corners, H)
        
        [x_min, y_min] = np.int32(corners_warped.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners_warped.max(axis=0).ravel() + 0.5)
        
        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])
        
        H_translated = translation.dot(H)
        output_size = (x_max - x_min, y_max - y_min)
        
        return H_translated, output_size, (-x_min, -y_min)
    
    def warp_image(self, image, H, output_size):
        """
        使用单应性矩阵变换图像
        """
        return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def create_mask(self, image, H, output_size, offset):
        """
        创建用于融合的mask
        """
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        mask[:, :5] = np.linspace(0, 1, 5).reshape(1, -1)
        mask[:, -5:] = np.linspace(1, 0, 5).reshape(1, -1)
        mask[:5, :] = np.minimum(mask[:5, :], np.linspace(0, 1, 5).reshape(-1, 1))
        mask[-5:, :] = np.minimum(mask[-5:, :], np.linspace(1, 0, 5).reshape(-1, 1))
        
        H_with_offset = np.array([[1, 0, offset[0]],
                                   [0, 1, offset[1]],
                                   [0, 0, 1]])
        
        mask_warped = cv2.warpPerspective(mask, H_with_offset.dot(np.linalg.inv(H_with_offset[:2, :2])), 
                                           output_size) if False else \
                       cv2.warpPerspective(mask, H_with_offset, output_size, 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return mask_warped
    
    def blend_images(self, img1, img2, mask1=None, mask2=None):
        """
        图像融合（线性混合）
        """
        if mask1 is None:
            mask1 = np.ones(img1.shape[:2], dtype=np.float32)
            mask1[np.all(img1 == [0, 0, 0], axis=2)] = 0
        
        if mask2 is None:
            mask2 = np.ones(img2.shape[:2], dtype=np.float32)
            mask2[np.all(img2 == [0, 0, 0], axis=2)] = 0
        
        mask_sum = mask1 + mask2
        mask_sum[mask_sum == 0] = 1
        
        weight1 = mask1 / mask_sum
        weight2 = mask2 / mask_sum
        
        weight1 = np.stack([weight1] * 3, axis=-1)
        weight2 = np.stack([weight2] * 3, axis=-1)
        
        blended = (img1.astype(np.float32) * weight1 + img2.astype(np.float32) * weight2).astype(np.uint8)
        
        return blended
    
    def stitch_pair(self, img_left, img_right):
        """
        拼接两张图片（左图和右图）
        """
        kp1, desc1 = self.detect_features(img_left)
        kp2, desc2 = self.detect_features(img_right)
        
        if desc1 is None or desc2 is None:
            print("警告: 无法检测到特征点")
            return None
        
        matches = self.match_features(desc1, desc2, ratio_thresh=0.7)
        print(f"找到 {len(matches)} 个匹配点")
        
        if len(matches) < 10:
            print("警告: 匹配点太少，尝试反向匹配...")
            matches = self.match_features(desc2, desc1, ratio_thresh=0.7)
            print(f"反向匹配找到 {len(matches)} 个匹配点")
            
            if len(matches) < 10:
                print("错误: 匹配点不足，无法拼接")
                return None
            
            H, mask = self.estimate_homography(kp2, kp1, matches, ransac_thresh=5.0)
            if H is None:
                print("错误: 无法估计单应性矩阵")
                return None
            
            h1, w1 = img_right.shape[:2]
            h2, w2 = img_left.shape[:2]
            
            H_translated, output_size, offset = self.get_warped_size(H, h1, w1)
            
            img_right_warped = self.warp_image(img_right, H_translated, output_size)
            
            result = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            result[offset[1]:offset[1]+h2, offset[0]:offset[0]+w2] = img_left
            
            mask_warped = np.all(img_right_warped != [0, 0, 0], axis=2).astype(np.float32)
            mask_result = np.all(result != [0, 0, 0], axis=2).astype(np.float32)
            
            final = self.blend_images(img_right_warped, result, mask_warped, mask_result)
            
        else:
            H, mask = self.estimate_homography(kp1, kp2, matches, ransac_thresh=5.0)
            if H is None:
                print("错误: 无法估计单应性矩阵")
                return None
            
            h1, w1 = img_left.shape[:2]
            h2, w2 = img_right.shape[:2]
            
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            
            corners1_warped = cv2.perspectiveTransform(corners1, H)
            all_corners = np.concatenate((corners2, corners1_warped), axis=0)
            
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            translation = np.array([[1, 0, -x_min],
                                    [0, 1, -y_min],
                                    [0, 0, 1]])
            
            H_translated = translation.dot(H)
            output_size = (x_max - x_min, y_max - y_min)
            
            img_left_warped = self.warp_image(img_left, H_translated, output_size)
            
            result = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            result[-y_min:-y_min+h2, -x_min:-x_min+w2] = img_right
            
            mask_warped = np.all(img_left_warped != [0, 0, 0], axis=2).astype(np.float32)
            mask_result = np.all(result != [0, 0, 0], axis=2).astype(np.float32)
            
            final = self.blend_images(img_left_warped, result, mask_warped, mask_result)
        
        return final
    
    def stitch_multiple(self, images, order='left_to_right'):
        """
        拼接多张图片
        :param images: 图片列表，按从左到右顺序
        :param order: 拼接顺序
        """
        if len(images) == 0:
            raise ValueError("图片列表为空")
        if len(images) == 1:
            return images[0]
        
        print(f"\n开始拼接 {len(images)} 张图片...")
        
        result = images[0]
        for i in range(1, len(images)):
            print(f"\n正在拼接第 {i+1} 张图片...")
            
            stitched = self.stitch_pair(result, images[i])
            
            if stitched is None:
                print(f"警告: 无法拼接第 {i+1} 张图片，尝试另一种方式...")
                
                stitched = self.stitch_pair(images[i], result)
                if stitched is None:
                    print(f"错误: 无法拼接第 {i+1} 张图片，跳过")
                    continue
            
            result = stitched
            print(f"拼接后尺寸: {result.shape[1]} x {result.shape[0]}")
        
        return result
    
    def remove_black_borders(self, image, threshold=10):
        """
        去除黑边
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return image[y:y+h, x:x+w]
        
        return image


def stitch_opencv_builtin(images):
    """
    使用OpenCV内置的Stitcher类进行拼接（备选方案）
    """
    print("尝试使用OpenCV内置Stitcher...")
    
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
    
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print("OpenCV Stitcher拼接成功!")
        return panorama
    else:
        print(f"OpenCV Stitcher失败，错误码: {status}")
        return None


def create_panorama(images, detector='SIFT', use_opencv_stitcher=False, remove_borders=True):
    """
    创建全景图
    :param images: 图片列表（从左到右顺序）
    :param detector: 特征检测器 'SIFT' 或 'ORB'
    :param use_opencv_stitcher: 是否使用OpenCV内置Stitcher
    :param remove_borders: 是否去除黑边
    :return: 拼接后的全景图
    """
    print("=" * 60)
    print("开始图片拼接")
    print("=" * 60)
    print(f"图片数量: {len(images)}")
    print(f"特征检测器: {detector}")
    
    if use_opencv_stitcher:
        result = stitch_opencv_builtin(images)
        if result is not None:
            if remove_borders:
                stitcher = PanoramaStitcher(detector)
                result = stitcher.remove_black_borders(result)
            return result
    
    stitcher = PanoramaStitcher(detector=detector)
    
    result = stitcher.stitch_multiple(images)
    
    if result is None:
        print("\n尝试使用OpenCV内置Stitcher作为备选...")
        result = stitch_opencv_builtin(images)
    
    if result is not None and remove_borders:
        print("\n去除黑边...")
        result = stitcher.remove_black_borders(result)
    
    return result
