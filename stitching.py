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
                return cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.005, edgeThreshold=20)
            except AttributeError:
                print("警告: SIFT不可用，使用ORB代替")
                return cv2.ORB_create(nfeatures=20000)
        elif self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=20000, scoreType=cv2.ORB_FAST_SCORE)
        else:
            raise ValueError(f"不支持的检测器: {self.detector_type}")
    
    def _create_matcher(self):
        """创建特征匹配器"""
        if self.detector_type == 'SIFT':
            index_params = dict(algorithm=1, trees=10)
            search_params = dict(checks=200)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_features(self, image):
        """检测特征点和描述符"""
        gray = convert_to_gray(image)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio_thresh=0.8):
        """
        特征匹配，使用Lowe's ratio test，降低阈值以获得更多匹配
        """
        if desc1 is None or desc2 is None:
            return []
        
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        
        good_matches = []
        try:
            if self.detector_type == 'SIFT':
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_thresh * n.distance:
                            good_matches.append(m)
            else:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_thresh * n.distance:
                            good_matches.append(m)
        except Exception as e:
            print(f"特征匹配错误: {e}")
        
        return good_matches
    
    def estimate_homography(self, kp1, kp2, matches, ransac_thresh=3.0):
        """
        估计单应性矩阵，降低RANSAC阈值
        """
        if len(matches) < 4:
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        
        return H, mask
    
    def check_homography_validity(self, H, img1_shape, img2_shape):
        """
        检查单应性矩阵是否合理
        放宽条件，允许更多变换
        """
        if H is None:
            return False
        
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]
        
        corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        
        try:
            corners_warped = cv2.perspectiveTransform(corners, H)
        except:
            return False
        
        x_coords = corners_warped[:, 0, 0]
        y_coords = corners_warped[:, 0, 1]
        
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        
        if width < 0.1 * w1 or width > 10 * w1:
            return False
        if height < 0.1 * h1 or height > 10 * h1:
            return False
        
        return True
    
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
    
    def create_alpha_mask(self, shape, blend_width=150):
        """
        创建alpha遮罩用于图像融合，增加融合宽度
        """
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        for i in range(blend_width):
            alpha = i / blend_width
            mask[:, i] = alpha
            mask[:, w - 1 - i] = alpha
        
        for i in range(blend_width):
            alpha = i / blend_width
            mask[i, :] = np.minimum(mask[i, :], alpha)
            mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
        
        return mask
    
    def blend_images_with_overlap(self, img1, img2, mask1=None, mask2=None):
        """
        带重叠区域的图像融合
        """
        if mask1 is None:
            mask1 = np.ones(img1.shape[:2], dtype=np.float32)
            mask1[np.all(img1 == [0, 0, 0], axis=2)] = 0
        
        if mask2 is None:
            mask2 = np.ones(img2.shape[:2], dtype=np.float32)
            mask2[np.all(img2 == [0, 0, 0], axis=2)] = 0
        
        overlap = (mask1 > 0) & (mask2 > 0)
        only1 = (mask1 > 0) & (mask2 == 0)
        only2 = (mask1 == 0) & (mask2 > 0)
        
        blended = np.zeros_like(img1, dtype=np.uint8)
        
        blended[only1] = img1[only1]
        blended[only2] = img2[only2]
        
        if np.any(overlap):
            weight1 = mask1[overlap]
            weight2 = mask2[overlap]
            weight_sum = weight1 + weight2
            weight_sum[weight_sum == 0] = 1
            
            weight1 = weight1 / weight_sum
            weight2 = weight2 / weight_sum
            
            weight1_3d = np.stack([weight1] * 3, axis=-1)
            weight2_3d = np.stack([weight2] * 3, axis=-1)
            
            blended[overlap] = (img1[overlap].astype(np.float32) * weight1_3d + 
                               img2[overlap].astype(np.float32) * weight2_3d).astype(np.uint8)
        
        return blended
    
    def stitch_left_to_right_improved(self, img_left, img_right):
        """
        改进的从左到右拼接
        尝试多种匹配策略
        """
        print(f"  尝试左→右拼接 (左图: {img_left.shape[1]}x{img_left.shape[0]}, 右图: {img_right.shape[1]}x{img_right.shape[0]})")
        
        kp_left, desc_left = self.detect_features(img_left)
        kp_right, desc_right = self.detect_features(img_right)
        
        if desc_left is None or desc_right is None:
            print("  警告: 无法检测到特征点，使用智能直接拼接")
            return self.stitch_smart_direct(img_left, img_right)
        
        print(f"  左图特征点: {len(kp_left)}, 右图特征点: {len(kp_right)}")
        
        matches_forward = self.match_features(desc_right, desc_left, ratio_thresh=0.8)
        print(f"  正向匹配点 (右→左): {len(matches_forward)}")
        
        if len(matches_forward) >= 4:
            H, mask = self.estimate_homography(kp_right, kp_left, matches_forward, ransac_thresh=3.0)
            
            if H is not None and self.check_homography_validity(H, img_right.shape, img_left.shape):
                inlier_count = np.sum(mask) if mask is not None else 0
                print(f"  估计单应性矩阵成功，内点数量: {inlier_count}")
                
                if inlier_count >= 8:
                    return self._warp_and_blend(img_left, img_right, H)
        
        print("  特征匹配不够好，尝试反向匹配...")
        matches_backward = self.match_features(desc_left, desc_right, ratio_thresh=0.8)
        print(f"  反向匹配点 (左→右): {len(matches_backward)}")
        
        if len(matches_backward) >= 4:
            H, mask = self.estimate_homography(kp_left, kp_right, matches_backward, ransac_thresh=3.0)
            
            if H is not None:
                try:
                    H_inv = np.linalg.inv(H)
                    if self.check_homography_validity(H_inv, img_right.shape, img_left.shape):
                        inlier_count = np.sum(mask) if mask is not None else 0
                        print(f"  反向单应性矩阵成功，内点数量: {inlier_count}")
                        
                        if inlier_count >= 8:
                            return self._warp_and_blend(img_left, img_right, H_inv)
                except:
                    pass
        
        print("  特征匹配不够好，使用智能直接拼接")
        return self.stitch_smart_direct(img_left, img_right)
    
    def _warp_and_blend(self, img_left, img_right, H_right_to_left):
        """
        使用单应性矩阵变换并融合
        """
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        
        corners_right = np.float32([[0, 0], [0, h_right], [w_right, h_right], [w_right, 0]]).reshape(-1, 1, 2)
        corners_left = np.float32([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]]).reshape(-1, 1, 2)
        
        corners_right_warped = cv2.perspectiveTransform(corners_right, H_right_to_left)
        all_corners = np.concatenate((corners_left, corners_right_warped), axis=0)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])
        
        H_translated = translation.dot(H_right_to_left)
        output_size = (x_max - x_min, y_max - y_min)
        
        img_right_warped = self.warp_image(img_right, H_translated, output_size)
        
        result = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        result[-y_min:-y_min+h_left, -x_min:-x_min+w_left] = img_left
        
        alpha_mask_right = self.create_alpha_mask(img_right.shape, blend_width=150)
        alpha_mask_warped = cv2.warpPerspective(alpha_mask_right, H_translated, output_size)
        
        alpha_mask_result = np.zeros(result.shape[:2], dtype=np.float32)
        alpha_mask_result[-y_min:-y_min+h_left, -x_min:-x_min+w_left] = 1.0
        
        final = self.blend_images_with_overlap(img_right_warped, result, alpha_mask_warped, alpha_mask_result)
        
        print(f"  拼接成功，输出尺寸: {output_size[0]}x{output_size[1]}")
        
        return final
    
    def stitch_smart_direct(self, img_left, img_right):
        """
        智能直接拼接
        尝试估计重叠区域
        """
        print("  使用智能直接拼接")
        
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        
        h_max = max(h_left, h_right)
        
        if h_left != h_max:
            scale = h_max / h_left
            new_w = int(w_left * scale)
            img_left = cv2.resize(img_left, (new_w, h_max))
        
        if h_right != h_max:
            scale = h_max / h_right
            new_w = int(w_right * scale)
            img_right = cv2.resize(img_right, (new_w, h_max))
        
        w_left_resized = img_left.shape[1]
        w_right_resized = img_right.shape[1]
        
        overlap_estimates = [0.1, 0.15, 0.2, 0.25, 0.3]
        best_result = None
        best_score = -1
        
        for overlap_ratio in overlap_estimates:
            result = self._stitch_with_overlap(img_left, img_right, overlap_ratio)
            
            if result is not None:
                score = self._evaluate_stitch_quality(img_left, img_right, result, overlap_ratio)
                if score > best_score:
                    best_score = score
                    best_result = result
        
        if best_result is not None:
            print(f"  智能直接拼接完成，输出尺寸: {best_result.shape[1]}x{best_result.shape[0]}")
            return best_result
        
        result = np.hstack([img_left, img_right])
        print(f"  简单拼接完成，输出尺寸: {result.shape[1]}x{result.shape[0]}")
        return result
    
    def _stitch_with_overlap(self, img_left, img_right, overlap_ratio):
        """
        使用指定的重叠比例拼接
        """
        w_left = img_left.shape[1]
        w_right = img_right.shape[1]
        
        overlap_pixels = int(min(w_left, w_right) * overlap_ratio)
        
        if overlap_pixels <= 0 or w_left <= overlap_pixels or w_right <= overlap_pixels:
            return None
        
        left_part = img_left[:, :-overlap_pixels]
        right_part = img_right[:, overlap_pixels:]
        overlap_left = img_left[:, -overlap_pixels:]
        overlap_right = img_right[:, :overlap_pixels]
        
        blend_mask = np.linspace(1, 0, overlap_pixels).reshape(1, -1, 1)
        blended_overlap = (overlap_left.astype(np.float32) * blend_mask + 
                         overlap_right.astype(np.float32) * (1 - blend_mask)).astype(np.uint8)
        
        result = np.hstack([left_part, blended_overlap, right_part])
        return result
    
    def _evaluate_stitch_quality(self, img_left, img_right, result, overlap_ratio):
        """
        评估拼接质量
        """
        h, w_left = img_left.shape[:2]
        overlap_pixels = int(min(w_left, img_right.shape[1]) * overlap_ratio)
        
        if overlap_pixels <= 0:
            return 0
        
        gray_left = convert_to_gray(img_left)
        gray_right = convert_to_gray(img_right)
        
        overlap_left_region = gray_left[:, -overlap_pixels:]
        overlap_right_region = gray_right[:, :overlap_pixels]
        
        if overlap_left_region.shape != overlap_right_region.shape:
            return 0
        
        similarity = cv2.matchTemplate(overlap_left_region, overlap_right_region, cv2.TM_CCOEFF_NORMED)
        max_sim = np.max(similarity) if similarity.size > 0 else 0
        
        return max_sim
    
    def stitch_multiple_left_to_right(self, images):
        """
        严格从左到右拼接多张图片
        :param images: 图片列表，按从左到右顺序
        """
        if len(images) == 0:
            raise ValueError("图片列表为空")
        if len(images) == 1:
            return images[0]
        
        print(f"\n开始从左到右拼接 {len(images)} 张图片...")
        print(f"拼接顺序: 图片1 → 图片2 → 图片3 → 图片4 → 图片5")
        
        result = images[0]
        for i in range(1, len(images)):
            print(f"\n正在拼接第 {i+1} 张图片到结果中...")
            
            stitched = self.stitch_left_to_right_improved(result, images[i])
            
            if stitched is None:
                print(f"警告: 无法拼接第 {i+1} 张图片，使用简单拼接")
                stitched = np.hstack([result, images[i]])
            
            result = stitched
            print(f"当前结果尺寸: {result.shape[1]} x {result.shape[0]}")
        
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
    
    try:
        stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
        
        status, panorama = stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            print("OpenCV Stitcher拼接成功!")
            return panorama
        else:
            print(f"OpenCV Stitcher失败，错误码: {status}")
            return None
    except Exception as e:
        print(f"OpenCV Stitcher异常: {e}")
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
    print("开始图片拼接（从左到右顺序）")
    print("=" * 60)
    print(f"图片数量: {len(images)}")
    print(f"特征检测器: {detector}")
    print(f"图片顺序: 1.png → 2.png → 3.png → 4.png → 5.png (从左到右)")
    
    if use_opencv_stitcher:
        result = stitch_opencv_builtin(images)
        if result is not None:
            if remove_borders:
                stitcher = PanoramaStitcher(detector)
                result = stitcher.remove_black_borders(result)
            return result
    
    stitcher = PanoramaStitcher(detector=detector)
    
    result = stitcher.stitch_multiple_left_to_right(images)
    
    if result is None:
        print("\n尝试使用OpenCV内置Stitcher作为备选...")
        result = stitch_opencv_builtin(images)
    
    if result is not None and remove_borders:
        print("\n去除黑边...")
        result = stitcher.remove_black_borders(result)
    
    return result
