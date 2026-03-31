from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops


def build_feature_dataset(data_dir, extractor):
    X = []
    y = []
    class_names = sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])

    label_map = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_path = Path(data_dir) / cls

        for img_path in cls_path.glob("*"):
            feat = extractor.extract(img_path)
            X.append(feat)
            y.append(label_map[cls])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y, class_names


class FeatureExtractor:

    def __init__(self, img_size=(128, 128), feature_set=None):
        self.img_size = img_size
        self.feature_set = feature_set or ["hog", "lbp", "hsv"]

        self.lbp_P = 8
        self.lbp_R = 1

        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)

        self.glcm_distances = [1]
        self.glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.glcm_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    def hog_feature(self, gray):
        hog_feat = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True
        )
        return hog_feat.astype(np.float32)

    def lbp_feature(self, gray):
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_P,
            R=self.lbp_R,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.lbp_P + 3),
            range=(0, self.lbp_P + 2)
        )

        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)

        return hist

    def hsv_histogram(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def glcm_feature(self, gray):
        gray_quantized = (gray // 32).astype(np.uint8)

        glcm = graycomatrix(
            gray_quantized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=8,
            symmetric=True,
            normed=True
        )

        feats = []
        for prop in self.glcm_props:
            vals = graycoprops(glcm, prop).flatten()
            feats.extend(vals)

        return np.array(feats, dtype=np.float32)

    def extract(self, img_path):
        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img, gray = self.preprocess(img)

        feature_parts = []

        for feature_name in self.feature_set:
            if feature_name == "hog":
                feature_parts.append(self.hog_feature(gray))
            elif feature_name == "lbp":
                feature_parts.append(self.lbp_feature(gray))
            elif feature_name == "hsv":
                feature_parts.append(self.hsv_histogram(img))
            elif feature_name == "glcm":
                feature_parts.append(self.glcm_feature(gray))
            else:
                raise ValueError(f"Unsupported feature name: {feature_name}")

        return np.concatenate(feature_parts).astype(np.float32)

    def get_feature_config(self):
        return {
            "img_size": self.img_size,
            "feature_set": self.feature_set,
            "lbp_P": self.lbp_P,
            "lbp_R": self.lbp_R,
            "hog_orientations": self.hog_orientations,
            "hog_pixels_per_cell": self.hog_pixels_per_cell,
            "hog_cells_per_block": self.hog_cells_per_block,
            "glcm_distances": self.glcm_distances,
            "glcm_angles": self.glcm_angles,
            "glcm_props": self.glcm_props
        }