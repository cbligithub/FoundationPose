import json
import os
import imageio
import glob
import cv2
import numpy as np

class CustomLoader:

    def __init__(self, data_path="demo_data/charger"):
        self.data_path = data_path
        self.color_files = sorted(glob.glob(os.path.join(data_path, "*_rgb.png")))
        self.depth_files = sorted(glob.glob(os.path.join(data_path, "*_depth2rgb.png")))
        self.mask_files = sorted(glob.glob(os.path.join(data_path, "*_mask.png")))
        self.info_files = sorted(glob.glob(os.path.join(data_path, "*_rgb_camera_info.json")))
        self.downscale = 0.5
        self.id_strs = []
        for i in range(len(self.color_files)):
            name = os.path.basename(self.color_files[i]).split('.')[0]
            self.id_strs.append(name)

    def get_color(self, index):
        img = cv2.imread(self.color_files[index], cv2.IMREAD_UNCHANGED)
        H, W = img.shape[:2]
        H = int(H*self.downscale)
        W = int(W*self.downscale)
        color = img[..., :3]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_depth(self, index):
        img = cv2.imread(self.depth_files[index], cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float16)/1000.0  # convert mm to meter
        # img = img.astype(np.float32)/1000.0  # convert mm to meter
        H, W = img.shape[:2]
        H = int(H*self.downscale)
        W = int(W*self.downscale)
        depth = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.001) | (depth>=np.inf)] = 0
        return depth

    def get_mask(self, index):
        img = cv2.imread(self.mask_files[index], cv2.IMREAD_UNCHANGED)
        H, W = img.shape[:2]
        H = int(H*self.downscale)
        W = int(W*self.downscale)
        mask = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        return mask
    
    @property
    def K(self):
        with open(self.info_files[0], 'r') as f:
            info = json.load(f)
        K = np.array(info['K']).reshape(3,3)
        K[:2] *= self.downscale
        return K