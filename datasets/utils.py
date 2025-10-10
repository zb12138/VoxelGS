from typing import NamedTuple, List
from dataclasses import dataclass
import numpy as np
import random
from copy import deepcopy
from torch import nn
import torch
from libs.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:
    def __init__(self, 
                    colmap_id = 0,
                     uid = 0, 
                       R = np.eye(3),
                       T = np.zeros((3,)),
                    FoVx = 1.0,
                    FoVy = 1.0, 
                    image = None,
                    image_path = '',
                    image_name = '',
                    gt_alpha_mask = None,
                    trans=np.array([0.0, 0.0, 0.0]),
                    scale=1.0,
                    width=None,
                    height=None,
                    znear=0.01,
                    zfar=100.0,
                    mincam=False):
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_path = image_path
        self.image_name = image_name

        if image is not None:
            self.image_width = image.shape[2]
            self.image_height = image.shape[1]
            self.original_image = image.clamp(0.0, 1.0)
        else:
            self.image_width = width
            self.image_height = height
            self.original_image =  torch.zeros((3, self.image_height, self.image_width)) 

        if not mincam and image is not None:
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale

        if not mincam:
            self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_cuda(self):
        self.original_image = self.original_image.cuda()
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()

class BaseDataset:
    def __init__(self, cfg):
        self.path = cfg.path
        self.white_background = cfg.white_background
        self.train_cameras: List[Camera] = []
        self.test_cameras: List[Camera] = []
        self.pcd = None

    def nerf_normalization(self):
        cam_info = self.train_cameras+self.test_cameras
        if len(cam_info) == 0:
            return {"translate": np.array([0.0, 0.0, 0.0]), "radius": 10.0}
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal
        cam_centers = []
        for cam in cam_info:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        return {"translate": translate, "radius": radius}

    def train_dataloader(self):
        if (len(self.train_cameras) == 0):
            self.train_cameras = self.readCameraInfo(train=True)
        return RandomIterator(self.train_cameras, random=self.cfg.shuffle, infinite=True)

    def test_dataloader(self):
        if (len(self.test_cameras) == 0):
            self.test_cameras = self.readCameraInfo(train=False)
        return RandomIterator(self.test_cameras, random=False, infinite=False)

    def read_data_set(self, split='train'):
        if split == 'train' or split == 'all': 
            self.train_cameras = self.readCameraInfo(train=True)
        if split == 'test' or split == 'all':
            self.test_cameras = self.readCameraInfo(train=False)

class RandomIterator:
    def __init__(self, items, random = False, infinite=False):
        self.items = deepcopy(items)
        self.idx = 0
        self.randomize = random
        self.infinite = infinite
        self._shuffle_items()

    def _shuffle_items(self):
        if self.randomize:
            random.shuffle(self.items)
        self.idx = 0

    def __next__(self):
        if self.idx >= len(self.items):
            if self.infinite:
                self._shuffle_items()
            else:
                raise StopIteration
        item = self.items[self.idx]
        self.idx += 1
        return item

    def __iter__(self):
        return self

    def __len__(self):
        if not self.infinite:
            return len(self.items)
        else:
            return 10e8

if __name__ == "__main__":
    iterator = RandomIterator([1, 2, 3, 4, 5], random=True, infinite=True)
    for item in iterator:
        print(item)