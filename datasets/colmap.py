import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.graphics_utils import focal2fov, fov2focal,BasicPointCloud,getWorld2View2
from libs.general_utils import image_resize
from libs.sh_utils import SH2RGB
from pathlib import Path
from PIL import Image
import json
import numpy as np
from datasets.utils import BaseDataset, RandomIterator, Camera
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from datasets.colmap_utilis import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)

class Dataset(BaseDataset):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.path = Path(cfg.path)
        self.white_background = cfg.white_background
        self.resolution = cfg.resolution
        assert self.path.exists(), f"Path {path} does not exist."
        self.train_cameras = []
        self.test_cameras = []
        self.split_step = 8
        self.test_extrinsics = []
        self.train_extrinsics = []
        self.intrinsics = None
        self.read_insics()
        self.read_data_set(split=cfg.split)
        self.pcd = self.load_init_pcd()

    def train_test_split(self, extrinsics):
        for i, e in enumerate(extrinsics.values()):
            if i % self.split_step != 0:
                self.train_extrinsics.append(e)
            else:
                self.test_extrinsics.append(e)
        pass
        # self.train_extrinsics.extend(self.test_extrinsics)

    def read_insics(self):
        root =  self.path / "sparse/0" 
        try:
            extrinsics = read_extrinsics_binary(root / "images.bin")
            intrinsics = read_intrinsics_binary(root / "cameras.bin")
        except:
            extrinsics = read_extrinsics_text(root / "images.txt")
            intrinsics = read_intrinsics_text(root / "cameras.txt")
        self.intrinsics = intrinsics
        # sort extrinsics by name
        extrinsics = dict(sorted(extrinsics.items(), key=lambda item: item[1].name))
        self.train_test_split(extrinsics)

    def loadCamInfo(self, extr, intr, uid=0) -> Camera:
        uid=extr.id
        R, T, focal_length_x, focal_length_y = self.parse_pose(extr, intr)
        image_path = self.path / "images" / extr.name
        image = self.open_image(image_path)
        image = image_resize(image, self.resolution)
        image = (image[:3] * image[3:4]) if image.shape[0] == 4 else image
        return Camera(
            colmap_id = uid,
            uid=uid,
            R=R,
            T=T,
            FoVx=focal2fov(focal_length_x, intr.width),
            FoVy=focal2fov(focal_length_y, intr.height),
            image=image,
            image_name=Path(image_path).stem,
            image_path=image_path,
        )


    def parse_pose(self, extr, intr):
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = focal_length_y = intr.params[0]
        elif intr.model in ["PINHOLE", "OPENCV"]:
            focal_length_x, focal_length_y = intr.params[0], intr.params[1]
        else:
            raise ValueError(
                "Colmap camera model not handled: only undistorted datasets "
                "(PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )
        return R, T, focal_length_x, focal_length_y

    def load_init_pcd(self):
        ply_dir = self.path / "sparse/0" 
        ply_path = ply_dir / "points3D.ply"
        pcd = BasicPointCloud()
        if not os.path.exists(ply_path):
            print(
                "Converting point3d.bin to .ply, "
                "will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(ply_dir / "points3D.bin")
            except:
                xyz, rgb, _ = read_points3D_text(ply_dir / "points3D.txt")
            pcd.storePly(ply_path, xyz, rgb)
        try:
            pcd.fetchPly(ply_path)
        except:
            pcd = None
        return pcd

    def readCameraInfo(self, train=True):
        if train:
            extrinsics = self.train_extrinsics
        else:
            extrinsics = self.test_extrinsics
        print(f'Loding {len(extrinsics)} cameras for { "train" if train else "test"}.')
        with ThreadPoolExecutor(max_workers=32) as executor:
            fn = lambda args: self.loadCamInfo(args[1], self.intrinsics[args[1].camera_id])
            items = list(executor.map(fn, enumerate(extrinsics)))
        return items


    def open_image(self, image_path) -> Image.Image:
        bg = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])
        image = Image.open(image_path)
        image_data = np.array(image.convert("RGBA")) / 255.0
        alpha = image_data[:, :, 3:4]
        image_data = (image_data[:, :, :3] * alpha + bg * (1 - alpha)) * 255
        image = Image.fromarray(np.array(image_data, dtype=np.byte), "RGB")
        return image

if __name__=='__main__':
    from omegaconf import OmegaConf as oc
    cfg = oc.create({'path': 'data/tandt_db/tandt/truck', 'white_background': True, 'split': 'all', 'shuffle': True,'resolution':1.0})
    dataset = ColmapDataset(cfg)
    for d in dataset.test_dataloader():
        print(d)
