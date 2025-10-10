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

class Dataset(BaseDataset):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.path = Path(cfg.path)
        self.white_background = cfg.white_background
        self.resolution = cfg.resolution
        assert self.path.exists(), f"Path {path} does not exist."
        self.train_cameras = []
        self.test_cameras = []
        self.extension = ".png" 
        self.read_data_set(split=cfg.split)
        self.pcd = self.load_init_pcd()
        
    def load_init_pcd(self):
        ply_path = os.path.join(self.path, "points3d.ply")
        pcd = BasicPointCloud()
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd.__init__(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            pcd.storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd.fetchPly(ply_path)
        except:
            pcd = None
        return pcd

    def readCameraInfo(self, train=True):
        if train:
            transformsfile = "transforms_train.json"
        else:
            transformsfile = "transforms_test.json"
        file = self.path/transformsfile
        contents = json.loads(file.read_text(encoding="utf-8"))
        fovx = contents["camera_angle_x"]
        # print(f'Loding {len(contents["frames"])} cameras in {transformsfile}')
        with ThreadPoolExecutor(max_workers=32) as executor:
            fn = lambda args: self.loadCamInfo(args[1], fovx, args[0])
            items = list(executor.map(fn, enumerate(contents["frames"])))
        return items

    def loadCamInfo(self, frame, fovx, uid=0) -> Camera:
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        # R is stored transposed due to 'glm' in CUDA code
        R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]

        image_path = self.path / (frame["file_path"] + self.extension)
        image = self.open_image(image_path)
        fovy = focal2fov(fov2focal(fovx, image.width), image.height)
        image = image_resize(image, self.resolution)
        image = (image[:3] * image[3:4]) if image.shape[0] == 4 else image
        return Camera(
            colmap_id = uid,
            uid=uid,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=image,
            image_name=Path(image_path).stem,
            image_path=image_path,
        )

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
    cfg = oc.create({'path': 'data/NeRF_Data/nerf_synthetic/lego', 'white_background': True, 'split': 'all', 'shuffle': True,'resolution':1.0})
    dataset = BlenderDataset(cfg)
    for d in dataset.test_dataloader():
        print(d)
