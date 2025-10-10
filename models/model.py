import numpy as np
import open3d as o3d
import cv2
from datetime import datetime
import json
from importlib import import_module
# from models.baseGS import Model
import torch
from datasets.utils import Camera
from libs import network_gui
class Viewer:
    def __init__(self, gs_model):
        self.gs_model = gs_model
        self.load_ply = gs_model.load_ply
        self.render = gs_model.render
        self.viewer_alive = False
        
    def __call__(self, ply=None, no_block=False):
        def renderImg():
            viewpoint_cam = Camera(R=self.R ,T=self.T, width=self.width,height=self.height,FoVy=self.fovy,FoVx=self.fovx,scale=self.scale)
            viewpoint_cam.to_cuda()
            rendering = self.render(viewpoint_cam,scaling_modifier=self.scaling_modifier)["render"]  # [3, H, W]
            rendering = torch.round(rendering.mul(255).clamp_(0, 255)) 
            rendering_np = rendering.permute(1,2,0).cpu().detach().numpy().astype(np.uint8)
            rendering_np = rendering_np.copy()[:, :, ::-1]
            return np.ascontiguousarray(rendering_np)

        def addscale(vis):
            self.scale += 0.1
            self.update = True
        def decscale(vis):
            self.scale -= 0.1
            self.update = True
        def addscaling_modifier(vis):
            self.scaling_modifier += 0.1
            self.update = True
        def decscaling_modifier(vis):
            self.scaling_modifier -= 0.1
            self.update = True

        def quit(vis):
            self.viewer_alive = False
            self.runing = False

        def pause(vis):
            self.runing = not self.runing

        def save(vis):
            file = f'ScreenCamera_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
            cv2.imwrite(file+'.png',self.img.astype('uint8'))
            cv2.waitKey(1)
            file_name = file+'.json'
            with open(file_name, 'w') as json_file:
                json.dump({'R':self.R.tolist(),'T':self.T.tolist(),'width':self.width,'height':self.height,'fovy':self.fovy,'fovx':self.fovx}, json_file, indent=4)

        if ply is not None:
            self.load_ply(ply)
            self.gs_model.setup()

        if not self.viewer_alive:
            self.scale = 1.0
            self.width, self.height = 800, 800
            self.fovx, self.fovy = 1, 1
            self.update = True
            self.runing = True
            self.scaling_modifier = 1.0
            
            # cv2.namedWindow("GS Viewer", cv2.WND_PROP_FULLSCREEN)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.register_key_callback(ord("Q"), quit)
            self.vis.register_key_callback(ord("W"), addscale)
            self.vis.register_key_callback(ord("S"), decscale)
            self.vis.register_key_callback(ord("A"), addscaling_modifier)
            self.vis.register_key_callback(ord("D"), decscaling_modifier)
            self.vis.register_key_callback(ord("P"), save)
            self.vis.register_key_callback(ord(" "), pause)

            self.vis.create_window(window_name="Coordinate Frame", width=200, height=200,left=0,top=0)
            self.vis.add_geometry(coordinate_frame)
            self.viewer_alive = True
            self.prev_view_param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            print(f'viewing {ply}...')
            print('Press q to exit when the cursor is focused on the coordinate.')

        if not self.runing:
            self.vis.poll_events()
        while self.runing:
            self.vis.poll_events()
            self.vis.update_renderer()
            curr_param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            if not np.allclose(curr_param.extrinsic, self.prev_view_param.extrinsic, atol=1e-6) or self.update or no_block:
                R = curr_param.extrinsic[:3, :3].copy()
                T = curr_param.extrinsic[:3, 3].copy()
                R[:,1] = - R[:,1]
                self.R = R
                self.T = T
                self.prev_view_param = curr_param
                self.img = renderImg()
                self.update = False
                cv2.imshow("GS Viewer", self.img/255)
            key = cv2.waitKey(1)
            if key == 27 or no_block: 
                break
        if not self.viewer_alive:
            cv2.destroyAllWindows()
            self.vis.destroy_window()
            if no_block: self.viewer_alive = True

def load_model(cfg):
    gaussian_model = import_module(f"models.{cfg.model_name}")
    class GS_Model(gaussian_model.Model):
        def __init__(self, cfg):
            self.cfg = cfg
            self.bg_color = torch.tensor([1.0, 1.0, 1.0]).cuda() if self.cfg.white_background else torch.tensor([0.0, 0.0, 0.0]).cuda()
            super().__init__(self.cfg)
            self.show = Viewer(self)
            
        def render_gui(self,step):
            if step%self.cfg.gui_flash_rate!=0:
                return
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, convert_SHs_python, compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = self.render(custom_cam, scaling_modifier = scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, '')
                    if do_training or not keep_alive:
                        break
                except Exception as e:
                    network_gui.conn = None
            pass

    return GS_Model(cfg)