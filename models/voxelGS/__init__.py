from .gaussian_model import GaussianModel, quantize_ste
import torch
from .render import render,prefilter_voxel
from libs.loss_utils import l1_loss, ssim, psnr, lpips_fn
import numpy as np
import time 
class Model(GaussianModel):
    def __init__(self, cfg):
        self.cfg = cfg

        super().__init__(**self.cfg.gaussian)

    def gussian_cdf(self, inputs, loc=0.0, scale=1.0, sigma3=False):
        if type(loc) is not torch.Tensor:
            loc = torch.ones_like(inputs) * loc
        if type(loc) is not torch.Tensor:
            scale = torch.ones_like(inputs) * scale
        scale = scale.clip(0.01,None)
        loc = torch.nan_to_num(loc, nan=0.0)
        m = torch.distributions.laplace.Laplace(loc, scale)
        if sigma3:
            inputs = inputs.clamp(loc - 3 * scale, loc + 3 * scale)
        lower = m.cdf(inputs - 0.5)
        upper = m.cdf(inputs + 0.5)
        likelihood = upper - lower
        bits = -torch.log2(likelihood.clip(1e-5)).sum()
        return bits

    def setup(self):
        self.set_appearance(self.cfg.num_cameras)
        self.training_setup(self.cfg.optimize)
        
    def setup_from_scene(self, scene):
        self.cameras_extent = scene.nerf_normalization()["radius"]
        self.cfg.white_background = scene.white_background
        if self.get_anchor.shape[0] == 0:
            self.create_from_pcd(scene.pcd,self.cameras_extent)
            # print("Creating from scene pcd.")
        self.setup()
        

    def render(self,viewpoint_camera, retain_grad=False, scaling_modifier = 1.0, override_color = None):
        voxel_visible_mask = prefilter_voxel(viewpoint_camera, self, pipe=self.cfg.render, bg_color = self.bg_color)
        render_pkg = render(viewpoint_camera, pc=self, pipe=self.cfg.render, bg_color=self.bg_color, scaling_modifier = scaling_modifier, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        render_pkg['voxel_visible_mask'] = voxel_visible_mask
        return render_pkg

    def training_step(self, viewpoint_cam, step, istest=False):
        if step>self.cfg.rate_proxy.update_from:
            self.setqs(1)
        else:
            self.setqs(128)
        viewpoint_cam.to_cuda()
        retain_grad = (step < self.cfg.densify.update_until) and (not istest)
        t = time.time()
        render_pkg = self.render(viewpoint_cam, retain_grad=retain_grad)
        t = time.time() - t
        # Loss
        image = render_pkg["render"] 
        scaling = render_pkg["scaling"]
        gt_image = viewpoint_cam.original_image 
        Ll1 = l1_loss(image, gt_image)
        loss0 = (1.0 - self.cfg.lambda_dssim) * Ll1 + self.cfg.lambda_dssim * (1.0 - ssim(image, gt_image)) +  self.cfg.scaling_reg*scaling.prod(dim=1).mean()

        if ( (step>=self.cfg.rate_proxy.update_from and step%self.cfg.rate_proxy.update_interval==0) or istest):
            bit_scale = self.gussian_cdf(self.get_quantize_scaling, loc=self.get_quantize_scaling.mean(), scale=self.get_quantize_scaling.std(),sigma3=True)
            bit_anchor_feat = self.gussian_cdf(self.get_quantize_anchor_feat, loc=self.get_quantize_anchor_feat.mean(), scale=self.get_quantize_anchor_feat.std(),sigma3=True)
            bit_offset = self.gussian_cdf(self.get_quantize_offset, loc=self.get_quantize_offset.mean(), scale=self.get_quantize_offset.std(),sigma3=True)
            bits = bit_scale + bit_anchor_feat + bit_offset
            abpp = bits/self.get_anchor_num

            loss = loss0 + abpp*self.cfg.rate_proxy.lamda 
            with torch.no_grad():
                metrics = {
                    "psnr": psnr(image, gt_image).mean().item(),
                    "ssim": ssim(image, gt_image).mean().item(),
                    "lips": lpips_fn(image, gt_image, normalize=False).detach().mean().item(),
                    "gs_num": self.get_anchor.shape[0],
                    "loss": loss.item(),
                    "abpp": abpp.item(),
                    "est_size": (abpp+self.cfg.rate_proxy.geometry_rate).item()*self.get_anchor_num/8/1024/1024,
                    "render_time": t,
                }
            return loss, metrics, render_pkg
        else:
            loss = loss0 
            with torch.no_grad():
                metrics = {
                    "loss": loss.item(),
                    "psnr": psnr(image, gt_image).mean().item(),
                    "anchor": self.get_anchor.shape[0]
                }
            return loss, metrics, render_pkg

    @torch.no_grad()
    def delte_global_mask(self,render_pkg):
        self._offset[render_pkg['entropy_data']['visible_mask']] = self._offset[render_pkg['entropy_data']['visible_mask']] * render_pkg['entropy_data']['mask_rate']
        prune_mask = ~render_pkg['entropy_data']['global_mask'] 
        self.opacity_accum = self.opacity_accum[~prune_mask]
        self.anchor_demon = self.anchor_demon[~prune_mask]
        prune_mask_coupled = torch.tile(prune_mask[:,None], (1,self.n_offsets)).reshape(-1)
        self.offset_gradient_accum = self.offset_gradient_accum[~prune_mask_coupled]
        self.offset_denom = self.offset_denom[~prune_mask_coupled]
        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)


    @torch.no_grad()
    def post_training_step(self, render_pkg, iteration):
        opt = self.cfg.densify 
        gaussians = self
        viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity,voxel_visible_mask = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["voxel_visible_mask"]
        # Densification
        if iteration < opt.update_until and iteration > opt.start_stat:
            # add statis
            gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
            
            # densification
            if iteration > opt.update_from and iteration % opt.update_interval == 0:
                self.delte_global_mask(render_pkg)
                gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
        elif iteration == opt.update_until:
            del gaussians.opacity_accum
            del gaussians.offset_gradient_accum
            del gaussians.offset_denom
            torch.cuda.empty_cache()
                
        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)