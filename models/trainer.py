from pathlib import Path
import os
from utils.logger import logger,format_decimals
from utils.config import Config
import torch
from tqdm import tqdm
from datasets.utils import BaseDataset, Camera
from libs import network_gui
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import json


class Trainer:

    def __init__(self, model, scene: BaseDataset, cfg: Config) -> None:
        self.cfg = cfg
        self.exp_dir = Path(cfg.exp_dir)
        self.ckpt_dir = self.exp_dir / 'ckpt'
        self.ply_dir = self.exp_dir / 'point_cloud'
        self.scene: BaseDataset = scene
        self.model = model

        self.global_step = 0
        self.training_steps = cfg.training_steps
        self.saving_ckpt_steps = set(cfg.saving_ckpt_steps + [self.training_steps])
        self.testing_steps = set(cfg.testing_steps + [self.training_steps])

        self.tb_writer = None  #SummaryWriter(self.exp_dir / 'tf')
        self.model.setup_from_scene(scene)
        self.load_ckpt(cfg.ckpt)

        self.metrics = {}
        self.loss = 0

    def tf_log_dict(self, scalar_dict):
        if self.tb_writer:
            for name, value in scalar_dict.items():
                try:
                    if isinstance(value, (int, float)) or len(value) == 1:
                        self.tb_writer.add_scalar(name, value, self.global_step)
                    else:
                        value = value.reshape(-1)
                        for k in range(min(len(value), 3)):
                            self.tb_writer.add_scalar(f'{name}_{k}', value[k], self.global_step)
                        self.tb_writer.add_scalar(f'{name}_mean', value.mean(), self.global_step)
                except:
                    pass

    def save_ckpt(self):
        return
        model_path = self.ckpt_dir / f'model_{self.global_step}.pth'
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save((self.model.capture(), self.global_step), model_path)
        logger.info(f"Ckpt saved at: {model_path}")

    def save_gsply(self):
        point_cloud_path = os.path.join(self.ply_dir, "point_cloud_{}".format(self.global_step) + ".ply")
        os.makedirs(self.ply_dir, exist_ok=True)
        self.model.save_ply(point_cloud_path, save_ascii=self.cfg.save_ply_ascii)
        logger.info(f"Pcd  saved at: {point_cloud_path}")

    def load_ckpt(self, file=None):
        if file is None:
            return
        assert Path(file).exists(), f"File {file} does not exist!"
        if file.endswith('.pth'):
            gs_ckpt, self.global_step = torch.load(file)
            self.model.restore(gs_ckpt)
        if file.endswith('.ply'):
            try:
                self.global_step = int(file.split('_')[-1].split('.')[0])
            except:
                pass
            self.model.setup()
            self.model.load_ply(file)
        self.model.setup()
        # logger.info(f"Loaded from {file} at step {self.global_step}")

    def training_loop(self):
        # network_gui.init(self.cfg.gui_ip, self.cfg.gui_port)
        if self.global_step >= self.training_steps:
            return
        train_dataloader = self.scene.train_dataloader()
        self.progress_bar = tqdm(range(self.training_steps), "Training", initial=self.global_step)

        for self.global_step in range(self.global_step + 1, self.training_steps + 1):
            data: Camera = next(train_dataloader)

            # self.model.render_gui(self.global_step)

            # self.model.show(no_block=True)

            self.model.update_learning_rate(self.global_step)

            self.loss, self.metrics, render_pkg = self.model.training_step(data, self.global_step)

            self.loss.backward()

            self.model.post_training_step(render_pkg, self.global_step)

            self.update_progress_bar()

            # self.tf_log_dict(self.metrics)
            if self.global_step in self.testing_steps:
                results = self.validation_loop()
                logger.info(f'test: {format_decimals(results)}')

            if self.global_step in self.saving_ckpt_steps:
                logger.info('train: ' + format_decimals(self.metrics))
                self.save_ckpt()
                self.save_gsply()
                #
        self.global_step = 0

    def save_rendering(self, image, img_dir, view_idx):
        os.makedirs(img_dir, exist_ok=True)
        image = torch.round(image.cpu() * 255).numpy().astype('uint8')
        image = image.transpose(1, 2, 0)
        Image.fromarray(image).save(os.path.join(img_dir, f'{view_idx:04d}.png'))

    @torch.no_grad()
    def validation_loop(self, save_imges=False):
        torch.cuda.empty_cache()
        test_dataloader = self.scene.test_dataloader()
        results = []
        for idx, data in enumerate(test_dataloader):
            _, metric, render_pkg = self.model.training_step(data, step=self.global_step, istest=True)
            # save images
            if save_imges:
                self.save_rendering(render_pkg['render'], self.exp_dir / 'test' / 'render', idx)
                self.save_rendering(data.original_image, self.exp_dir / 'test' / 'gt', idx)
            metric['view_idx'] = idx
            results.append(metric)
        # save results to json
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(self.exp_dir / f'perview_results_{self.global_step}.json', 'w') as f:
            json.dump(results, f, indent=4)

        results_mean = pd.DataFrame(results).mean().to_dict()
        return results_mean

    def update_progress_bar(self, step=10):
        if self.global_step % step == 0:
            metrics_pbar = {k: format_decimals(v) for k, v in self.metrics.items()}
            self.progress_bar.set_postfix(metrics_pbar)
            self.progress_bar.update(step)
        if self.global_step == self.training_steps:
            self.progress_bar.close()
