import os,glob,time,gc,sys 
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from copy import deepcopy
from utils.config import Config
from utils.logger import create_logger
from models.model import load_model
from models.trainer import Trainer
from datasets import load_dataset
from gscoder.compressor import gs_encoder, gs_decoder
from utils.read_log import stat_log

for config_path in glob.glob('config/MipNerf360.yaml'):
    configer = Config(config_path=config_path)
    cfg_dataset = configer.parse_args()
    cfg_dataset.model.white_background = cfg_dataset.data.white_background

    for scene in glob.glob(cfg_dataset.train.dataset):
        cfg = deepcopy(cfg_dataset)
        cfg.train.dataset = scene
        cfg.data.path = scene
        logger = create_logger(cfg.train)
        if os.path.exists(cfg.train.test_log):
            print(f'skipping {scene}, already done')
            # continue
        # save config
        print('start training...', scene)
        cfg.config = os.path.join(cfg.train.exp_dir, 'config.yaml')
        configer.save_as_yaml(cfg.config, cfg)
        # logger.warning(cfg)
        gs_model = load_model(cfg.model)
        dataset = load_dataset(cfg.data)
        trainer = Trainer(gs_model, dataset, cfg.train)
        
        # training
        train_time = time.time()
        trainer.training_loop()
        train_time = time.time() - train_time

        gs_ply = os.path.join(cfg.train.exp_dir, 'point_cloud', f'point_cloud_{cfg.train.training_steps}.quantized.ply')
        train_result = {'train_time (min)': round(train_time/60,2)}

        # compressing
        print('start compressing...', gs_ply)
        logger.set_logger_path(cfg.train.test_log)
        encode_result = gs_encoder(gs_ply,print=logger.info)
        
        bin_dir = os.path.join(cfg.train.exp_dir,'gsbin')

        # decompressing
        print('start decompressing...', bin_dir)
        dec_ply,decode_result = gs_decoder(bin_dir)
        
        # testing
        print('start testing...', dec_ply)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
        trainer.load_ckpt(dec_ply)
        eval_result = trainer.validation_loop(save_imges=True)

        logger.warning(train_result)
        logger.warning(encode_result)
        logger.warning(decode_result)
        logger.warning(eval_result) 

    stat_log(os.path.dirname(cfg.train.exp_dir))