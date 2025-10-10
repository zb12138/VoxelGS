import os,glob,time,torch,gc 
from copy import deepcopy
from utils.config import Config
from utils.logger import create_logger
from models.model import load_model
from models.trainer import Trainer
from datasets import load_dataset
from gscoder.compressor import gs_encoder, gs_decoder
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('-ply', type=str, help='GS ply path', default=None)
argparser.add_argument('-bin', type=str, help='GS bin folder to decode', default=None)
argparser.add_argument('-config', type=str, help='config file', default='')
argparser.add_argument('-show', action='store_true',  help='Show GS ply', default=None)
argparser.add_argument('-eval',action='store_true',  help='Evaluate GS ply', default=None)
argparser.add_argument('-encode',action='store_true',  help='Encode GS ply', default=None)
args = argparser.parse_args()

if args.encode:
    result = gs_encoder(args.ply,print=print)

if args.bin is not None:
    args.ply,dtime = gs_decoder(args.bin)
    print('decompressed gs ply saved at:', args.ply)
    
if args.show is not None or args.eval is not None:
    assert args.ply is not None, 'please specify ply path with -ply'
    if args.config == '':
        # try to find config file
        args.config = os.path.join(os.path.dirname(args.ply),'..','config.yaml')
        assert os.path.exists(args.config), 'config file not found, please specify it with --config'
    config = Config(config_path=args.config)
    cfg = config.parse_args()
    gs_model = load_model(cfg.model)
    if args.show is not None:
        gs_model.show(args.ply)
    if args.eval is not None:
        print('evaluating...')
        dataset = load_dataset(cfg.data)
        trainer = Trainer(gs_model, dataset, cfg.train)
        trainer.load_ckpt(args.ply)
        print(trainer.validation_loop())