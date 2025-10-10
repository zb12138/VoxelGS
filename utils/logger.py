from loguru import logger as loguru_logger
import sys
import re
import inspect, os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path    

class Logger:
    def __init__(self, path=''):
        self.logger_path = path  
        self.configure_logger()

    def configure_logger(self):
        loguru_logger.remove() 
        loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> |' \
                        ' <level>{level: <8}</level> |' \
                        ' <level>{message}</level>'
        if self.logger_path != '':
            loguru_logger.add(self.logger_path, rotation="1 MB", level="INFO", format=loguru_format)
        loguru_logger.add(
            sys.stdout,
            level="INFO",
            format=loguru_format,
            colorize=True 
        )

    def set_logger_path(self, path,append=True):
        if not append:
            if os.path.exists(path):
                os.remove(path) 
        self.logger_path = path
        self.configure_logger()

    def error(self, *args):
        message = self.line_info() + ' '.join(map(str, args))
        loguru_logger.error(message)

    def info(self, *args):
        message = self.line_info() + ' '.join(map(str, args))
        loguru_logger.info(message)

    def warning(self, *args):
        message = self.line_info() + ' '.join(map(str, args))
        loguru_logger.warning(message)

    def line_info(self):
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        return f'{filename} line:{lineno} | '

    def format_info(self, *argsIn, decimals=2):
        args = []
        for arg in argsIn:
            if type(arg) == torch.Tensor:
                arg = arg.detach().cpu().numpy()
            if type(arg) == np.ndarray:
                arg = np.array2string(arg, precision=decimals, separator=', ')
            args.append(arg)
        return ' '.join(map(str, args))

def format_decimals(input_string):
    if type(input_string) != str:
        input_string = str(input_string)
    pattern = r'\d+\.\d+'
    def format_match(match):
        return f"{float(match.group()):.3f}"
    result = re.sub(pattern, format_match, input_string)
    return result

logger = Logger()

def create_logger(cfg):
    if type(cfg.ckpt) is int:
        cfg.ckpt = str(Path(cfg.output_dir) / cfg.experiment_name / cfg.dataset.split('data/')[-1] / 'point_cloud' / f'point_cloud_{cfg.ckpt}.ply')
    if cfg.ckpt is not None:
        if not Path(cfg.ckpt).exists():
            logger.warning(f"Checkpoint path {cfg.ckpt} does not exist. Creating a new experiment directory.")
            cfg.ckpt = None
    if cfg.resume and cfg.ckpt is not None:
        cfg.ckpt_dir = Path(cfg.ckpt)
        cfg.exp_dir = cfg.ckpt_dir.parent.parent
        cfg.log_file = cfg.exp_dir / 'log.log'
        logger.set_logger_path(str(cfg.log_file), append=True)
        cfg.ckpt_dir = str(cfg.ckpt_dir)
    else:
        now = datetime.now()
        formatted_time = ''#now.strftime("%y%m%d_%H%M")
        cfg.exp_dir = Path(cfg.output_dir) / cfg.experiment_name / cfg.dataset.split('data/')[-1] / formatted_time
        # os.makedirs(cfg.exp_dir, exist_ok=True)
        cfg.log_file = cfg.exp_dir / 'log.log'
        logger.set_logger_path(str(cfg.log_file), append=True)

    cfg.test_log = str(cfg.exp_dir / 'test.log')
    cfg.exp_dir = str(cfg.exp_dir)
    cfg.log_file = str(cfg.log_file)
    return logger 