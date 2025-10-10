from omegaconf import OmegaConf as oc
import argparse, os
from types import SimpleNamespace
from datetime import datetime
from utils.logger import logger
import yaml
class Config():
    def __init__(self, config_path='config/base.yaml') -> None:
        self.parser = argparse.ArgumentParser(description="Configuration for the application")
        self.parser.add_argument('-config','-c', type=str, default=config_path, help='Path to the configuration file')
        self.args = None

    def add_argument(self, *args, **kwargs):
        """Add an argument to the parser."""
        self.parser.add_argument(*args, **kwargs)
        
    def parse_args(self):
        """Parse the command line arguments."""
        args, override_cfgs = self.parser.parse_known_args()
        override_cfgs = (' '.join(override_cfgs)).replace('--','-')
        parms = []
        k = 0
        for i in range(len(override_cfgs)):
            if override_cfgs[i]=='-':
                if k: parms[-1]+='='+(override_cfgs[k+1:i].strip())
                for j in range(i, len(override_cfgs)):
                    if override_cfgs[j]==' ' or override_cfgs[j]=='=':
                        parms.append(override_cfgs[i+1:j])
                        k = j
                        break
        if len(parms): parms[-1]+='='+(override_cfgs[k+1:].strip())
        override_cfgs = oc.from_cli(parms)
        if os.path.exists(args.config):
            config = oc.load(args.config)
            override_cfgs = oc.merge(config, override_cfgs)
        self.args = oc.merge(oc.create({**vars(args)}),override_cfgs)
        return self.args
    
    def save_as_yaml(self, path, cfg=None):
        """Save the configuration to a YAML file."""
        if cfg is None:
            if self.args is None:
                self.parse_args()
            cfg = self.args
        if len(os.path.dirname(path)) > 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        oc.save(cfg, path)


if __name__ == "__main__":
    config = Config()
    # config.add_argument('--pipeline', type=str, help='', default='scaffold')
    # config.save_as_yaml('test.yaml')
    