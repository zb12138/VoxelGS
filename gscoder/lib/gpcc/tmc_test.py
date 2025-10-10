'''
Author: chunyangf@qq.com
'''
import re
import glob
EXPNAME = 'Exp/GPCC'
from tqdm import tqdm
import pandas as pd
from termcolor import cprint
import subprocess
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataclasses import dataclass
import ptIO

TMC_PATH = os.path.join(os.path.dirname(__file__),'tmc13v23')
PCERROR_PATH = os.path.join(os.path.dirname(__file__),'pc_error') 
TEMP_DIR = os.path.join(os.path.dirname(__file__),'temp')

def parse_output(strs, type='tmc13Encode', attributeName='color'):
    if attributeName=='': attributeName='xyz'

    def parse_helper(head, head2="\n", count=0):
        try:
            return float(strs.split(head)[count + 1].split(head2)[0])
        except:
            return np.nan

    if type in ['tmc13Encode']:
        SliceNum = int(parse_helper('Slice number: '))
        positions_byte, positions_time, color_byte, color_time = [], [], [], []
        for i in range(SliceNum):
            positions_byte.append(parse_helper('positions bitstream size ', ' B', i))
            positions_time.append(parse_helper('positions processing time (user): ', ' s', i))
            color_byte.append(parse_helper(f'{attributeName}s bitstream size ', ' B', i))
            color_time.append(parse_helper(f'{attributeName}s processing time (user): ', ' s', i))
        Total_size = (parse_helper('Total bitstream size ', ' B'))
        Total_time = (parse_helper('Processing time (user): ', ' s'))
        result = {'pos_bits': sum(positions_byte) * 8, 'atr_bits': sum(color_byte) * 8, 'total_bits': Total_size * 8, 'en_pos_time': np.round(sum(positions_time), 5), 'en_atr_time': np.round(sum(color_time), 5), 'en_total_time': Total_time * 1.0}
        return result

    if type in ['tmc13Decode']:
        SliceNum = strs.count('positions processing time (user)')
        positions_byte, positions_time, color_byte, color_time = [], [], [], []
        for i in range(SliceNum):
            positions_time.append(parse_helper('positions processing time (user): ', ' s', i))
            color_time.append(parse_helper(f'{attributeName}s processing time (user): ', ' s', i))
        Total_time = (parse_helper('Processing time (user): ', ' s')) * 1.0
        color_time = np.round(sum(color_time), 5)
        positions_time = np.round(sum(positions_time), 5)
        result = {'de_total_time': Total_time, 'de_pos_time': positions_time, 'de_atr_time': color_time}
        return result

    if type in ['pc_error']:
        msed1 = parse_helper('mseF      (p2point): ')
        psnrd1 = parse_helper('mseF,PSNR (p2point): ')
        msed2 = parse_helper('mseF      (p2plane): ')
        psnrd2 = parse_helper('mseF,PSNR (p2plane): ')
        if attributeName == 'color':
            # mseY = parse_helper('c[0],    F         : ')
            # mseU = parse_helper('c[1],    F         : ')
            # mseV = parse_helper('c[2],    F         : ')
            psnrY = parse_helper('c[0],PSNRF         : ')
            psnrU = parse_helper('c[1],PSNRF         : ')
            psnrV = parse_helper('c[2],PSNRF         : ')
            result = {'d1': psnrd1, 'd2': psnrd2, 'Y': psnrY, 'U': psnrU, 'V': psnrV}
        if attributeName == 'reflectance':
            # mseR = parse_helper('  r,       F         : ')
            psnrR = parse_helper(' r,PSNR   F         : ')
            result = {'d1': psnrd1, 'd2': psnrd2, 'Reflectance': psnrR}
        if attributeName == 'xyz':
            result = {'d1': psnrd1, 'd2': psnrd2}
        return result




@dataclass
class point_cloud:
    position = np.empty((0,3))
    colors = np.empty((0,3))
    reflectance = np.empty((0,1))
    features = np.empty((0,3))
    path = ""
    def __repr__(self) -> str:
        s = f"point_cloud: {self.path}\n"
        s += "  position: {}\n".format(self.position.shape)
        if self.colors.size>0:
            s += "  colors: {}\n".format(self.colors.shape)
        if self.reflectance.size>0:
            s += "  reflectance: {}\n".format(self.reflectance.shape)
        if self.features.size>0:
            s += "  features: {}\n".format(self.features.shape)
        return s

class PointCloudTest():
    def __init__(self, attributeName="",tmc_temp_path=TEMP_DIR,log_path='',print_screen=False,gpcc_path=TMC_PATH) -> None:
        """
        attribute: "", "color", "reflectance"
        """
        self.raw_pt = point_cloud()
        self.inputPointNum = None
        self.attributeName = attributeName
        self.tmc_temp_path = tmc_temp_path
        self.log_path = log_path
        self.print_screen = print_screen
        self.gpcc_path = gpcc_path
        self.recon_path = os.path.join(tmc_temp_path,'recon.ply')
        self.rate_id = '-1' # 0 for lossless, -1 for undefined (no cfg)
        os.makedirs(tmc_temp_path, exist_ok=True)

    def run_cmd(self, cmd):
        out = subprocess.check_output(cmd, shell=True, encoding='utf-8', errors='ignore')
        if self.print_screen:
            cprint(cmd, 'red', 'on_green')
            print(out, flush=True)
        if len(self.log_path):
            with open(self.log_path, 'w+', newline='') as f:
                f.write(out)
        return out

    def set_ply(self,input):
        if isinstance(input, str):
            path = input
        if isinstance(input, np.ndarray):
            assert input.shape[1] in [3, 4, 6]
            path = os.path.join(self.tmc_temp_path, 'src', 'input.ply')
            if input.shape[1] == 3:
                self.attributeName = ""
            ptIO.pc_write_gpcc(path, points=input[:, :3], attribute=input[:, 3:], asAscii=True)
        self.source_path = path
        self.basename = os.path.basename(path)[:-4].replace('/', '_')
        self.recon_path = os.path.join(self.tmc_temp_path,'recon.ply')
        self.tmc_bin_path = os.path.join(self.tmc_temp_path, 'bin', self.basename) + '.bin'

    def read(self, path=None):
        if path is None:
            path = self.source_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        assert self.attributeName in ["","color", "reflectance"]
        xyz, feat = ptIO.pc_read(path, feat_names=self.attributeName)
        self.raw_pt.position = xyz
        self.raw_pt.features = feat
        self.raw_pt.path = path
        if self.attributeName in ["color"]:
            self.raw_pt.colors = feat
        if self.attributeName in ["reflectance"]:
            self.raw_pt.reflectance = feat
        self.inputPointNum = self.raw_pt.position.shape[0]
        self.set_ply(path)

    @property
    def pt_num(self):
        if self.inputPointNum is None:
            self.read()
        return self.inputPointNum

    def compressByTmc(self, config='', otherParams=''):
        if "--compressedStreamPath=" in otherParams:
            bin_path = otherParams.split('--compressedStreamPath=')[-1].split(' ')[0]
            if bin_path  != self.tmc_bin_path:
                self.tmc_bin_path = bin_path
        os.makedirs(os.path.dirname(self.tmc_bin_path), exist_ok=True)
        config = self.match_cfg(config, cfg_type='encoder')
        if len(config): config = f'-c {config}'
        cmd = f'{self.gpcc_path} --mode=0 --uncompressedDataPath={self.source_path} --compressedStreamPath={self.tmc_bin_path} {otherParams}  {config}'
        out = self.run_cmd(cmd)
        out = parse_output(out, type='tmc13Encode', attributeName=self.attributeName)
        for k, v in list(out.items()):
            if k.endswith('_bits'):
                out[k.replace('_bits', '_bpp')] = round(v / self.pt_num, 4)
        return out

    def deCompressByTmc(self, tmc_bin_path=None, config='', otherParams=''):
        if tmc_bin_path is not None:
            self.tmc_bin_path = tmc_bin_path
        os.makedirs(os.path.dirname(self.recon_path), exist_ok=True)
        config = self.match_cfg(config, cfg_type='decoder')
        if len(config): config = f'-c {config}'
        cmd = f'{self.gpcc_path} --mode=1 --reconstructedDataPath={self.recon_path} --compressedStreamPath={self.tmc_bin_path} {otherParams} {config} '
        out = self.run_cmd(cmd)
        return parse_output(out, type='tmc13Decode', attributeName=self.attributeName)

    def psnrMPEG(self, config=''):
        config = self.match_cfg(config, cfg_type='pcerror')
        resolution = ''
        if self.attributeName == 'color':
            attri_parm = '-c 1'
        elif self.attributeName == 'reflectance':
            attri_parm = '-l 1'
            resolution = '-r 30000'  # for lidar
        else:
            attri_parm = ''
        if len(config): config = f'-c {config}'
        cmd = f"{PCERROR_PATH} -a {self.source_path} -b {self.recon_path} {resolution} {attri_parm} {config}"
        out = self.run_cmd(cmd)
        return parse_output(out, type='pc_error', attributeName=self.attributeName)


    def match_cfg(self, cfg_path, cfg_dir='lib/gpcc/cfg',cfg_type='encoder'):
        if cfg_path is None or cfg_path=='':
            return ""
        if re.match(r'r(\d{2})', cfg_path):
            cfg_path=os.path.join(cfg_dir,self.attributeName,cfg_path,cfg_type)
        if os.path.exists(cfg_path): 
            match = re.search(r'r(\d{2})', cfg_path)
            if match and cfg_type=='encoder':
                self.rate_id = match.group(1)
            return cfg_path
        else:
            print(f'No cfg matched: {cfg_path} in {cfg_dir}')
        return ""
        


def TMC_compress(input, attributeName='color', encoder_cfg='', decoder_cfg=None, pc_error_cfg=None, encoder_otherParams='', print_scree=False, gpcc_path=TMC_PATH, tmc_temp_path=TEMP_DIR,log_path=''):
    """
    input: ply path or ndarray Nx6 or Nx4 or Nx3
    attribute: "", "color", "reflectance"
    encoder_cfg: gpcc encoder cfg path, None to disable compression
    decoder_cfg: gpcc decoder cfg path, None to disable decompression
    pc_error_cfg: pc_error cfg path, None to disable psnr calculation
    """
    cpt = PointCloudTest(attributeName=attributeName, print_screen=print_scree, gpcc_path=gpcc_path, log_path=log_path,tmc_temp_path=tmc_temp_path)
    cpt.set_ply(input)
    result = {'name': cpt.source_path, 'inputPointNum': cpt.pt_num, 'attribute': cpt.attributeName}
    if cpt.attributeName!='':
        encoder_otherParams += f' --attribute={attributeName}'
    else:
        encoder_otherParams += f' --disableAttributeCoding=1'

    result.update(cpt.compressByTmc(encoder_cfg, otherParams=encoder_otherParams))
    if decoder_cfg is not None or pc_error_cfg is not None:
        result.update(cpt.deCompressByTmc(config=decoder_cfg, otherParams=''))
    if pc_error_cfg is not None:
        result.update(cpt.psnrMPEG(config=pc_error_cfg))
    result.update({'rate_id':cpt.rate_id})
    return result

if __name__ == '__main__':
    # cfg = f'lib/gpcc/cfg/longdress_vox10_1300/r06/encoder.cfg'
    # cfg2 = f'lib/gpcc/cfg/longdress_vox10_1300/r06/decoder.cfg'
    # result = TMC_compress('Data/8iVFBv2/longdress/Ply/longdress_vox10_1051.ply','color',cfg, cfg2, print_scree=True)
    # print(result)

    cfg = f'lib/gpcc/cfg/longdress_vox10_1300/r00/encoder.cfg'
    cfg2 = f'lib/gpcc/cfg/longdress_vox10_1300/r00/decoder.cfg'
    result = TMC_compress('Data/8iVFBv2/longdress/Ply/longdress_vox10_1051.ply','color',cfg, cfg2, print_scree=True)
    print(result)

    # cfg = f'lib/gpcc/cfg/ford/r00/encoder.cfg'
    # cfg2 = f'lib/gpcc/cfg/ford/r00/decoder.cfg'
    # result = TMC_compress('Data/MPEG/MPEGCat3Frame/Ford_01_q1mm/Ford_01_vox1mm-0100.ply','reflectance',cfg, cfg2, print_scree=True)
    # print(result)