from gscoder.lib import ptIO
from gscoder.lib.gpcc.tmc_test import TMC_compress,PointCloudTest,TMC_PATH
from gscoder.lib.resAc.ac_warpper import encode_res, encode_res_multichannel, decode_res_multichannel
import numpy as np
import os, glob
import time

QS = {'f_offset':1, 'f_anchor_feat':1, 'scale':1}
LOD_NUM = 1
def gs_encoder(file, print=None):
    assert os.path.exists(file), f'file not found: {file}'
    start0 = time.time()
    bin_folder = os.path.dirname(os.path.dirname(file)) + '/gsbin'
    if os.path.exists(bin_folder):
        os.system(f'rm -r {bin_folder}')
    os.makedirs(bin_folder)

    # compress MLP
    src_mlp = file.replace('.quantized','').replace('.ply','.pth')
    os.system(f'zip -r {bin_folder}/mlp.bin {src_mlp} -j -q')
    mlp_bits = os.path.getsize(f'{bin_folder}/mlp.bin')*8

    p, c = ptIO.pc_read(file, feat_names=['f_offset','f_anchor_feat','scale'])
    p, idx = ptIO.sortByMorton(p, return_idx=True)
    for k in c.keys():
        c[k] = np.round(c[k][idx]/QS[k]) 

    # compress GEO
    cfg = f'gscoder/lib/gpcc/cfg/color/r00/encoder.cfg'
    xyz_result = TMC_compress(p,attributeName='',encoder_cfg = cfg, decoder_cfg=None, pc_error_cfg=None, print_scree=False, tmc_temp_path=f'{bin_folder}', encoder_otherParams=f'--compressedStreamPath={bin_folder}/geo.bin')
    os.system(f'rm -r {bin_folder}/src')
    pos_bits = os.path.getsize(f'{bin_folder}/geo.bin')*8

    t_coder = {'pos':xyz_result['en_pos_time']}
    # compress ATR
    atr_bits = 0
    part_bits = {'geo':pos_bits,'mlp':mlp_bits}
    for c_name,c_value in c.items():
        start = time.time()
        code=encode_res_multichannel(c_value)
        run_length = len(code)*8
        t2 =(time.time()-start)
        t_coder[c_name] = t2
        bin_path = f'{bin_folder}/{c_name}_bypass.bin'
        part_bits[c_name] = run_length
        atr_bits += run_length
        with open(bin_path,'wb') as f:
            f.write(code)
        if print is not None: print(f'  {c_name} {run_length} bits, range {c_value.max()-c_value.min()}')

    t_coder['coder_time'] = sum(list(t_coder.values()))
    t_all = time.time()-start0
    t_coder['all_time'] = t_all
    t_coder = {k:round(v,3) for k,v in t_coder.items()}

    total_bits = pos_bits + atr_bits + mlp_bits
    ptNum = p.shape[0]
    if print is not None:
        for k,v in part_bits.items():
            print(' ',k, round(v/ptNum, 3) , 'bpp', round(v/8/1024/1024,3), 'MB', round(v/total_bits*100,3), '%')
        total_size2 = sum([os.path.getsize(x)*8 for x in glob.glob(f'{bin_folder}/*.bin')])
        print('  gs num',ptNum,'total size MB', round(total_bits/1024/1024/8,3))
        print('  check total size:', total_size2, total_bits, 'bits', round(total_bits/1024/1024/8,3), 'MB')
        print('  time (s)', t_coder)

    result = {'total_bitsMB':total_bits/1024/1024/8, 'pos_bits':pos_bits, 'atr_bits':atr_bits, 'mlp_bits':mlp_bits, 'part_bits':part_bits, 'gsNum':ptNum, 'enc_time':t_coder}
    return result


def gs_decoder(bin_folder):
    assert os.path.exists(bin_folder), f'bin folder not found: {bin_folder}'
    start = time.time()
    t_coder = {}
    decoder_pth = os.path.join(os.path.dirname(bin_folder), "dec_point_cloud")
    if os.path.exists(decoder_pth):
        os.system(f'rm -r {decoder_pth}')
    os.makedirs(decoder_pth)

    mlp_bin = f'{bin_folder}/mlp.bin'
    os.system(f'unzip -q {mlp_bin} -d {decoder_pth}')
    step=glob.glob(f'{decoder_pth}/*.pth')[0].split('point_cloud_')[-1].split('.pth')[0]

    geo_bin = f'{bin_folder}/geo.bin'
    cfg = f'gscoder/lib/gpcc/cfg/color/r00/decoder.cfg'
    cpt = PointCloudTest(attributeName="", print_screen=False, gpcc_path=TMC_PATH,tmc_temp_path=bin_folder)
    cpt.deCompressByTmc(tmc_bin_path=geo_bin, config=cfg)
    cpt.read(cpt.recon_path)
    xyz = cpt.raw_pt.position 
    os.system(f'rm -r {cpt.recon_path}')
    c = {'opacity': np.ones((xyz.shape[0],1), dtype=np.uint8), 'rot': np.zeros((xyz.shape[0],4), dtype=np.uint8)}
    for feat in ['f_offset','f_anchor_feat','scale']:
        start = time.time()
        gpcc_bin_path = f'{bin_folder}/{feat}_gpcc.bin'
        bypass_bin_path = f'{bin_folder}/{feat}_bypass.bin'
        if os.path.exists(gpcc_bin_path):
            with open(gpcc_bin_path,'rb') as f:
                code = f.read()
            c[feat] = lod_lossless_decode(code, xyz, num_lod=LOD_NUM)[:,3:]*QS[feat] 
        if os.path.exists(bypass_bin_path):
            with open(bypass_bin_path,'rb') as f:
                code = f.read()
            c[feat] = decode_res_multichannel(code, xyz.shape[0])*QS[feat]
        t_coder[feat] = time.time()-start
    t_coder['coder_time'] = sum(list(t_coder.values()))

    # save ply
    decode_ply = f'{decoder_pth}/point_cloud_{step}.quantized.ply'
    ptIO.pc_write(decode_ply, xyz, c, asAscii=False)
    t_coder['all_time'] = time.time()-start
    t_coder = {k:round(v,3) for k,v in t_coder.items()}
    return decode_ply,{'dec_time':t_coder}