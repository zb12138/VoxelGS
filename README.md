## Voxel-GS: Quantized Scaffold Gaussian Splatting Compression with Run-Length Coding [DCC26](https://arxiv.org/abs/2512.17528)

### Dataset
```
voxelGS/data
├── DeepBlending 
    ├── drjohnson
    └── playroom
├── MipNerf360 
    ├── bicycle
    ├── bonsai
    ├── counter
    ├── flowers
    ├── garden
    ├── kitchen
    ├── room
    ├── stump
    └── treehill
├── Nerf_Synthetic 
    ├── chair
    ├── drums
    ├── ficus
    ├── hotdog
    ├── lego
    ├── materials
    ├── mic
    └── ship
└── T2T 
    ├── train
    └── truck
```
### Environment
cuda 11.7+python3.8
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install omegaconf,loguru,open3d==0.19.0,opencv-python,plyfile,tensorboard,termcolor,torch_scatter,jaxtyping,einops,lpips
pip install models/submodules/* # from Scaffold-GS
```

### Train
1. Run ```python Train.py```. After training, ```stat_log``` will generate statistical information and save the results in the output folder.
2. Note: Due to the randomness of the training process, there may be slight differences in the results.
3. Alternatively, you can [download](https://huggingface.co/datasets/zb1213899/VoxelGS_BIN/tree/main) and unzip ```Results/BIN.zip``` to directly access the compressed bin and decompress it using the command:
```python Coder.py -bin Results/BIN/DeepBlending/playroom/gsbin -eval```

### Encoding, decoding, evaluation, show from separated GS PLY
1. Provide -ply -encode for encoding  
```python Coder.py -ply output/base/Nerf_Synthetic/hotdog/point_cloud/point_cloud_30000.quantized.ply -encode```

2. Provide -bin for decoding  
```python Coder.py -bin output/base/Nerf_Synthetic/hotdog/gsbin```

3. Add -show -eval for visualization and evaluation  
```python Coder.py -bin output/base/Nerf_Synthetic/hotdog/gsbin -show -eval```


