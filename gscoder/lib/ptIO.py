from gscoder.lib.ptio_src.gpcc_ptio import pcread as pc_read_gpcc
from gscoder.lib.ptio_src.gpcc_ptio import pcwrite as pc_write_gpcc
from plyfile import PlyData,PlyElement
import os 
import numpy as np
import open3d as o3d
import matplotlib
def mkdirs(path): 
    if len(os.path.dirname(path)): 
        os.makedirs(os.path.dirname(path), exist_ok=True)

def pc_read(path, feat_names = []):
    """
    feat_names: list of names for each feature channel, e.g. "color"/['red', 'green', 'blue']/"reflectance", or 'all' to load all features
    if feat_names is tuple, will concatenate all features
    """
    if feat_names == 'color' or feat_names == 'refc' or feat_names == 'reflectance' or feat_names == [] or feat_names == "":
        try:
            return pc_read_gpcc(path, attribute=True)
        except:
            pass 
    return pc_read_ply(path, feat_names)

def pc_write(path, xyz, feats = None, asAscii = False, feat_names = [], xyz_types = 'f4', feats_types = ['u1']):
    """
    feats_types: list of data types for each feature channel, 'f4' for float32, 'u1' for uint8, 'i4' for int32
    feat_names: list of names for each feature channel, e.g. ['red', 'green', 'blue'], or ['fa@featDim', 'fb@featDim', ...]] to split features into multiple properties
    """
    if feat_names == 'color' or feat_names == 'refc' or feat_names == 'reflectance' or feat_names == []:
        try:
            return pc_write_gpcc(path, xyz, attribute=feats, asAscii=asAscii)
        except:
            pass 
    return pc_write_ply(path, xyz, feats, asAscii, feat_names, xyz_types, feats_types)
            
##############
def pc_write_ply(path, xyz, feats = None, asAscii = False, feat_names = ['red', 'green', 'blue'], xyz_types = 'f4', feats_types = ['u1']):
    """
    types: list of data types for each feature channel, 'f4' for float32, 'u1' for uint8, 'u2' for uint16
    feat_names: list of names for each feature channel, e.g. ['red', 'green', 'blue'], or ['fa@featDim', 'fb@featDim', ...]] to split features into multiple properties
    """
    mkdirs(path)
    dtype_full = [('x', xyz_types), ('y', xyz_types), ('z', xyz_types)]
    if feats is not None:
        if type(feats) is dict:
            feat_names = list(feats.keys())
            feats_dims = [array.shape[1] if len(array.shape)>1 else 1 for array in feats.values()]
            feats_types = [array.dtype.str for array in feats.values()]
            for i,t in enumerate(feats_types):
                if t == '<i8':
                    feats_types[i] = 'int8'
            feat_names = [x+'@'+str(y) if y>1 else x for x,y in zip(feat_names, feats_dims)]
            feats = np.concatenate(list(feats.values()), axis=1)

        featDims = []
        featList = []
        featNames = []
        featTypes = []
        if len(feats_types) == 1:
            feats_types = feats_types * len(feat_names)
        assert len(feat_names) == len(feats_types), "Number of feature names and types must match"

        for feattype, featname in zip(feats_types, feat_names):
            if '@' in featname:
                name, dim = featname.split('@')
                dim = int(dim)
                for i in range(dim):
                    featNames.append(name + '_' + str(i))
                featDims.append(dim)
                featTypes.extend([feattype] * dim)
            else:
                featNames.append(featname)
                featDims.append(1)
                featTypes.append(feattype)

        assert feats.shape[1] == sum(featDims), f"Feature dimension does not match number of feature names, found {list(zip(featNames,featDims))}, total {sum(featDims)}, expected {feats.shape[1]}"
        assert len(featNames) == len(featTypes), "Number of feature names and types must match"
        # split
        # idx = 0
        # for dim in featDims:
        #     featList.append(feats[:, idx:idx+dim])
        #     idx += dim
        feats_dtype_full = list(zip(featNames, featTypes))
        dtype_full = dtype_full + feats_dtype_full

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    data = np.concatenate((xyz, feats), axis=1)
    elements[:] = list(map(tuple, data))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el],text=asAscii).write(path)


def pc_read_ply(path, feat_names = []):
    """
    feat_names: list of names for each feature channel, e.g. "color"/['red', 'green', 'blue']/"reflectance", or 'all' to load all features
    """
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1) 

    featNames = list(plydata.elements[0].data.dtype.names)
    featNames.remove('x')
    featNames.remove('y')
    featNames.remove('z')

    if feat_names == []:
        featNames = []

    concat = False
    if type(feat_names) is tuple:
        concat = True
    if type(feat_names) is str:
        concat = True
        feat_names = [feat_names]

    # group feature names
    grouped_feats = {}
    for f in featNames:
        if f not in feat_names:
            if '_' in f:
                base_name = '_'.join(f.split('_')[:-1])
            elif f in ['red', 'green', 'blue']:
                base_name = 'color'
            elif f in ['nx', 'ny', 'nz']:
                base_name = 'normal'
            else:
                base_name = f
        else:
            base_name = f
        if base_name not in feat_names and 'all' not in feat_names:
            continue
        if base_name not in grouped_feats:
            grouped_feats[base_name] = []
        grouped_feats[base_name].append(np.asarray(plydata.elements[0][f]))
    for k in grouped_feats:
        grouped_feats[k] = np.stack(grouped_feats[k], axis=1)
    if len(grouped_feats) ==1 and len(feat_names) ==1:
        grouped_feats = grouped_feats[list(grouped_feats.keys())[0]]
    if concat and isinstance(grouped_feats, dict):
        grouped_feats = np.concatenate(list(grouped_feats.values()), axis=1)
    return xyz, grouped_feats


#############

def convertRGBtoYUV_BT709(rgb,shift = np.array([[0,128,128]]), round=False):
    # color space conversion to YUV
    transformMat = np.array([[0.212600,-0.114572,0.5000],[0.715200,-0.385428,-0.454153],[0.072200,0.5000,-0.045847]])
    out_yuv = np.matmul(rgb,transformMat) + shift
    if round:
        return np.round(out_yuv).astype(int).clip(0,255)
    return out_yuv

def convertYUVtoRGB_BT709(yuv,shift = np.array([[0,128,128]])):
    # color space conversion from YUV to RGB
    transformMat_inv = np.array([[1.0000, 1.0000, 1.0000], [0, -0.1873, 1.8556], [1.5748, -0.4681, 0]])
    rgb = np.matmul((yuv - shift), transformMat_inv)
    return np.round(rgb) 

def RGB2YCoCg(rgb, offset=np.array([0, 255, 255])):
    rgb = rgb.astype(int)
    R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)
    return np.hstack((Y, Co, Cg)) + offset.T


def YCoCg2RGB(YCoCg, offset=np.array([0, 255, 255])):
    YCoCg = YCoCg - offset.T
    Y, Co, Cg = YCoCg[:, 0:1], YCoCg[:, 1:2], YCoCg[:, 2:3]
    t = Y - (Cg >> 1)
    G = Cg + t
    B = t - (Co >> 1)
    R = B + Co
    return np.hstack((R, G, B)).astype(np.uint8)

############
def pc_read_o3d(path, attribute=False):  # fast but do not support reflectance
    pc_o3d = o3d.io.read_point_cloud(path)
    if attribute:
        assert pc_o3d.has_colors()
        pc = np.asarray(pc_o3d.points), np.asarray(pc_o3d.colors) * 255
    else:
        pc = np.asarray(pc_o3d.points)
    return pc

def pc_write_o3d(path, points, colors=None, normals=None, asAscii=True, saveInfloat=True):
    mkdirs(path)
    pcd = o3dPointCloud(points = points, colors = colors, normals = normals)
    o3d.io.write_point_cloud(path, pcd, write_ascii=asAscii)

############
def showMESparseTensor(tensor, convertFromYUV=False, attributeScale=1):
    coord = tensor.C[:,1:4].detach().cpu().numpy()
    feats = tensor.F.detach().cpu().numpy()/attributeScale
    if convertFromYUV:
        feats = convertYUVtoRGB_BT709(feats)
    pcshow(coord,feats)

def o3dPointCloud(points, colors=None, normals=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    return point_cloud

def gen_color(color, num_points=1):
    COLORMAP = {
        'b': [0, 0.4470, 0.7410],
        'o': [0.8500, 0.3250, 0.0980],
        'y': [0.9290, 0.6940, 0.1250],
        'p': [0.4940, 0.1840, 0.5560],
        'g': [0.4660, 0.6740, 0.1880],
        'c': [0.3010, 0.7450, 0.9330],
        'r': [0.6350, 0.0780, 0.1840],
        'k': [0, 0, 0],
        'w': [1, 1, 1]
    }
    if isinstance(color, str):
        c = COLORMAP.get(color, [0, 0, 0])  # default to black if color not found
        return np.tile(c, (num_points, 1)) * 255
    if hasattr(color, 'detach') :
        color = color.detach().cpu().numpy()
    if isinstance(color, list):
        color = np.array(color)
    if isinstance(color, np.ndarray):
        if color.size==num_points:
            c = color.reshape(-1)
            qs = np.diff(np.sort(c)).mean()
            c = np.round(c / qs).astype(int)
            c = c - c.min()
            c = c / c.max()  # normalize to [0, 1]
            viridis = matplotlib.colormaps.get_cmap('gist_ncar')  #plasma  viridis Accent CMRmap gist_ncar
            c = np.array(viridis(c)[:, :3] * 255)
            return c
        color = color.reshape(-1, 3)  
        if color.shape[0] == num_points:
            return color
        if color.shape[0] == 1:
            return np.tile(color, (num_points, 1))
    assert False, 'color should be a string or Nx3 array, or a 1D array of length N'

def pcshow(points, colors=None, normals=None, grid=None, egrid=[2, 2, 2]):
    """
    grid: [n, m, l] for n*m*l grid for show
    """
    if isinstance(points, list):
        points = [x if isinstance(x, np.ndarray) else x.detach().cpu().numpy() for x in points]
        if colors is not None:
            assert isinstance(colors, list) and len(points) == len(colors)
            colors_all = []
            for i,c in enumerate(colors):
                colors_all.append(gen_color(c, points[i].shape[0]))
            colors = np.vstack(colors_all) 
        if normals is not None:
            assert isinstance(normals, list) and len(points) == len(normals)
            normals = np.vstack(normals)
    else:
        points = [points if isinstance(points, np.ndarray) else points.detach().cpu().numpy()]
    points = [x.copy() for x in points]  # make a copy of points to avoid modifying the original data
    if grid is not None:
        assert grid[0]*grid[1]*grid[2] >= len(points), 'grid size should be larger than point clouds number'
        egrid = np.array(egrid)
        mgrid = np.meshgrid(range(grid[0]), range(grid[1]), range(grid[2]))
        for i in range(len(points)):
            tarPos = np.array([mgrid[0].reshape(-1)[i], mgrid[1].reshape(-1)[i], mgrid[2].reshape(-1)[i]]) * egrid
            tempC = points[i].mean()
            points[i] += (tarPos - tempC)
    points = np.vstack(points)
    return pc_show(points, colors=colors, normals=normals)


def pc_show(points, colors=None, normals=None):
    assert points.shape[1] == 3, 'points should be Nx3'
    if colors is not None:
        if isinstance(colors, str) or isinstance(colors, list) or colors.shape != points.shape:
            colors = gen_color(colors, points.shape[0])
        assert colors.shape[0] == points.shape[0], 'colors should match points number'
    point_cloud = o3dPointCloud(points, colors=colors, normals=normals)
    vis = o3d.visualization.draw_geometries([point_cloud])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Open3D Visualizer")
    # vis.add_geometry(point_cloud)
    # # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100))
    # vis.run()
    return vis

def bin2decAry_numpy(x):
    x = np.asarray(x, dtype=np.float64)
    bits = x.shape[1]
    mask = 2 ** np.arange(bits - 1, -1, -1).reshape(1, -1)
    return (x * mask).sum(axis=1)

def compute_morton_code_numpy(points):
    if len(points) == 0:
        return np.array([])
    points = points.astype(np.int64)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ptnum = points.shape[0]
    n = int(np.ceil(np.log2(np.max(points) + 1)))
    morton_code = np.zeros((ptnum, 3 * n), dtype=np.int32)
    for i in range(n):
        morton_code[:, 3 * i] = (z >> i) & 1
        morton_code[:, (3 * i + 1)] = (y >> i) & 1
        morton_code[:, (3 * i + 2)] = (x >> i) & 1 
    morton_code = np.flip(morton_code, axis=1)
    return bin2decAry_numpy(morton_code)

def sortByMorton(points, return_idx=False):
    data = points.copy()
    offset = data[:,0:3].min(0)
    data[:,0:3] -= offset
    morton_codes = compute_morton_code_numpy(data.astype(np.int64))
    idx = np.argsort(morton_codes)
    if return_idx:
        return points[idx],idx
    else:
        return points[idx]

if __name__=='__main__':
    a, b = pc_read("/home/fuchy/workspace/lod_comp/Data/MPEG/MPEGCat3Frame2/Ford_02_q1mm/Ford_02_vox1mm-0100.ply")
    # print(pc_write("test.ply", a, b, asAscii=True, feat_names=['color@1'], feats_types=['f4']))
    # a,b = pc_read("Data/8iVSLF/Static/boxer_viewdep_vox12.ply", feat_names='color')
    # a,b = pc_read_o3d("Data/8iVFBv2/longdress/Ply/longdress_vox10_1051.ply", attribute=True)
    # pc_write_o3d("test_o3d.ply", a, b, asAscii=False)
    # a,b = pc_read("test_o3d.ply", feat_names='color')
    # a,b = pc_read_ply("test_o3d.ply", feat_names='color')
    pc_write("test.ply", a, b, asAscii=True,feat_names=['ref'], feats_types=['f4'])
    pcshow(a,b)