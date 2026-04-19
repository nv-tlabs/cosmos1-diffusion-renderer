# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import imageio.v3 as imageio
import numpy as np
import torch
try:
    import nvdiffrast.torch as dr
except:
    print('nvdiffrast not found!')
    dr = None


GBUFFER_INDEX_MAPPING = {
    'basecolor':        0,
    'metallic':         1,
    'roughness':        2,
    'normal':           3,
    'depth':            4,
    'diffuse_albedo':   5,
    'specular_albedo':  6,
}


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))


#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


# envmap part
def latlong_vec(res, device=None):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    dir_vec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    # return dr.texture(cubemap[None, ...], dir_vec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]
    return dir_vec #[H, W, 3]


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_z(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, s, 0, 0],
                         [-s, c, 0, 0],
                         [ 0, 0, 1, 0],
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_xyz(rx, ry, rz, device=None):
    """Compose an XYZ rotation matrix from three Euler angles in radians.
    Order is rotate_z @ rotate_y @ rotate_x so axes are applied as
    pitch -> yaw -> roll on the rotated frame, matching Three.js /
    OrbitControls conventions.
    """
    if rx == 0 and ry == 0 and rz == 0:
        return torch.eye(4, dtype=torch.float32, device=device)
    Rx = rotate_x(rx, device=device) if rx != 0 else torch.eye(4, dtype=torch.float32, device=device)
    Ry = rotate_y(ry, device=device) if ry != 0 else torch.eye(4, dtype=torch.float32, device=device)
    Rz = rotate_z(rz, device=device) if rz != 0 else torch.eye(4, dtype=torch.float32, device=device)
    return Rz @ Ry @ Rx


def rotate_latlong(latlong_img, rot_matrix, device=None):
    """Rotate an equirectangular HDR/LDR image (H, W, 3) by a 3x3 (or 4x4)
    rotation matrix. For each output pixel the source direction is computed
    by applying the inverse rotation, and the input is bilinearly resampled.

    This is strictly more general than the horizontal `torch.roll` trick
    used for yaw-only rotations in `load_and_preprocess_hdr`; use it when
    any of the pitch / roll axes are non-zero.
    """
    if dr is None:
        raise RuntimeError("rotate_latlong requires nvdiffrast.torch for lat-long sampling")
    if rot_matrix.shape[-1] == 4:
        R = rot_matrix[:3, :3]
    else:
        R = rot_matrix
    R = R.to(device=device, dtype=torch.float32)

    H, W = latlong_img.shape[:2]
    out_dirs = latlong_vec((H, W), device=device)                  # (H, W, 3)
    # `dir @ R` applies the inverse rotation (R is orthogonal, so R.T == inv).
    src_dirs = out_dirs @ R                                        # (H, W, 3)

    # Convert source directions back to (gx, gy) using latlong_vec's convention:
    #   dir = (sin(theta)*sin(phi), cos(theta), -sin(theta)*cos(phi))
    #   theta = acos(y);   phi = atan2(x, -z)
    # `latlong_vec` maps gy ∈ [0, 1] -> theta * pi and gx ∈ [-1, 1] -> phi * pi.
    x = src_dirs[..., 0]
    y = src_dirs[..., 1].clamp(-1.0, 1.0)
    z = src_dirs[..., 2]
    theta = torch.arccos(y)
    phi = torch.atan2(x, -z)
    gy = theta / np.pi
    gx = phi / np.pi

    texcoord = torch.stack([gx, gy], dim=-1)[None]                 # (1, H, W, 2)
    sampled = dr.texture(
        latlong_img[None].contiguous(), texcoord.contiguous(), filter_mode='linear'
    )[0]
    return sampled


def envmap_vec(res, device=None):
    return -latlong_vec(res, device).flip(0).flip(1) #[H, W, 3]

def envmap_xfm(vec, env_rot, cam_rot):
    # env_rot: envmap rotation
    # cam_rot: camera rotation, camera2world
    vec_inv = vec @ env_rot[:3, :3]
    vec_inv = vec_inv @ cam_rot[:3, :3]
    return vec_inv

def get_ideal_ball(size, flip_x=False):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    # normal_map = normal_map.numpy()
    # mask = mask.numpy()
    return normal_map, mask

def get_ref_vector(normal, incoming_vector):
    #R = 2(N ⋅ I)N - I
    R = 2 * (normal * incoming_vector).sum(-1, keepdims=True) * normal - incoming_vector
    return R

def envmap_chrome_ball(size):
    normal_map, mask = get_ideal_ball(size, flip_x=False)
    vec_ref = get_ref_vector(normal_map, torch.tensor([0, 0, 1], dtype=torch.float32))
    return vec_ref

def luminance(rgb):
    lumi = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return lumi[..., None]

def rgb2srgb(rgb):
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)

def reinhard(x, max_point=16):
    # lumi = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    # lumi = lumi[..., None]
    # y_rein = x * (1 + lumi / (max_point ** 2)) / (1 + lumi)
    # y_rein = x / (1+x)
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein

def cam_intrinsics(fov, width, height, device=None):
    """
    fov is along the height axis
    """
    focal = 0.5 * height / np.tan(0.5 * fov)
    intrinsics = torch.tensor([
        [focal, 0, 0.5 * width],
        [0, focal, 0.5 * height],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    return intrinsics

def uv_mesh(width, height, device=None):
    uv = torch.stack(
        torch.meshgrid(torch.arange(width) + 0.5, torch.arange(height) + 0.5, indexing='xy'), dim=-1
    ).float().to(device)
    uv = torch.cat([uv, torch.ones((height, width, 1), device=device)], dim=-1) # [H, W, 3]
    return uv

def ray2zdepth(ray_depth, width, height, fov=0.82, uv=None, device=None):
    if uv is None:
        uv = uv_mesh(width, height, device=device)

    intrinsics = cam_intrinsics(fov, width, height, device=device)
    ray_dir = uv @ torch.inverse(intrinsics).T
    z_depth = ray_depth * ray_dir[..., 2:3] / torch.norm(ray_dir, dim=-1, keepdim=True) # [H, W, 1]
    return z_depth


def depth2disparity(depth):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    return disparity

def disparity2depth(disparity):
    return depth2disparity(disparity)

def normalize_depth(depth, mask=None, min_percentile=None, max_percentile=None, bg_value=1.0):
    # NOTE: only work on single image, not batch
    depth_m = depth[mask] if mask is not None else depth
    depth_min = depth_m.min() if min_percentile is None else np.percentile(depth_m, min_percentile)
    depth_max = depth_m.max() if max_percentile is None else np.percentile(depth_m, max_percentile)

    depth = (depth - depth_min) / (depth_max - depth_min) # normalize to [0, 1]
    depth = np.clip(depth, 0, 1)
    if mask is not None:
        depth[~mask] = bg_value
    return depth

def read_image(file_name):
    if file_name.endswith('.exr'): # imageio may not properly read exr file
        img = imageio.imread(file_name, flags=cv2.IMREAD_UNCHANGED, plugin='opencv')
    else:
        img = imageio.imread(file_name)
    if img.ndim == 2:
        img = img[..., None]
    return img

def center_crop(img):
    # make sure HWC format
    H, W = img.shape[:2]
    if H == W:
        return img
    elif H > W:
        start = (H - W) // 2
        return img[start:start+W, :, :]
    else:
        start = (W - H) // 2
        return img[:, start:start+H, :]