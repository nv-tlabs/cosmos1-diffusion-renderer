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

import json
import os

import cv2
import imageio
import imageio.v3 as imageio_v3
import numpy as np
import torch

import nvdiffrast.torch as dr
from . import rendering_utils as util

# Enable OpenEXR support in OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def hdr_mapping(env_hdr, log_scale):
    """Map HDR environment maps to LDR and logarithmic representations."""
    env_ev0 = util.rgb2srgb(util.reinhard(env_hdr, max_point=16).clamp(0, 1))
    env_log = util.rgb2srgb(torch.log1p(env_hdr) / np.log1p(log_scale)).clamp(0, 1)
    return {
        'env_hdr': env_hdr,    # Original HDR image
        'env_ev0': env_ev0,    # LDR image after tone mapping
        'env_log': env_log,    # Logarithmic scaling
    }


def process_environment_map(
    hdr_dir,
    resolution=(512, 512),
    num_frames=1,
    fixed_pose=True,
    pose_file=None,
    pose_offset=0,
    pose_reset=False,
    rotate_envlight=False,
    env_format=['proj', 'fixed', 'ball'],
    log_scale=10000,
    env_strength=1.0,
    env_flip=True,
    env_rot=180.0,
    env_rot_x=0.0,
    env_rot_z=0.0,
    save_dir=None,
    prefix='0000',
    device=None,
):
    """
    Preprocess HDR environment maps for rendering.
    FIXME: Note that this function bakes in a flip and rotate operation for the environment light. Set to env_flip=True and env_rot=180 is considered as loading the original environment map.

    Args:
        hdr_dir (str): Path to the HDR environment map file.
        resolution (tuple of int): Resolution of the output images (H, W).
        num_frames (int): Number of frames to process.
        fixed_pose (bool): Use a fixed camera pose (identity matrix) if True.
        pose_file (str): Path to the camera pose file (JSON).
        pose_offset (int): Offset for the pose frames in the pose file.
        pose_reset (bool): Reset camera poses to be relative to the first frame.
        rotate_envlight (bool): Rotate the environment light over frames if True.
        env_format (list of str): Formats of the environment maps to generate ('proj', 'fixed', 'ball').
        log_scale (int): Log scale factor for HDR mapping.
        env_strength (float): Strength multiplier for the environment map.
        env_flip (bool): Flip the environment map horizontally if True.
        env_rot (float): Yaw rotation of the environment map in degrees (Y axis).
        env_rot_x (float): Pitch rotation of the environment map in degrees (X axis). 0 = no pitch.
        env_rot_z (float): Roll rotation of the environment map in degrees (Z axis). 0 = no roll.
        save_dir (str): Directory to save the processed images (optional).
        prefix (str): Prefix for the output files (used if saving images).

    Returns:
        dict: A dictionary containing the processed environment maps and metadata.
        {
            'metadata': env_meta,
            'fixed': mapping_results_for_fixed_envmap,  # Only if 'fixed' in env_format
            'env_ldr': stacked_tensor_of_proj_env_ldr,  # Only if 'proj' in env_format
            'env_log': stacked_tensor_of_proj_env_log,  # Only if 'proj' in env_format
            'ball_env_ldr': stacked_tensor_of_ball_env_ldr,  # Only if 'ball' in env_format
            'ball_env_log': stacked_tensor_of_ball_env_log,  # Only if 'ball' in env_format
        }
        Tensors are with shape (T, H, W, 3) in [0, 1]
    """
    H, W = resolution
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vec = util.latlong_vec((H, W), device=device)

    # Prepare camera poses
    poses = prepare_camera_poses(
        num_frames=num_frames,
        fixed_pose=fixed_pose,
        pose_file=pose_file,
        pose_offset=pose_offset,
        pose_reset=pose_reset,
        device=device
    )

    # Prepare rotations for the environment light
    rots = np.linspace(0, 2 * np.pi, num_frames) if rotate_envlight else [0] * num_frames

    # Load and preprocess the HDR environment map
    cubemap = load_and_preprocess_hdr(
        hdr_dir=hdr_dir,
        env_strength=env_strength,
        env_flip=env_flip,
        env_rot=env_rot,
        env_rot_x=env_rot_x,
        env_rot_z=env_rot_z,
        device=device
    )

    # Prepare metadata
    env_meta = prepare_metadata(
        hdr_dir=hdr_dir,
        env_rot=env_rot,
        env_flip=env_flip,
        env_strength=env_strength,
        fixed_pose=fixed_pose,
        rotate_envlight=rotate_envlight,
        save_dir=save_dir,
        prefix=prefix,
        env_rot_x=env_rot_x,
        env_rot_z=env_rot_z
    )

    # Initialize result dictionary
    results = {
        'metadata': env_meta,
    }

    # Prepare lists to collect per-frame tensors
    if 'proj' in env_format:
        proj_env_ldr_list = []
        proj_env_log_list = []

    if 'ball' in env_format:
        ball_env_ldr_list = []
        ball_env_log_list = []

    # Process fixed environment map
    if 'fixed' in env_format:
        env_proj = dr.texture(cubemap.unsqueeze(0), -vec.unsqueeze(0).contiguous(),
                              filter_mode='linear', boundary_mode='cube')[0]
        env_proj = torch.flip(env_proj, dims=[0, 1])

        mapping_results = hdr_mapping(env_proj, log_scale=log_scale)
        results['fixed'] = mapping_results

        # Optionally save to disk if save_dir is specified
        if save_dir:
            save_mapping_results(mapping_results, save_dir, prefix, env_prefix='')

    # Prepare vectors for 'ball' format if needed
    if 'ball' in env_format:
        assert H == W, 'Ball environment map requires square resolution.'
        vec_ball, _ = util.get_ideal_ball(H, flip_x=False)
        vec_ref = util.get_ref_vector(vec_ball, np.array([0, 0, 1]))
        vec_ref = vec_ref.float().to(device)

    # Process per-frame environment maps
    for i in range(num_frames):
        frame_prefix = f'{prefix}.{i:04d}'
        c2w = torch.from_numpy(poses[i]).float().to(device)
        y_rot = util.rotate_y(rots[i], device=device)

        if 'proj' in env_format:
            env_proj = process_projected_envmap(cubemap, vec, c2w, y_rot, H, W)
            mapping_results = hdr_mapping(env_proj, log_scale=log_scale)
            proj_env_ldr_list.append(mapping_results['env_ev0'])
            proj_env_log_list.append(mapping_results['env_log'])

            # Optionally save to disk if save_dir is specified
            if save_dir:
                save_mapping_results(mapping_results, save_dir, frame_prefix, env_prefix='')

        if 'ball' in env_format:
            env_proj = process_ball_envmap(cubemap, vec_ref, c2w, y_rot, H, W)
            mapping_results = hdr_mapping(env_proj, log_scale=log_scale)
            ball_env_ldr_list.append(mapping_results['env_ev0'])
            ball_env_log_list.append(mapping_results['env_log'])

            # Optionally save to disk if save_dir is specified
            if save_dir:
                save_mapping_results(mapping_results, save_dir, frame_prefix, env_prefix='ball_')

    # Stack collected tensors along a new dimension (e.g., dim=0)
    if 'proj' in env_format:
        results['env_ldr'] = torch.stack(proj_env_ldr_list, dim=0)
        results['env_log'] = torch.stack(proj_env_log_list, dim=0)

    if 'ball' in env_format:
        results['ball_env_ldr'] = torch.stack(ball_env_ldr_list, dim=0)
        results['ball_env_log'] = torch.stack(ball_env_log_list, dim=0)

    return results


def prepare_camera_poses(num_frames, fixed_pose, pose_file, pose_offset, pose_reset, device):
    """Prepare camera poses based on the provided arguments."""
    if fixed_pose or pose_file is None:
        return [np.eye(4) for _ in range(num_frames)]

    with open(pose_file, 'r') as f:
        meta = json.load(f)
    frames = meta['frames'][pose_offset:pose_offset + num_frames]
    c2w_list = [np.array(frame['transform_matrix']) for frame in frames]

    if pose_reset:
        w2c_0 = np.linalg.inv(c2w_list[0])
        c2w_list = [w2c_0 @ c2w_i for c2w_i in c2w_list]

    return c2w_list


def load_and_preprocess_hdr(hdr_dir, env_strength, env_flip, env_rot, device,
                            env_rot_x=0.0, env_rot_z=0.0):
    """Load and preprocess the HDR environment map.

    Rotation semantics:
      - When only `env_rot` (yaw) is non-zero and `env_rot_x` / `env_rot_z`
        are zero, we keep the fast horizontal pixel-roll for bitwise
        backward compatibility.
      - When either `env_rot_x` (pitch) or `env_rot_z` (roll) is non-zero we
        switch to a full 3D direction-remap via `util.rotate_latlong`. This
        also absorbs `env_rot` so all three axes compose correctly.
    """
    latlong_img = imageio_v3.imread(hdr_dir, flags=cv2.IMREAD_UNCHANGED, plugin='opencv')
    latlong_img = torch.tensor(latlong_img, dtype=torch.float32, device=device)
    latlong_img *= env_strength

    # Cleanup NaNs and Infs
    latlong_img = torch.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0)
    latlong_img = latlong_img.clamp(0.0, 65504.0)

    if env_flip:
        latlong_img = torch.flip(latlong_img, dims=[1])

    needs_3d_rotation = env_rot_x != 0 or env_rot_z != 0
    if needs_3d_rotation:
        R = util.rotate_xyz(
            np.deg2rad(env_rot_x),
            np.deg2rad(env_rot),
            np.deg2rad(env_rot_z),
            device=device,
        )
        latlong_img = util.rotate_latlong(latlong_img, R, device=device)
    elif env_rot != 0:
        lat_h, lat_w = latlong_img.shape[:2]
        pixel_rot = int(lat_w * env_rot / 360)
        latlong_img = torch.roll(latlong_img, shifts=pixel_rot, dims=1)

    # Convert to cubemap
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])
    return cubemap


def prepare_metadata(hdr_dir, env_rot, env_flip, env_strength, fixed_pose,
                     rotate_envlight, save_dir, prefix,
                     env_rot_x=0.0, env_rot_z=0.0):
    """Prepare metadata about the environment map processing."""
    env_meta = {
        'envmap': os.path.basename(hdr_dir),
        'envmap_rot': env_rot,
        'envmap_rot_x': env_rot_x,
        'envmap_rot_z': env_rot_z,
        'envmap_flip': env_flip,
        'envmap_strength': env_strength,
        'fixed_pose': fixed_pose,
        'rotate_envlight': rotate_envlight,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        meta_path = os.path.join(save_dir, f'{prefix}.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(env_meta, f, indent=4)

    return env_meta


def save_mapping_results(mapping_results, save_dir, prefix, env_prefix=''):
    """Save the mapping results to disk."""
    os.makedirs(save_dir, exist_ok=True)
    env_ev0_uint8 = (mapping_results['env_ev0'] * 255).byte().cpu().numpy()
    env_log_uint8 = (mapping_results['env_log'] * 255).byte().cpu().numpy()

    imageio.imwrite(os.path.join(save_dir, f'{prefix}.{env_prefix}env_ldr.png'), env_ev0_uint8)
    imageio.imwrite(os.path.join(save_dir, f'{prefix}.{env_prefix}env_log.png'), env_log_uint8)


def process_projected_envmap(cubemap, vec, c2w, y_rot, H, W):
    """Process the camera-oriented projected environment map."""
    vec_cam = vec.view(-1, 3) @ c2w[:3, :3].T
    vec_query = (vec_cam @ y_rot[:3, :3].T).view(1, H, W, 3)

    env_proj = dr.texture(cubemap.unsqueeze(0), -vec_query.contiguous(),
                          filter_mode='linear', boundary_mode='cube')[0]
    env_proj = torch.flip(env_proj, dims=[0, 1])
    return env_proj


def process_ball_envmap(cubemap, vec_ref, c2w, y_rot, H, W):
    """Process the environment map projected onto a ball."""
    vec_ball = -vec_ref.view(-1, 3) @ c2w[:3, :3].T
    vec_query = (vec_ball @ y_rot[:3, :3].T).view(1, H, W, 3)
    env_proj = dr.texture(cubemap.unsqueeze(0), -vec_query.contiguous(),
                          filter_mode='linear', boundary_mode='cube')[0]
    return env_proj


if __name__ == '__main__':
    """ Example usage when running the script directly

    python utils_env_proj.py \
    --num_frames 24 \
    --hdr_dir <path_to_hdr_file>.hdr \
    --env_rot 180 \
    --env_strength 1.0 \
    --rotate_envlight \
    --fixed_pose \
    --save_dir testing_1_hdriheaven_original_factory_yard_2k

    python utils_env_proj.py \
        --num_frames 24 \
        --hdr_dir <path_to_hdr_file>.hdr \
        --env_rot 180 \
        --env_strength 1.0 \
        --fixed_pose \
        --save_dir testing_2_hdriheaven_original_factory_yard_2k
    """

    import argparse

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Preprocess HDR environment maps for rendering.")
        parser.add_argument('--hdr_dir', type=str, required=True, help="Path to the HDR environment map.")
        parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the processed images.")
        parser.add_argument('--prefix', type=str, default='0000', help="Prefix for the output files.")
        parser.add_argument('--num_frames', type=int, default=1, help="Number of frames to process.")

        parser.add_argument('--pose_file', type=str, default=None, help="Path to the camera pose file (JSON).")
        parser.add_argument('--pose_offset', type=int, default=0, help="Offset for the pose frames.")
        parser.add_argument('--pose_reset', action='store_true', help="Reset camera poses to be relative to the first frame.")

        parser.add_argument('--fixed_pose', action='store_true', help="Use a fixed camera pose (identity matrix).")
        parser.add_argument('--rotate_envlight', action='store_true', help="Rotate the environment light over frames.")

        parser.add_argument('--resolution', nargs=2, type=int, default=[512, 512], help="Resolution of the output images.")
        parser.add_argument('--env_rot', type=float, default=180, help="Rotation angle for the environment map in degrees.")
        parser.add_argument('--env_flip', action='store_true', help="Flip the environment map horizontally.")
        parser.add_argument('--env_strength', type=float, default=1.0, help="Strength multiplier for the environment map.")

        parser.add_argument('--env_format', nargs='+', type=str, default=['proj', 'fixed', 'ball'],
                            choices=['proj', 'fixed', 'ball'], help="Formats of the environment maps to generate.")
        parser.add_argument('--log_scale', type=float, default=10000, help="Log scale factor for HDR mapping.")

        return parser.parse_args()

    args = parse_arguments()
    results = process_environment_map(
        hdr_dir=args.hdr_dir,
        resolution=tuple(args.resolution),
        num_frames=args.num_frames,
        fixed_pose=args.fixed_pose,
        pose_file=args.pose_file,
        pose_offset=args.pose_offset,
        pose_reset=args.pose_reset,
        rotate_envlight=args.rotate_envlight,
        env_format=args.env_format,
        log_scale=args.log_scale,
        env_strength=args.env_strength,
        env_flip=args.env_flip,
        env_rot=args.env_rot,
        save_dir=args.save_dir,
        prefix=args.prefix
    )

