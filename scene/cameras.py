#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device = "cuda", depth_map=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.depth_map = depth_map

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_matrix = torch.tensor([
            [self.image_width / (2 * np.tan(self.FoVx / 2)), 0, self.image_width / 2],
            [0, self.image_height / (2 * np.tan(self.FoVy / 2)), self.image_height / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.data_device)

    def world_to_camera_coords(self, world_coords):
        """
        Transforms world coordinates to camera coordinates.

        Args:
            world_coords (torch.Tensor): World coordinates [N, 3].

        Returns:
            torch.Tensor: Camera coordinates [N, 3].
        """
        ones = torch.ones((world_coords.shape[0], 1), dtype=world_coords.dtype, device=world_coords.device)
        world_coords_homogeneous = torch.cat((world_coords, ones), dim=1)  # Shape: [N, 4]

        transform_matrix = self.world_view_transform.T[:-1, :]
        coords_expanded = world_coords_homogeneous.unsqueeze(-1)
        camera_coords = torch.matmul(transform_matrix, coords_expanded).squeeze(-1)

        return camera_coords
    
    def camera_to_world_coords(self, camera_coords):
        """
        Transforms camera coordinates to world coordinates.

        Args:
            camera_coords (torch.Tensor): Camera coordinates [N, 3].

        Returns:
            torch.Tensor: World coordinates [N, 3].
        """
        ones = torch.ones((camera_coords.shape[0], 1), dtype=camera_coords.dtype, device=camera_coords.device).cuda()
        camera_coords_homogeneous = torch.cat((camera_coords.cuda(), ones), dim=1).cuda()

        world_coords = []
        for coord in camera_coords_homogeneous:
            world_coord = self.world_view_transform @ coord
            world_coords.append(world_coord[:3])

        world_coords = torch.stack(world_coords)
        return world_coords
        
    def unproject_depth_map_to_point_cloud(self, batch_size=1):
        """
        For each pixel in the depth map, compute the corresponding 3D point in camera coordinates.
        Supports batching to produce point clouds in the shape (N, P, D).

        Args:
            batch_size (int): Number of batches (N) to divide the point cloud into.

        Returns:
            torch.Tensor: Point cloud in camera coordinates [N, P, 3].
        """
        if self.depth_map is None:
            return None

        # Ensure the camera matrix is on the same device as the depth map
        self.camera_matrix = self.camera_matrix.to(self.depth_map.device)

        # Get the height and width of the depth map
        depth_map = self.depth_map.to(torch.float32).mean(dim=-1)
        height, width = depth_map.shape

        # Create a meshgrid of pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=depth_map.device),
            torch.arange(width, dtype=torch.float32, device=depth_map.device),
            indexing='ij'
        )

        # Flatten the meshgrid and depth map for easier processing
        x = x.flatten()  # [N]
        y = y.flatten()  # [N]
        depths = depth_map.flatten()  # [N]

        # Compute the normalized image coordinates
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x_camera = (x - cx) / fx * depths  # [N]
        y_camera = (y - cy) / fy * depths  # [N]
        z_camera = depths  # [N]

        points_in_camera_coords = torch.stack((x_camera, y_camera, z_camera), dim=1)
        points_in_world_coords = self.camera_to_world_coords(points_in_camera_coords)

        # Stack into a point cloud [N, 3]
        point_cloud = points_in_world_coords

        # Reshape the point cloud into batches
        total_points = point_cloud.shape[0]
        points_per_batch = total_points // batch_size

        # Ensure the points divide evenly into batches; if not, pad or truncate
        if total_points % batch_size != 0:
            padding_needed = points_per_batch * batch_size - total_points
            point_cloud = torch.nn.functional.pad(point_cloud, (0, 0, 0, padding_needed))

        # Reshape into (N, P, D)
        batched_point_cloud = point_cloud.view(batch_size, points_per_batch, 3)

        return batched_point_cloud.to('cuda')

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

