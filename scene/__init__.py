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
from scene.cameras import Camera
import os
import random
import json
import torch
from tqdm import tqdm
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.pose_utils import pose_spherical, render_wander_path
import numpy as np 
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraDynamic_to_JSON
import copy

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], other_args=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse1")):
            scene_info = sceneLoadTypeCallbacks["dynamic_realWorld"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Dynamic Blender Dataset!")
            scene_info = sceneLoadTypeCallbacks["dynamic_synthetic"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "points.npy")):
            print("Found points.npy file, assuming HyperNeRF dataset!")
            scene_info = sceneLoadTypeCallbacks["hypernerf"](args.source_path, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            
            for id, cam in enumerate(camlist):
                json_cams.append(cameraDynamic_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter: # self.loaded_iter = None
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
           
            self.gaussians.load_transform_field_model(os.path.join(self.model_path,
                                                            "transform_field_model",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "transform_field.pth"), other_args)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        model_save_path = os.path.join(self.model_path, "transform_field_model/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_model(os.path.join(model_save_path, "transform_field.pth"))
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getCameras_time_variation(self, frame_pose, scale=1.0, duration=2, fps=30):
        '''
        Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, t=cam_info.t, data_device=args.data_device)
        '''
        cam = self.test_cameras[scale][frame_pose]
        self.Cameras_time_variation = []
        time_list = torch.linspace(0, 1, duration*fps)
        
        for time in time_list:
            cam_time_variation = copy.deepcopy(cam)
            cam_time_variation.t = time

            self.Cameras_time_variation.append(cam_time_variation)
        
        return self.Cameras_time_variation
    
    def getInterpolate_allCameras(self, scale=1.0, frame=150):
        cam = self.train_cameras[scale]
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
        to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        self.Cameras_all_variation = []
        idx = torch.randint(0, len(cam), (1,)).item()
        view = cam[idx]
        for i, pose in enumerate(tqdm(render_poses, desc='get cam ....')):
            fid = i/(frame-1)
            matrix = np.linalg.inv(np.array(pose))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            cam_all_variation = copy.deepcopy(view)
            cam_all_variation.t = fid
            cam_all_variation.reset_extrinsic(R, T)
            self.Cameras_all_variation.append(cam_all_variation)
        return self.Cameras_all_variation
    
    def getPoseinterpolateCameras(self, scale=1.0):
        cam = self.train_cameras[scale]
        frame = 150
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
        to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        self.Cameras_pose_variation = []
        idx = torch.randint(0, len(cam), (1,)).item()
        view = cam[idx]
        for i, pose in enumerate(tqdm(render_poses, desc='get cam ....')):
            fid = 0.5
            matrix = np.linalg.inv(np.array(pose))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            cam_all_variation = copy.deepcopy(view)
            cam_all_variation.t = fid
            cam_all_variation.reset_extrinsic(R, T)
            self.Cameras_pose_variation.append(cam_all_variation)
        return self.Cameras_pose_variation
        
        