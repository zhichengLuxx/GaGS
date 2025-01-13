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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    iteration_test = 40000
    if args.render_type == 'metrics':
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        makedirs(render_path, exist_ok=True)
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        makedirs(gts_path, exist_ok=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            rendering = render(iteration_test, view, gaussians, pipeline, background, args=args)["render"]
            gt = view.original_image[0:3, :, :]
            #rendering = torch.clamp(render(iteration_test, mode, method, view, gaussians, pipeline, background)["render"], 0.0, 1.0)
            #gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    elif args.render_type == 'time':
        render_path = os.path.join(model_path, name, f"ours_{iteration}_TrainingPose_{args.frame_pose}", "renders")
        makedirs(render_path, exist_ok=True)
        time_list = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            start_time = time.time()
            rendering = render(iteration_test, view, gaussians, pipeline, background, args=args)["render"]
            end_time = time.time()
            time_list.append(end_time - start_time)
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        _time = torch.tensor(time_list)
        print(_time.sum())
    elif args.render_type == 'pose':
        render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
        makedirs(render_path, exist_ok=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(iteration_test, view, gaussians, pipeline, background, args=args)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    elif args.render_type == 'time_pose':
        render_path = os.path.join(model_path, name, f"ours_{iteration}_time_pose", "renders")
        makedirs(render_path, exist_ok=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(iteration_test, view, gaussians, pipeline, background, args=args)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        
        gaussians = GaussianModel(dataset.sh_degree)
       
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, other_args=args)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if args.render_type == 'metrics':
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
        elif args.render_type == 'time':
            render_set(dataset.model_path, "time", scene.loaded_iter, scene.getCameras_time_variation(args.frame_pose), gaussians, pipeline, background, args)
        elif args.render_type == 'pose':
            render_set(dataset.model_path, "pose", scene.loaded_iter, scene.getPoseinterpolateCameras(), gaussians, pipeline, background, args)
        elif args.render_type == 'time_pose':
            render_set(dataset.model_path, "time_pose", scene.loaded_iter, scene.getInterpolate_allCameras(), gaussians, pipeline, background, args)
def load_args_from_file(filepath):
    loaded_args = {}
    with open(filepath, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or ':' not in line:
                print(f"Skipping line {line_number}: '{line}' (invalid format)")
                continue  
            try:
                key, value = line.split(': ', 1)
            except ValueError as e:
                print(f"Error parsing line {line_number}: '{line}' - {e}")
                continue  
        
            try:
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                    value = float(value)
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    value = value.strip()
            except ValueError as e:
                print(f"Error converting value on line {line_number}: '{line}' - {e}")
                value = value.strip()
            loaded_args[key] = value
    return loaded_args




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_type", type=str, default='metrics')
    parser.add_argument("--frame_pose", default=-1, type=int)
    args = get_combined_args(parser)
 
    
    loaded_args = load_args_from_file(os.path.join(args.model_path, 'config.txt'))
    for key, value in loaded_args.items():
        if not hasattr(args, key):  
            setattr(args, key, value)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    for iteration in range(39000, 40100, 100):
        try:
            render_sets(model.extract(args), iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
        except Exception as e:
            print(f"Do not find iteration {iteration}")
            continue  
    # render_sets(model.extract(args), 40000, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39900, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39800, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39700, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39600, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39500, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39400, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39300, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39200, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39100, pipeline.extract(args), args.skip_train, args.skip_test, args)
    # render_sets(model.extract(args), 39000, pipeline.extract(args), args.skip_train, args.skip_test, args)