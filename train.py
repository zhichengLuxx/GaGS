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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, 
             saving_iterations, checkpoint_iterations, 
             checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, args.expname, args)
    gaussians = GaussianModel(dataset.sh_degree)  # dataset.sh_degree = 3
    scene = Scene(dataset, gaussians)
    gaussians.expname = args.expname
    gaussians.voxelsize = args.voxelsize
    gaussians.training_setup(opt, args)
    

    print(f'Grad threshould is {opt.densify_grad_threshold}')
    print(f'compute_cov3D_python = {pipe.compute_cov3D_python}')

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    add_network_flag = 0
    for iteration in range(first_iter, opt.iterations + 1):        
       
        iter_start.record()

        if iteration > args.start_dynamic_iter and add_network_flag == 0:
            gaussians.optimizer.add_param_group(
                {'params': list(gaussians.transformfield.parameters()), 'lr': args.network_lr_init, "name": "transform_field"}
            )
            add_network_flag = 1
   
        gaussians.update_learning_rate(iteration, args.start_dynamic_iter)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
        

    
        render_pkg = render(iteration, viewpoint_cam, gaussians, pipe, background, args=args)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        psnr_ = psnr(image, gt_image).mean().double()
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 \
             + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) \
             + 0.01 * gaussians.delta_loss
        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          #"dl":f"{gaussians.delta_loss.item()}",
                                          "psnr":f"{psnr_:.{2}f}",
                                          "points":f"{gaussians.get_xyz.shape[0]}",
                                          "vp":f"{gaussians.voxel_points}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
          
            training_report(gaussians, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:         # opt.densify_until_iter = 15000
                # Keep track of max radii in image-space for pruning
                if iteration <= args.start_dynamic_iter or iteration > args.stable_until_iter:
                    
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, iteration, args)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
def prepare_output_and_logger(args, expname, other_args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", expname)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    save_args_to_file(other_args, os.path.join(args.model_path, "config.txt"))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(gaussians, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(iteration, viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def save_args_to_file(args, filepath):
    with open(filepath, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 39_000, 39_100, 39_200, 39_300, 39_400, 39_500, 39_600, 39_700, 39_800, 39_900, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str,default='chicken')
    parser.add_argument("--voxelsize", type=float, default=0.005)
    parser.add_argument("--start_dynamic_iter", type=float, default=3000)
    parser.add_argument("--stable_until_iter", type=float, default=5000)
    parser.add_argument("--network_lr_init", type=float, default=8e-4)
    parser.add_argument("--network_lr_final", type=float, default=1.6e-6)
    parser.add_argument("--time_emb_level", type=float, default=6)
    parser.add_argument("--position_emb_level", type=float, default=10)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    #op.densify_grad_threshold = args.densify_grad_threshold
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG) random number generator
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    
    
    training(lp.extract(args), 
             op.extract(args), 
             pp.extract(args), 
             args.test_iterations,           # [7000, 30000]
             args.save_iterations,           # [7000, 30000, 30000]
             args.checkpoint_iterations,     # []
             args.start_checkpoint,          # None
             args.debug_from,
             args)                # -1

    # All done
    print("\nTraining complete.")
