#!/usr/bin/env python3
'''Script to run Depth Completion on Synthetic and Real datasets, visualizing the results and computing the error metrics.
This will save all intermediate outputs like surface normals, etc, create a collage of all the inputs and outputs and
create pointclouds from the input, modified input and output depth images.
'''

import argparse
import sys

import yaml
from attrdict import AttrDict #TODO IST SCHEISSE
import glob
import os
import termcolor
import shutil
from skimage.io import imread_collection
import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np
# import ruamel.yaml as ry

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import depth_completion_api
from api import utils as api_utils



def initialize_depth_complete(config, results_dir):
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)

    depthcomplete = depth_completion_api.DepthToDepthCompletion(normalsWeightsFile=config.normals.pathWeightsFile,
                                                                outlinesWeightsFile=config.outlines.pathWeightsFile,
                                                                masksWeightsFile=config.masks.pathWeightsFile,
                                                                normalsModel=config.normals.model,
                                                                outlinesModel=config.outlines.model,
                                                                masksModel=config.masks.model,
                                                                depth2depthExecutable=config.depth2depth.pathExecutable,
                                                                outputImgHeight=outputImgHeight,
                                                                outputImgWidth=outputImgWidth,
                                                                fx=int(config.depth2depth.fx),
                                                                fy=int(config.depth2depth.fy),
                                                                cx=int(config.depth2depth.cx),
                                                                cy=int(config.depth2depth.cy),
                                                                filter_d=config.outputDepthFilter.d,
                                                                filter_sigmaColor=config.outputDepthFilter.sigmaColor,
                                                                filter_sigmaSpace=config.outputDepthFilter.sigmaSpace,
                                                                maskinferenceHeight=config.masks.inferenceHeight,
                                                                maskinferenceWidth=config.masks.inferenceWidth,
                                                                normalsInferenceHeight=config.normals.inferenceHeight,
                                                                normalsInferenceWidth=config.normals.inferenceWidth,
                                                                outlinesInferenceHeight=config.normals.inferenceHeight,
                                                                outlinesInferenceWidth=config.normals.inferenceWidth,
                                                                min_depth=config.depthVisualization.minDepth,
                                                                max_depth=config.depthVisualization.maxDepth,
                                                                tmp_dir=results_dir)

    return depthcomplete

def show_img(img):

    hist = img.flatten()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    #plt.hist(hist, 255)
    plt.show()
    sys.exit()

def format_rgb(img):
    #unint8, flipped
    img = cv2.rotate(img, cv2.ROTATE_180)
    return img

def format_depth(img):
    #float64 to float32, flipped, pixel from mm to m
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = img * 0.001
    img = np.float32(img)
    #img[img > 2] = 2

    return img

def format_mask(img, config):#TOD
    #rotate, rezise, create mask
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.resize(img, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
    img = (img > 0)
    return img

def format_gt(img, config):
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = img * 0.001
    img = np.float32(img)
    img = cv2.resize(img, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
    print(outputImgWidth)
    print(outputImgHeight)
    #img[img>1.5] = 1.5
    # img[np.isnan(img)] = 0
    # img[np.isinf(img)] = 0
    return img

def format_cg_depth(img):
    img = img * 0.001
    img = np.float32(img)
    img = cv2.resize(img, (256, 144), interpolation=cv2.INTER_NEAREST)

    #img[img>1.5] = 1.5
    return img

def main(exp_dir, masked_error=False):

    #CONFIG_FILE_PATH = os.path.join(exp_dir, 'config.yaml')
    CONFIG_FILE_PATH = '/home/dalina/David/Uni/BachelorThesis/cleargrasp/eval_depth_completion/config/config.yaml'
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)
    print(exp_dir)
    results_dir = os.path.join(exp_dir, 'metrics/')
    results_dir = '/home/dalina/David/Uni/BachelorThesis/implicit_depth/src/results/pred_depth_gt_mask/metrics'
    depthcomplete = initialize_depth_complete(config, results_dir)
    results_dir = '/home/dalina/David/Uni/BachelorThesis/implicit_depth/src/results/pred_depth_gt_mask/metrics'

    # depths_dir = os.path.join(exp_dir, '*-output-depth.png')
    # print(depths_dir)
    # gts = imread_collection(str(config.files.GT))
    # masks = imread_collection(str(config.files.masks))
    # depths = imread_collection(depths_dir)

    depths = imread_collection('/home/dalina/David/Uni/BachelorThesis/implicit_depth/src/results/pred_depth_gt_mask/*.png')

    gts = imread_collection('/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/002/gt_background/*.png')
    masks = imread_collection('/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/002/mask/*png')

    # print(depths)
    print('Total Num of depth_files:', len(depths))
    print('Total Num of gt_depth_files:', len(gts))
    print('Total Num of segmentation_masks:', len(masks))


    if masked_error:
        csv_filename = 'masked_input_gt_metrics.csv'
    else:
        csv_filename = 'unmasked_input_gt_metrics.csv'

    field_names = ["Image Num", "RMSE", "REL", "MAE", "Delta 1.25", "Delta 1.25^2", "Delta 1.25^3"]
    with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()

    a1_mean = 0.0
    a2_mean = 0.0
    a3_mean = 0.0
    rmse_mean = 0.0
    abs_rel_mean = 0.0
    mae_mean = 0.0
    sq_rel_mean = 0.0

    for i in range(len(depths)):
        print(range(len(depths)))
        print(i)
        cg_depth = depths[i]
        gt = gts[i]
        mask = masks[i]

        if masked_error:
            seg_mask = format_mask(mask, config)
            depth_gt = format_gt(gt, config)
            mask_valid_region = (depth_gt > 0)
            mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
            mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)

        else:
            depth_gt = format_gt(gt, config)
            mask_valid_region = (depth_gt > 0)
            mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)


        ###cleargrasp format
        cg_depth = format_cg_depth(cg_depth)

        # cv2.imshow('img', depth_gt)
        # cv2.waitKey(0)
        # cv2.imshow('img', cg_depth)
        # cv2.waitKey(0)
        # cv2.imshow('img', mask_valid_region)
        # cv2.waitKey(0)

        metrics = depthcomplete.compute_errors(depth_gt, cg_depth, mask_valid_region)

        print('\nImage {:09d} / {}:'.format(i, len(depths) - 1))
        print('{:>15}:'.format('rmse'), metrics['rmse'])
        print('{:>15}:'.format('abs_rel'), metrics['abs_rel'])
        print('{:>15}:'.format('mae'), metrics['mae'])
        print('{:>15}:'.format('a1.05'), metrics['a1'])
        print('{:>15}:'.format('a1.10'), metrics['a2'])
        print('{:>15}:'.format('a1.25'), metrics['a3'])

        with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
            row_data = [
                i, metrics["rmse"], metrics["abs_rel"], metrics["mae"], metrics["a1"], metrics["a2"], metrics["a3"]
            ]
            writer.writerow(dict(zip(field_names, row_data)))

        a1_mean += metrics['a1']
        a2_mean += metrics['a2']
        a3_mean += metrics['a3']
        rmse_mean += metrics['rmse']
        abs_rel_mean += metrics['abs_rel']
        mae_mean += metrics['mae']
        sq_rel_mean += metrics['sq_rel']

    # Calculate Mean Errors over entire Dataset
    quantity = len(depths)
    print(quantity)
    a1_mean = round(a1_mean / quantity, 2)
    a2_mean = round(a2_mean / quantity, 2)
    a3_mean = round(a3_mean / quantity, 2)
    rmse_mean = round(rmse_mean / quantity, 3)
    abs_rel_mean = round(abs_rel_mean / quantity, 3)
    mae_mean = round(mae_mean / quantity, 3)
    sq_rel_mean = round(sq_rel_mean / quantity, 3)

    print('\n\nMean Error Stats for Entire Dataset:')
    print('{:>15}:'.format('rmse_mean'), rmse_mean)
    print('{:>15}:'.format('abs_rel_mean'), abs_rel_mean)
    print('{:>15}:'.format('mae_mean'), mae_mean)
    print('{:>15}:'.format('a1.05_mean'), a1_mean)
    print('{:>15}:'.format('a1.10_mean'), a2_mean)
    print('{:>15}:'.format('a1.25_mean'), a3_mean)

    # Write the data into a csv file
    with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        row_data = ['MEAN', rmse_mean, abs_rel_mean, mae_mean, a1_mean, a2_mean, a3_mean]
        writer.writerow(dict(zip(field_names, row_data)))

if __name__ == '__main__':
    #Parser
    #exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/014_both_objects/cleargrasp/results/resolution_exp_not_masked/'
    #exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/014_both_objects/cleargrasp/results/resolution exp_masked_out'
    # exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/002/cleargrasp/results/Cleargrasp_res_exp_masked_out'
    #exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/002/cleargrasp/results/Cleargrasp_res_exp_not_masked'
    #exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/014_only_canister/cleargrasp/results'
    exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/002'
    exps = sorted(os.listdir(exps_dir))
    for exp in exps:
        exp_dir = os.path.join(exps_dir, exp)
        exp_dir = exps_dir
        main(exp_dir, masked_error=True)
        main(exp_dir, masked_error=False)
        sys.exit()