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


def load_config(args):
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)
    return config, CONFIG_FILE_PATH


def create_results_directory(config, CONFIG_FILE_PATH):
    RESULTS_ROOT_DIR = config.resultsDir
    runs = sorted(glob.glob(os.path.join(RESULTS_ROOT_DIR, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(results_dir):
        if len(os.listdir(results_dir)) > 1:
            # Min 1 file always in folder: copy of config file
            results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id + 1))
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)
    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('\nSaving results to folder: ' + termcolor.colored('"{}"\n'.format(results_dir), 'green'))
    return results_dir

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

def read_input_data(config):
    rgb = imread_collection(str(config.files.image))
    depth = imread_collection(str(config.files.depth))
    gt = imread_collection(str(config.files.GT))
    mask = imread_collection(str(config.files.masks))

    print('Total Num of rgb_files:', len(rgb))
    print('Total Num of depth_files:', len(depth))
    print('Total Num of gt_depth_files:', len(gt))
    print('Total Num of segmentation_masks:', len(mask))

    return rgb, depth, gt, mask, len(rgb)

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
    # img[np.isnan(img)] = 0
    # img[np.isinf(img)] = 0
    return img


def main(config, CONFIG_FILE_PATH):
    results_dir = create_results_directory(config, CONFIG_FILE_PATH)

    depthcomplete = initialize_depth_complete(config, results_dir)

    rgb, depth, gt, mask, quantity = read_input_data(config)

    # Create CSV File to store error metrics
    csv_filename = 'computed_errors.csv'
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
    #print(len(rgb))
    for i in range(len(rgb)):
        #print(range(len(rgb)))
        #print(i)
        color_img = format_rgb(rgb[i])
        input_depth = format_depth(depth[i])

        # TOD line 175 in eval is unacceary

        try:
            output_depth, filtered_output_depth = depthcomplete.depth_completion(
                color_img,
                input_depth,
                inertia_weight=float(config.depth2depth.inertia_weight),
                smoothness_weight=float(config.depth2depth.smoothness_weight),
                tangent_weight=float(config.depth2depth.tangent_weight),
                mode_modify_input_depth=config.modifyInputDepth.mode,
                dilate_mask=True)
        except depth_completion_api.DepthCompletionError as e:
            print('Depth Completion Failed:\n  {}\n  ...skipping image {}'.format(e, i))
            continue

        seg_mask = format_mask(mask[i], config)
        depth_gt = format_gt(gt[i], config)
        mask_valid_region = (depth_gt > 0)
        mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
        mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)
        
        metrics = depthcomplete.compute_errors(depth_gt, output_depth, mask_valid_region)

        print('\nImage {:09d} / {}:'.format(i, len(rgb) - 1))
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

        # Save Results of Depth Completion
        error_output_depth, error_filtered_output_depth = depthcomplete.my_store_depth_completion_outputs(
            root_dir=results_dir,
            files_prefix=i,
            min_depth=config.depthVisualization.minDepth,
            max_depth=config.depthVisualization.maxDepth)
        # # print('    Mean Absolute Error in output depth (if Synthetic Data)   = {:.4f} cm'.format(error_output_depth))
        # # print('    Mean Absolute Error in filtered depth (if Synthetic Data) = {:.4f} cm'.format(error_filtered_output_depth))

    # Calculate Mean Errors over entire Dataset
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

def write_config(path, intr):

    with open(path) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
    print(data['depth2depth']['xres'])
    data['depth2depth']['xres'] = float(intr[0])
    data['depth2depth']['yres'] = float(intr[1])
    data['depth2depth']['fx'] = float(intr[2])
    data['depth2depth']['fy'] = float(intr[3])
    data['depth2depth']['cx'] = float(intr[4])
    data['depth2depth']['cy'] = float(intr[5])
    with open(path, 'w') as fp:
        yaml.dump(data, fp)

def eval_script(args):
    #xres, yres, fx, fy, cx,cy
    data = np.array([1280, 720, 909.9260864, 907.9168701, 643.5625, 349.0171814])
    data = data / 2

    path = args.configFile


    for i in range(1,2):
        data *= i
        write_config(path, data)
        print(data)
        config, path = load_config(args)
        #print("start")
        #print(i)
        main(config, path)
        data = data / i
    sys.exit()



if __name__ == '__main__':
    #Parser
    parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    parser.add_argument('-m', '--maskInputDepth', action="store_true", help='Whether we should mask out objects in input depth')
    args = parser.parse_args()

    eval_script(args)
    sys.exit()


    # config, CONFIG_FILE_PATH = load_config(args)


    # main(config, CONFIG_FILE_PATH)