import argparse
import fnmatch
import glob
import itertools
import json
import open3d as o3d
from pathlib import Path
import shutil
import time
import cv2
import matplotlib
import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from termcolor import colored
import imageio
from scipy.spatial.transform import Rotation as R

from matplotlib.image import imsave
import numpy as np
import yaml
import os
from PIL import Image
import concurrent.futures
import utils as api_utils
from utils import exr_loader, exr_saver, normal_to_rgb

RENDERING_PATH=os.path.dirname(os.path.abspath(__file__))
setting=yaml.load(open(os.path.join(RENDERING_PATH,"setting.yaml")),Loader=yaml.FullLoader)

tmp="/home/ubuntu/Users/yong/.clear/bioclear/utils/rendered_output"
# source_path=setting["output_root_path"]+"/00000"
# biodataset_path=setting["output_root_path"]+"/biodataset"
source_path=tmp+"/00000"
biodataset_path=tmp+"/biodataset"
biodataset_camera_normals_path=biodataset_path+"/camera-normals"
biodataset_depth_imgs_rectified_path=biodataset_path+"/depth-imgs-rectified"
biodataset_json_files_path=biodataset_path+"/json-files"
biodataset_outlines_path=biodataset_path+"/outlines"
biodataset_rgb_imgs_path=biodataset_path+"/rgb-imgs"
biodataset_segmentation_masks_path=biodataset_path+"/segmentation-masks"
biodataset_variant_masks_path=biodataset_path+"/variant-masks"

if os.path.exists(biodataset_path):
    shutil.rmtree(biodataset_path)
if not os.path.exists(biodataset_path):
    os.mkdir(biodataset_path)
biodataset_path=tmp+"/biodataset/source-files"
if not os.path.exists(biodataset_path):
    os.mkdir(biodataset_path)
# if not os.path.exists(biodataset_camera_normals_path):
#     os.mkdir(biodataset_camera_normals_path)
# if not os.path.exists(biodataset_depth_imgs_rectified_path):
#     os.mkdir(biodataset_depth_imgs_rectified_path)
# if not os.path.exists(biodataset_outlines_path):
#     os.mkdir(biodataset_outlines_path)
# if not os.path.exists(biodataset_rgb_imgs_path):
#     os.mkdir(biodataset_rgb_imgs_path)
# if not os.path.exists(biodataset_segmentation_masks_path):
#     os.mkdir(biodataset_segmentation_masks_path)
# if not os.path.exists(biodataset_variant_masks_path):
#     os.mkdir(biodataset_variant_masks_path)

class DatasetCreator:
    def __init__(self) -> None:
        pass
    
    def generate_outline_from_depth_normals(self):

        rgb_imgs_path=source_path+"/%09d_color.png"
        depth_imgs_path=source_path+"/%09d_depth_120.exr"
        normals_img_path=source_path+"/%09d_normal_120.exr"
        masks_imgs_path=source_path+"/%09d_mask.png"

        # 直接移动
        save_rgb_imgs_path=biodataset_rgb_imgs_path+"/%09d-rgb.jpg"
        save_depth_imgs_rectified_path=biodataset_depth_imgs_rectified_path+"/%09d-depth-rectified.exr"
        save_camera_normals_path=biodataset_camera_normals_path+"/%09d-camera-normals.exr"
        save_segmentation_masks_path=biodataset_segmentation_masks_path+"/%09d-segmentation-mask.png"
        save_variant_masks_path=biodataset_variant_masks_path+"/%09d-variantMasks.exr"

        save_rgb_imgs_path2=biodataset_path+"/%09d-rgb.jpg"
        save_depth_imgs_rectified_path2=biodataset_path+"/%09d-depth.exr"
        save_camera_normals_path2=biodataset_path+"/%09d-normals.exr"
        save_segmentation_masks_path2=biodataset_path+"/%09d-segmentation-mask.png"
        save_variant_masks_path2=biodataset_path+"/%09d-variantMasks.exr"

        # 需要转换的
        save_depth_edges_path=biodataset_depth_imgs_rectified_path+"/%09d-depth-edges.png"
        save_normal_edges_path=biodataset_camera_normals_path+"/%09d-normals-edges.png"
        save_outlines_path=biodataset_outlines_path+"/%09d-outlines.png"
        save_outlines_viz_path = biodataset_outlines_path+"/%09d-outlines-viz.png"

        save_depth_edges_path2=biodataset_path+"/%09d-depth-edges.png"
        save_normal_edges_path2=biodataset_path+"/%09d-normals-edges.png"
        save_outlines_path2=biodataset_path+"/%09d-outlines.png"
        save_outlines_viz_path2 = biodataset_path+"/%09d-outlines-viz.png"

        height=360
        # height=1080
        width=640
        # width=1920
        
        def outline_from_depth(depth_img_orig):
            '''深度图-->轮廓线'''
            kernel_size=9
            threshold=10
            max_depth_to_object=2.5

            # 应用拉普拉斯滤波器对深度图像进行边缘检测
            depth_img_blur=cv2.GaussianBlur(depth_img_orig,(5,5),0)

            # 将所有大于2.5m的深度值设为0(用于屏蔽边缘矩阵)
            depth_img_mask = depth_img_blur.copy()
            depth_img_mask[depth_img_mask>2.5]=0
            depth_img_mask[depth_img_mask>0]=1

            edges_lap=cv2.Laplacian(depth_img_orig,cv2.CV_64F,ksize=kernel_size,borderType=0)
            edges_lap=(np.absolute(edges_lap).astype(np.uint8))

            edges_lap_binary=np.zeros(edges_lap.shape,dtype=np.uint8)
            edges_lap_binary[depth_img_orig>max_depth_to_object]=0

            return edges_lap_binary

        def outline_from_normal(surface_normal):
            '''表面法线(3*H*W)-->轮廓线'''
            surface_normal=(surface_normal+1)/2 #转换到[0,1]
            surface_normal_rgb16=(surface_normal*65535).astype(np.uint16)
            sobelxy_list=[]
            for surface_normal_gray in surface_normal_rgb16:
                # Sobel过滤器参数
                # 这些参数是通过反复试验选择的。
                # 注意! !sobel输出的最大值随着内核大小的增加呈指数增长。
                # 打印下面数组的最小/最大值，以了解Sobel输出的值范围。
                kernel_size=5
                threshold=60000

                sobelx=cv2.Sobel(surface_normal_gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
                sobely = cv2.Sobel(surface_normal_gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
                print('\ntype0', sobelx.dtype, sobely.dtype)
                print('min', np.amin(sobelx), np.amin(sobely))
                print('max', np.amax(sobelx), np.amax(sobely))

                sobelx=np.abs(sobelx)
                sobely=np.abs(sobely)
                print('\ntype1', sobelx.dtype, sobely.dtype)
                print('min', np.amin(sobelx), np.amin(sobely))
                print('max', np.amax(sobelx), np.amax(sobely))

                # 转换成二进制
                sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
                sobelx_binary[sobelx >= threshold] = True

                sobely_binary = np.full(sobely.shape, False, dtype=bool)
                sobely_binary[sobely >= threshold] = True

                sobelxy_binary = np.logical_or(sobelx_binary, sobely_binary)
                sobelxy_list.append(sobelxy_binary)
            
            sobelxy_binary3d=np.array(sobelxy_list).transpose((1,2,0))
            sobelxy_binary3d=sobelxy_binary3d.astype(np.uint8)*255

            sobelx_binary=np.zeros((surface_normal_rgb16.shape[1],surface_normal_rgb16.shape[2]))

            for channel in sobelxy_list:
                sobelxy_binary[channel>0]=255
            return sobelxy_binary

        def label_to_rgb(label):
            '''输出标签的RGB可视化(轮廓)，假设标签的值为int，最大类数为3
            '''
            rgbArray = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            rgbArray[:, :, 0][label == 0] = 255
            rgbArray[:, :, 1][label == 1] = 255
            rgbArray[:, :, 2][label == 2] = 255

            return rgbArray
        
        for i in range(0,1):
            # 加载深度图像并转化为轮廓线
            print('Loadig img %d' % (i))
            depth_img_orig=exr_loader(depth_imgs_path % (i), ndim=1)
            depth_egdes = outline_from_depth(depth_img_orig)
            depth_edges_img=Image.fromarray(depth_egdes,'L').resize((width,height),resample=Image.NEAREST)

            depth_edges=np.asarray(depth_edges_img)

            # 加载表面图像转换成轮廓线
            surface_normal=exr_loader(normals_img_path % (i))
            normals_edges=outline_from_normal(surface_normal)

            # 保存
            # depth_edges_img.save(save_depth_edges_path % (i))
            # depth_edges_img.save(save_depth_edges_path2 % (i))
            # imsave(save_normal_edges_path % (i), normals_edges)
            # imsave(save_normal_edges_path2 % (i), normals_edges)

            # 深度和轮廓不因该重叠，优先考虑深度
            depth_edges=depth_edges.astype(np.uint8)
            normals_edges[depth_edges==255]=0

            # 修改边缘，创建掩码
            output = np.zeros((height, width), 'uint8')
            output[normals_edges == 255] = 2
            output[depth_edges == 255] = 1

            # 将img的顶部和底部删除渐变条
            num_of_rows_to_delete = 2
            output[:num_of_rows_to_delete, :] = 0
            output[-num_of_rows_to_delete:, :] = 0

            img = Image.fromarray(output, 'L')
            # img.save(save_outlines_path % i)
            # img.save(save_outlines_path2 % i)

            output_color = label_to_rgb(output)

            img = Image.fromarray(output_color, 'RGB')
            # img.save(save_outlines_viz_path % i)
            # img.save(save_outlines_viz_path2 % i)


            # srcfile 需要复制、移动的文件   
            # dstpath 目的地址

            f1=rgb_imgs_path % (i)
            f1=Image.open(f1)
            f1=f1.convert('RGB')
            # f1.save(save_rgb_imgs_path % (i), quality=95)
            f1.save(save_rgb_imgs_path2 % (i), quality=95)
            # shutil.copy(depth_imgs_path % (i), save_depth_imgs_rectified_path % (i))
            shutil.copy(depth_imgs_path % (i), save_depth_imgs_rectified_path2 % (i))
            # shutil.copy(normals_img_path % (i), save_camera_normals_path % (i))
            shutil.copy(normals_img_path % (i), save_camera_normals_path2 % (i))

            # masks_imgs=exr_loader(masks_imgs_path % (i), ndim=1)
            # masks_imgs2=Image.fromarray(masks_imgs,'RGB')
            # masks_imgs2.save(save_segmentation_masks_path % i)
            # masks_imgs2.save(save_segmentation_masks_path2 % i)
            # shutil.copy(masks_imgs_path % (i), save_variant_masks_path % (i))
            shutil.copy(masks_imgs_path % (i), save_segmentation_masks_path2 % (i))
            # shutil.copy(masks_imgs_path % (i), save_variant_masks_path2 % (i))

            # display_output = 1
            # if(display_output):
            #     fig1 = plt.figure(figsize=(12,12))
            #     plt.imshow(depth_img_orig, cmap='gray')
            #     plt.show()
            #     # fig1 = plt.figure(figsize=(12,12))
            #     # plt.imshow(depth_img_blur, cmap='gray')
            #     plt.show()
            #     # fig2 = plt.figure(figsize=(12,12))
            #     # plt.imshow(edges_lap, cmap='gray')
            #     plt.show()
            #     fig3 = plt.figure(figsize=(12,12))
            #     plt.imshow(depth_edges, cmap='gray')
            #     plt.show()
            #     # fig4 = plt.figure(figsize=(12,12))
            #     # plt.imshow(edges, cmap='gray')
            #     plt.show()

    def data_process(self):
        '''
        预处理为表面法线估计和轮廓估计模型提供了数据集。
        |- dataset/
        |--000000000-rgb.jpg
        |--000000000-depth.exr
        |--000000000-normals.exr
        |--000000000-variantMask.exr
        ...
        |--000000001-rgb.jpg
        |--000000001-depth.exr
        |--000000001-normals.exr
        |--000000001-variantMask.exr
        '''

        parser = argparse.ArgumentParser(
        description='Rearrange non-contiguous numbered images in a dataset, move to separate folders and process.')

        parser.add_argument('--p', 
                            default='/home/ubuntu/Users/yong/.clear/bioclear/utils/rendered_output/biodataset/source-files',
                            type=str,
                            help='Path to dataset', metavar='path/to/dataset')
        parser.add_argument('--dst',
                            default='/home/ubuntu/Users/yong/.clear/bioclear/utils/rendered_output/biodataset/',
                            type=str,
                            help='Path to directory of new dataset. Files will moved to and created here.',
                            metavar='path/to/dir')
        parser.add_argument('--num_start',
                            default=0,
                            type=int,
                            help='The initial value from which the numbering of renamed files must start')
        parser.add_argument('--test_set',
                            action='store_true',
                            help='Whether we\'re processing a test set, which has only rgb images, and optionally \
                                depth images.\
                                If this flag is passed, only rgb/depth images are processed, all others are ignored.')
        parser.add_argument('--fov_y',
                            default=0.7428327202796936,
                            type=float,
                            help='Vertical FOV of camera in radians (Field of View along the height of image)')
        parser.add_argument('--fov_x',
                            default=1.2112585306167603,
                            type=float,
                            help='Horizontal FOV of camera in radians (Field of View along the width of image)')
        parser.add_argument('--outline_clipping_h',
                            default=0.015,
                            type=float,
                            help='When creating outlines, height from ground that is marked as contact edge')
        parser.add_argument('--thresh_depth',
                            default=7,
                            type=float,
                            help='When creating outlines, thresh for getting boundary from depth image gradients')
        parser.add_argument('--thresh_mask',
                            default=7,
                            type=float,
                            help='When creating outlines, thresh for getting boundary from masks of objects')
        args = parser.parse_args()

        self.SUBFOLDER_MAP_SYNTHETIC=setting['SUBFOLDER_MAP_SYNTHETIC']
        SUBFOLDER_MAP_REAL=setting['SUBFOLDER_MAP_REAL']
        self.NEW_DATASET_PATHS=setting['NEW_DATASET_PATHS']
        self.NEW_DATASET_PATHS['root']=os.path.expanduser(args.dst)
        if args.test_set:
            self.SUBFOLDER_MAP_SYNTHETIC=SUBFOLDER_MAP_REAL
        
        # 0、检查源文件是否合法
        src_dir_path=self.NEW_DATASET_PATHS['root']
        if not os.path.isdir(src_dir_path):
            if not os.path.isdir(args.p):
                print(colored('ERROR: Did not find {}. Please pass correct path to dataset'.format(args.p), 'red'))
                exit()
            if not os.listdir(args.p):
                print(colored('ERROR: Empty dir {}. Please pass correct path to dataset'.format(args.p), 'red'))
                exit()
        else:
            if ((not os.path.isdir(args.p)) or (not os.listdir(args.p))):
                print(
                    colored(
                        "\nWARNING: Source directory '{}' does not exist or is empty.\
                            \n  However, found dest dir '{}'.\n".format(args.p, src_dir_path), 'red'))
                print(
                    colored(
                        "  Assuming files have already been renamed and moved from Source directory.\
                            \n  Proceeding to process files in Dest dir.", 'red'))
                time.sleep(2)

        # 1、将数据移动到子文件夹
        if not os.path.isdir(src_dir_path):
            os.makedirs(src_dir_path)
            print("\n Created dirs to store new dataset:",src_dir_path)
        else:
            print("\n Dataset dir exists:",src_dir_path)
        
        print("Moving files to",src_dir_path,"and renaming them to start from prefix {:09}.".format(args.num_start))
        count_renamed = self.move_and_rename_dataset(args.p, src_dir_path, int(args.num_start))
        if (count_renamed > 0):
            color = 'green'
        else:
            color = 'red'
        print(colored("Renamed {} files".format(count_renamed), color))

        print("\nSeparating dataset into folders.")
        self.move_to_subfolders(src_dir_path)

        # 2、创建训练数据-相机法线，轮廓，矫正深度
        print('\n\n' + '=' * 20, 'Stage 2 - Create Training Data', '=' * 20)
        if not (args.test_set):
            # 2.1 将世界坐标系下法线转换到相机坐标系下法线==>不需要这一步，已经是相机视角下的法向图？
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     # Get a list of files to process
            #     world_normals_dir = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['world-normals']['folder-name'])
            #     json_files_dir = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['json-files']['folder-name'])

            #     world_normals_files_list = sorted(
            #         glob.glob(os.path.join(world_normals_dir, "*" + self.SUBFOLDER_MAP_SYNTHETIC['world-normals']['postfix'])))
            #     json_files_list = sorted(
            #         glob.glob(os.path.join(json_files_dir, "*" + self.SUBFOLDER_MAP_SYNTHETIC['json-files']['postfix'])))

            #     print("\nConverting World co-ord Normals to Camera co-ord Normals...Check your CPU usage!!")
            #     results = list(
            #         tqdm.tqdm(executor.map(self.preprocess_world_to_camera_normals, world_normals_files_list, json_files_list),
            #                 total=len(json_files_list)))
            #     print(colored('\n  Converted {} world-normals'.format(results.count(True)), 'green'))
            #     print(colored('  Skipped {} world-normals'.format(results.count(False)), 'red'))

            #  2.2 创建矫正的深度
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # 获取待处理的文件列表
                depth_files_dir = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['depth-files']['folder-name'])
                depth_files_list = sorted(
                    glob.glob(os.path.join(depth_files_dir, "*" + self.SUBFOLDER_MAP_SYNTHETIC['depth-files']['postfix'])))

                print("\nRectifiing depth images...")
                
                # 计算cos指标
                depth_img_file_path = depth_files_list[0]
                cos_matrix_y, cos_matrix_x = self.calculate_cos_matrix(depth_img_file_path, args.fov_y, args.fov_x)

                # 应用cos指标到深度图
                results = list(
                    tqdm.tqdm(executor.map(self.create_rectified_depth_image, depth_files_list, itertools.repeat(cos_matrix_y),
                                        itertools.repeat(cos_matrix_x)),
                            total=len(depth_files_list)))
                print(colored('\n  rectified {} depth images'.format(results.count(True)), 'green'))
                print(colored('  Skipped {} depth images'.format(results.count(False)), 'red'))

            # 2.3 创建分割掩码
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Get a list of files to process
                variant_mask_files_dir = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['variant-masks']['folder-name'])
                variant_mask_files_list = sorted(
                    glob.glob(
                        os.path.join(variant_mask_files_dir, "*" + self.SUBFOLDER_MAP_SYNTHETIC['variant-masks']['postfix'])))

                json_files_dir = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['json-files']['folder-name'])
                json_files_list = sorted(
                    glob.glob(os.path.join(json_files_dir, "*" + self.SUBFOLDER_MAP_SYNTHETIC['json-files']['postfix'])))

                print("\ncreating segmentation masks...")
                # Apply Cos matrices to rectify depth
                results = list(
                    tqdm.tqdm(executor.map(self.create_seg_masks, variant_mask_files_list, json_files_list),
                            total=len(variant_mask_files_list)))
                print(colored('\n  created {} segmentation masks'.format(results.count(True)), 'green'))
                print(colored('  Skipped {} images'.format(results.count(False)), 'red'))

            # 2.4 从深度图中创建法线
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Get a list of files to process
                depth_files_dir = os.path.join(src_dir_path,
                                            self.SUBFOLDER_MAP_SYNTHETIC['depth-files-rectified']['folder-name'])
                # camera_normals_dir = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['camera-normals']['folder-name'])

                depth_files_list = sorted(
                    glob.glob(
                        os.path.join(depth_files_dir, "*" + self.SUBFOLDER_MAP_SYNTHETIC['depth-files-rectified']['postfix'])))

                # Applying sobel filter on segmentation masks
                segmentation_mask_dir = os.path.join(src_dir_path,
                                                    self.SUBFOLDER_MAP_SYNTHETIC['segmentation-masks']['folder-name'])
                depth_files_list_mask_data = sorted(
                    glob.glob(
                        os.path.join(segmentation_mask_dir,
                                    "*" + self.SUBFOLDER_MAP_SYNTHETIC['segmentation-masks']['postfix'])))
                # camera_normals_list = sorted(glob.glob(os.path.join(camera_normals_dir,
                #                              "*" + self.SUBFOLDER_MAP_SYNTHETIC['camera-normals']['postfix'])))
                rgb_imgs_path = os.path.join(src_dir_path, self.SUBFOLDER_MAP_SYNTHETIC['rgb-files']['folder-name'])
                image_files_rgb_list = sorted(
                    glob.glob(os.path.join(rgb_imgs_path, "*" + self.SUBFOLDER_MAP_SYNTHETIC['rgb-files']['postfix'])))

                print("\nCreating Outline images...")
                results = list(
                    tqdm.tqdm(executor.map(self.create_outlines_training_data, depth_files_list, variant_mask_files_list,
                                        json_files_list, image_files_rgb_list, itertools.repeat(args.outline_clipping_h),
                                        itertools.repeat(args.thresh_depth), itertools.repeat(args.thresh_mask)),
                            total=len(depth_files_list)))
                print(colored('\n  created {} outlines'.format(results.count(True)), 'green'))
                print(colored('  Skipped {} outlines'.format(results.count(False)), 'red'))
            
    def scene_prefixes(self,dataset_path):
        '''返回数据集中所有rgb文件的前缀列表:named 000000234-rgb.jpb--->000000234'''
        dataset_prefixes = []
        for root, dirs, files in os.walk(dataset_path):
            # one mask json file per scene so we can get the prefixes from them
            rgb_filename = self.SUBFOLDER_MAP_SYNTHETIC['rgb-files']['postfix']
            for filename in fnmatch.filter(files, '*' + rgb_filename):
                dataset_prefixes.append(filename[0:0 - len(rgb_filename)])
            break
        dataset_prefixes.sort()
        print(len(dataset_prefixes))
        unsorted = list(map(lambda x: int(x), dataset_prefixes))
        unsorted.sort()
        return unsorted

    def move_and_rename_dataset(self,old_dataset_path, new_dataset_path, initial_value):
        '''所有文件都被移动到new dir并重命名，以便它们的前缀从提供的初始值开始'''
        sorted_prefixes = self.scene_prefixes(old_dataset_path)

        count_renamed = 0
        for i in range(len(sorted_prefixes)):
            old_prefix_str = "{:09}".format(sorted_prefixes[i])
            new_prefix_str = "{:09}".format(initial_value + i)
            print("\tMoving files with prefix", old_prefix_str, "to", new_prefix_str)

            for root, dirs, files in os.walk(old_dataset_path):
                for filename in fnmatch.filter(files, (old_prefix_str + '*')):
                    shutil.copy(os.path.join(old_dataset_path, filename),
                                os.path.join(new_dataset_path, filename.replace(old_prefix_str, new_prefix_str)))
                    count_renamed += 1
                break

        return count_renamed

    def move_to_subfolders(self,dataset_path):
        '''移动每个文件类型到它自己的子文件夹。'''
        for filetype in self.SUBFOLDER_MAP_SYNTHETIC:
            subfolder_path = os.path.join(dataset_path, self.SUBFOLDER_MAP_SYNTHETIC[filetype]['folder-name'])

            if not os.path.isdir(subfolder_path):
                os.makedirs(subfolder_path)
                print("\tCreated dir:", subfolder_path)
            # else:
            #     print("\tAlready Exists:", subfolder_path)

        for filetype in self.SUBFOLDER_MAP_SYNTHETIC:
            file_postfix = self.SUBFOLDER_MAP_SYNTHETIC[filetype]['postfix']
            subfolder = self.SUBFOLDER_MAP_SYNTHETIC[filetype]['folder-name']

            count_files_moved = 0
            files = os.listdir(dataset_path)
            for filename in fnmatch.filter(files, '*' + file_postfix):
                shutil.move(os.path.join(dataset_path, filename), os.path.join(dataset_path, subfolder))
                count_files_moved += 1
            if count_files_moved > 0:
                color = 'green'
            else:
                color = 'red'
            print("\tMoved", colored(count_files_moved, color), "files to dir:", subfolder)

    def world_to_camera_normals(self,normals_to_convert):
        '''将表面法线数组转换为RGB图像。
            表面法线的范围为(-1,1)，
            这将转换为要写入的(0,255)范围
            变成一个图像。
            表面法线通常在相机坐标内，
            z轴向外。坐标轴是
            映射为(x,y,z) -> (R,G,B)。'''
        camera_normal_rgb = normals_to_convert + 1
        camera_normal_rgb *= 127.5
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
        return camera_normal_rgb
    
    def preprocess_world_to_camera_normals(self,path_world_normals_file, path_json_file):
        '''将法线从世界坐标转换为相机坐标,将创建一个文件夹来存储转换后的文件。四元数，用于将世界法线转换为相机法线,Co-ords从json文件中读取，并与源文件中的每个法线相乘。'''
        #  Output paths and filenames
        camera_normal_dir_path = os.path.join(self.NEW_DATASET_PATHS['root'], self.NEW_DATASET_PATHS['source-files'],
                                            self.SUBFOLDER_MAP_SYNTHETIC['camera-normals']['folder-name'])
        camera_normal_rgb_dir_path = os.path.join(self.NEW_DATASET_PATHS['root'], self.NEW_DATASET_PATHS['source-files'],
                                                self.SUBFOLDER_MAP_SYNTHETIC['camera-normals-rgb']['folder-name'])

        prefix = os.path.basename(path_world_normals_file)[0:0 - len(self.SUBFOLDER_MAP_SYNTHETIC['world-normals']['postfix'])]
        output_camera_normal_filename = (prefix + self.SUBFOLDER_MAP_SYNTHETIC['camera-normals']['postfix'])
        camera_normal_rgb_filename = (prefix + self.SUBFOLDER_MAP_SYNTHETIC['camera-normals-rgb']['postfix'])
        output_camera_normal_file = os.path.join(camera_normal_dir_path, output_camera_normal_filename)
        camera_normal_rgb_file = os.path.join(camera_normal_rgb_dir_path, camera_normal_rgb_filename)

        # If cam normal already exists, skip
        if Path(output_camera_normal_file).is_file():
            return False

        world_normal_file = os.path.join(self.SUBFOLDER_MAP_SYNTHETIC['world-normals']['folder-name'],
                                        os.path.basename(path_world_normals_file))
        camera_normal_file = os.path.join(self.SUBFOLDER_MAP_SYNTHETIC['camera-normals']['folder-name'],
                                        prefix + self.SUBFOLDER_MAP_SYNTHETIC['camera-normals']['postfix'])
        # print("  Converting {} to {}".format(world_normal_file, camera_normal_file))

        # Read EXR File
        exr_np = exr_loader(path_world_normals_file, ndim=3)

        # Read Camera's Inverse Quaternion
        json_file = open(path_json_file)
        data = json.load(json_file)
        inverted_camera_quaternation = np.asarray(data['camera']['world_pose']['rotation']['inverted_quaternion'],
                                                dtype=np.float32)

        # Convert Normals to Camera Space
        camera_normal = self.world_to_camera_normals(inverted_camera_quaternation, exr_np)

        # Output Converted Surface Normal Files
        exr_saver(output_camera_normal_file, camera_normal, ndim=3)

        # Output Converted Normals as RGB images for visualization
        camera_normal_rgb = normal_to_rgb(camera_normal.transpose((1, 2, 0)))
        imageio.imwrite(camera_normal_rgb_file, camera_normal_rgb)

        return True

    def outlines_from_depth(self,depth_img_orig, thresh_depth):
        '''从深度图像创建轮廓
            这用于创建遮挡边界的二进制掩码，也称为深度轮廓。
            轮廓是指深度值突然变化较大的区域，即梯度较大的区域。

            示例:背景下的对象的边界会有很大的梯度，因为深度值从
            背景的对象。
        '''
        # Sobel Filter Params
        # These params were chosen using trial and error.
        # NOTE!!! The max value of sobel output increases exponentially with increase in kernel size.
        # Print the min/max values of array below to get an idea of the range of values in Sobel output.
        kernel_size = 7
        threshold = thresh_depth

        # Apply Sobel Filter
        depth_img_blur = cv2.GaussianBlur(depth_img_orig, (5, 5), 0)
        sobelx = cv2.Sobel(depth_img_blur, cv2.CV_32F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(depth_img_blur, cv2.CV_32F, 0, 1, ksize=kernel_size)

        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)

        # print('minx:', np.amin(sobelx), 'maxx:', np.amax(sobelx))
        # print('miny:', np.amin(sobely), 'maxy:', np.amax(sobely))

        # Create Boolean Mask
        sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
        sobelx_binary[sobelx >= threshold] = True

        sobely_binary = np.full(sobely.shape, False, dtype=bool)
        sobely_binary[sobely >= threshold] = True

        sobel_binary = np.logical_or(sobelx_binary, sobely_binary)

        sobel_result = np.zeros_like(depth_img_orig, dtype=np.uint8)
        sobel_result[sobel_binary] = 255

        # Clean the mask
        kernel = np.ones((3, 3), np.uint8)
        # sobel_result = cv2.erode(sobel_result, kernel, iterations=1)
        # sobel_result = cv2.dilate(sobel_result, kernel, iterations=1)

        # Make all depth values greater than 2.5m as 0
        # This is done because the gradient increases exponentially on far away objects. So pixels near the horizon
        # will create a large zone of depth edges, but we don't want those. We want only edges of objects seen in the scene.
        max_depth_to_object = 2.5
        sobel_result[depth_img_orig > max_depth_to_object] = 0

        return sobel_result

    def outlines_from_masks(self,variant_mask_files, json_files, thresh_mask):
        '''从深度图像创建轮廓
            这用于创建遮挡边界的二进制掩码，也称为深度轮廓。
            轮廓是指深度值突然变化较大的区域，即梯度较大的区域。

            示例:背景下的对象的边界会有很大的梯度，因为深度值从
            背景的对象。
        '''
        json_file = open(json_files)
        data = json.load(json_file)

        variant_mask = exr_loader(variant_mask_files, ndim=1)

        object_id = []
        for key, values in data['variants']['masks_and_poses_by_pixel_value'].items():
            object_id.append(key)

        # create different masks
        final_sobel_result = np.zeros(variant_mask.shape, dtype=np.uint8)
        # create mask for each instance and merge
        for i in range(len(object_id)):
            mask = np.zeros(variant_mask.shape, dtype=np.uint8)
            mask[variant_mask == int(object_id[i])] = 255 #这是干什么

            # Sobel Filter Params
            # These params were chosen using trial and error.
            # NOTE!!! The max value of sobel output increases exponentially with increase in kernel size.
            # Print the min/max values of array below to get an idea of the range of values in Sobel output.
            kernel_size = 7
            threshold = thresh_mask

            # Apply Sobel Filter
            depth_img_blur = cv2.GaussianBlur(mask, (5, 5), 0)
            sobelx = cv2.Sobel(depth_img_blur, cv2.CV_32F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(depth_img_blur, cv2.CV_32F, 0, 1, ksize=kernel_size)

            sobelx = np.abs(sobelx)
            sobely = np.abs(sobely)

            # print('minx:', np.amin(sobelx), 'maxx:', np.amax(sobelx))
            # print('miny:', np.amin(sobely), 'maxy:', np.amax(sobely))

            # Create Boolean Mask
            sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
            sobelx_binary[sobelx >= threshold] = True

            sobely_binary = np.full(sobely.shape, False, dtype=bool)
            sobely_binary[sobely >= threshold] = True

            sobel_binary = np.logical_or(sobelx_binary, sobely_binary)

            sobel_result = np.zeros_like(mask, dtype=np.uint8)
            sobel_result[sobel_binary] = 255

            # Clean the mask
            kernel = np.ones((3, 3), np.uint8)
            sobel_result = cv2.erode(sobel_result, kernel, iterations=2)
            # sobel_result = cv2.dilate(sobel_result, kernel, iterations=1)
            final_sobel_result[sobel_result == 255] = 255

            # cv2.imshow('outlines using mask', sobel_result)
            # cv2.waitKey(0)
        return final_sobel_result

    def create_outlines_training_data(self,path_depth_file,
                                    variant_mask_files,
                                    json_files,
                                    image_files_rgb,
                                    clipping_height=0.03,
                                    thresh_depth=7,
                                    thresh_mask=7):
        '''为轮廓预测模型创建训练数据
            它从深度图像和表面法线图像创建轮廓。
            深度和正常轮廓重叠的地方，优先级给予深度像素。
            期望深度图像为.exr格式，dtype=float32，其中每个像素表示以米为单位的深度
            期望表面法线图像为.exr格式，dtype=float32。每个像素包含
            表面法线，RGB通道映射到XYZ轴。
        '''
        #  Output paths and filenames
        outlines_dir_path = os.path.join(self.NEW_DATASET_PATHS['root'], self.NEW_DATASET_PATHS['source-files'],
                                        self.SUBFOLDER_MAP_SYNTHETIC['outlines']['folder-name'])
        outlines_rgb_dir_path = os.path.join(self.NEW_DATASET_PATHS['root'], self.NEW_DATASET_PATHS['source-files'],
                                            self.SUBFOLDER_MAP_SYNTHETIC['outlines-rgb']['folder-name'])

        prefix = os.path.basename(path_depth_file)[0:0 - len(self.SUBFOLDER_MAP_SYNTHETIC['depth-files-rectified']['postfix'])]
        output_outlines_filename = (prefix + self.SUBFOLDER_MAP_SYNTHETIC['outlines']['postfix'])
        outlines_rgb_filename = (prefix + self.SUBFOLDER_MAP_SYNTHETIC['outlines-rgb']['postfix'])
        output_outlines_file = os.path.join(outlines_dir_path, output_outlines_filename)
        output_outlines_rgb_file = os.path.join(outlines_rgb_dir_path, outlines_rgb_filename)

        # If outlines file already exists, skip
        if Path(output_outlines_file).is_file() and Path(output_outlines_rgb_file).is_file():
            return False

        # Create outlines from depth image
        depth_img_orig = exr_loader(path_depth_file, ndim=1)
        depth_edges = self.outlines_from_depth(depth_img_orig, thresh_depth)

        # Create outlines from surface normals
        # surface_normal = exr_loader(path_camera_normal_file)
        # normals_edges = outline_from_normal(surface_normal)

        # create outlines from segmentation masks
        # seg_mask_img = imageio.imread(path_mask_file)
        mask_edges = self.outlines_from_masks(variant_mask_files, json_files, thresh_mask)

        # outlines_touching_floor = depth_edges_from_mask - depth_edges
        # kernel = np.ones((3, 3), np.uint8)
        # outlines_touching_floor = cv2.erode(outlines_touching_floor, kernel, iterations=2)
        # outlines_touching_floor = cv2.dilate(outlines_touching_floor, kernel, iterations=3)

        # Depth and Normal outlines should not overlap. Priority given to depth.
        # normals_edges[depth_edges == 255] = 0

        # Modified edges and create mask
        assert (depth_edges.shape == mask_edges.shape), " depth and cameral normal shapes are different"

        # height, width = depth_edges.shape
        # output = np.zeros((height, width), 'uint8')
        # output[mask_edges == 255] = 2
        # output[depth_edges == 255] = 1

        combined_outlines = np.zeros((depth_edges.shape[0], depth_edges.shape[1], 3), dtype=np.uint8)
        # combined_outlines[:, :, 0][label == 0] = 255
        combined_outlines[:, :, 1][depth_edges == 255] = 255
        combined_outlines[:, :, 2][mask_edges == 255] = 255

        # Removes extraneous outlines near the border of the image
        # In our outlines image, the borders of the image contain depth and/or surface normal outlines, where there are none
        # The cause is unknown, we remove them by setting all pixels near border to background class.
        # num_of_rows_to_delete_y_axis = 6
        # output[:num_of_rows_to_delete_y_axis, :] = 0
        # output[-num_of_rows_to_delete_y_axis:, :] = 0

        # num_of_rows_to_delete_x_axis = 6
        # output[:, :num_of_rows_to_delete_x_axis] = 0
        # output[:, -num_of_rows_to_delete_x_axis:] = 0

        # cut out depth at the surface of the bottle using point cloud method
        CLIPPING_HEIGHT = clipping_height  # Meters
        IMG_HEIGHT = mask_edges.shape[0]
        IMG_WIDTH = mask_edges.shape[1]

        # Get Rotation Matrix and Euler Angles
        json_f = open(json_files)
        data = json.load(json_f)

        rot_mat_json = data['camera']['world_pose']['matrix_4x4']
        transform_mat = np.zeros((4, 4), dtype=np.float64)
        transform_mat[0, :] = rot_mat_json[0]
        transform_mat[1, :] = rot_mat_json[1]
        transform_mat[2, :] = rot_mat_json[2]
        transform_mat[3, :] = rot_mat_json[3]
        # rot_mat = transform_mat[:3, :3]  # Note that the transformation matrix may be incorrect. Quaternions behaving properly.
        translate_mat = transform_mat[:3, 3]
        translate_mat = np.expand_dims(translate_mat, 1)

        # Get Rot from Quaternions
        quat_mat_json = data['camera']['world_pose']['rotation']['quaternion']
        r = R.from_quat(quat_mat_json)
        rot_euler_xyz = r.as_euler('xyz', degrees=False)
        rot_euler_xyz = np.expand_dims(rot_euler_xyz, 1)

        # Get PointCloud
        # TODO: Calculate the camera intrinsics from image dimensions and fov_x
        # depth_img = api_utils.exr_loader(depth_file, ndim=1)
        rgb_img = imageio.imread(image_files_rgb)
        fx = 1386
        fy = 1386
        cx = 960
        cy = 540
        fx = (float(IMG_WIDTH) / (cx * 2)) * fx
        fy = (float(IMG_HEIGHT) / (cy * 2)) * fy

        xyz_points, rgb_points = api_utils._get_point_cloud(rgb_img, depth_img_orig, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector((rgb_points / 255.0).astype(np.float64))
        # o3d.io.write_point_cloud(pt_file_orig, pcd)

        # Rotate & Translate PointCloud To World Co-Ords
        pcd.rotate(rot_euler_xyz, center=False, type=o3d.RotationType.ZYX)
        pcd.translate(-1 * translate_mat)

        # Color Low Pixels in PointCloud
        rot_xyz_points = np.asarray(pcd.points)
        rot_xyz_points = rot_xyz_points.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        mask_low_pixels = rot_xyz_points[:, :, 2] > (-1 * CLIPPING_HEIGHT)

        # mask = combined_outlines[:, :, 1]
        # mask[mask_low_pixels] = 0
        combined_outlines[mask_low_pixels, 1] = 0
        # increase the width of depth outline
        kernel = np.ones((5, 5), np.uint8)
        # sobel_result = cv2.erode(sobel_result, kernel, iterations=2)
        combined_outlines[:, :, 1] = cv2.dilate(combined_outlines[:, :, 1], kernel, iterations=2)
        combined_outlines[:, :, 2][combined_outlines[:, :, 1] == 255] = 0
        combined_outlines[:, :, 0] = 255
        combined_outlines[:, :, 0][combined_outlines[:, :, 1] == 255] = 0
        combined_outlines[:, :, 0][combined_outlines[:, :, 2] == 255] = 0
        imageio.imwrite(output_outlines_rgb_file, combined_outlines)

        # Save the outlines
        combined_outlines_png = np.zeros((depth_edges.shape[0], depth_edges.shape[1]), dtype=np.uint8)
        combined_outlines_png[combined_outlines[:, :, 1] == 255] = 1
        combined_outlines_png[combined_outlines[:, :, 2] == 255] = 2
        imageio.imwrite(output_outlines_file, combined_outlines_png)

        return True

    def calculate_cos_matrix(self,depth_img_path, fov_y=0.7428327202796936, fov_x=1.2112585306167603):
        '''计算每个像素、相机中心和图像中心之间角度的cos值
            首先，它将取x轴上从图像中心到每个像素的角度，取每个角度的cos并存储
            在矩阵中。然后它会做同样的事情，除了y轴上的角度。
            这些余弦矩阵用于校正深度图像。
        '''
        depth_img = exr_loader(depth_img_path, ndim=1)
        height, width = depth_img.shape
        center_y, center_x = (height / 2), (width / 2)

        angle_per_pixel_along_y = fov_y / height  # angle per pixel along height of the image
        angle_per_pixel_along_x = fov_x / width  # angle per pixel along width of the image

        # create two static arrays to calculate focal angles along x and y axis
        cos_matrix_y = np.zeros((height, width), 'float32')
        cos_matrix_x = np.zeros((height, width), 'float32')

        # calculate cos matrix along y - axis
        for i in range(height):
            for j in range(width):
                angle = abs(center_y - (i)) * angle_per_pixel_along_y
                cos_value = np.cos(angle)
                cos_matrix_y[i][j] = cos_value

        # calculate cos matrix along x-axis
        for i in range(width):
            for j in range(height):
                angle = abs(center_x - (i)) * angle_per_pixel_along_x
                cos_value = np.cos(angle)
                cos_matrix_x[j][i] = cos_value

        return cos_matrix_y, cos_matrix_x

    def create_rectified_depth_image(self,path_rendered_depth_file, cos_matrix_y, cos_matrix_x):
        '''从呈现的深度图像创建并保存校正的深度图像
            渲染的深度图像包含从对象到相机中心/镜头的每个像素的深度。它是得到的
            通过类似于射线追踪的技术。

            然而，我们的算法(比如点云的创建)期望深度图像与输出具有相同的格式
            立体声深度相机。立体相机输出一个深度图像，其中深度是从物体到
            相机平面(平面垂直于相机镜头出来的轴)。因此，如果前面有一堵平墙
            垂直于它的相机，在墙上的每个像素的深度包含相同的深度值。
            这被称为校正深度图像。
        '''
        #  Output paths and filenames
        outlines_dir_path = os.path.join(self.NEW_DATASET_PATHS['root'], self.NEW_DATASET_PATHS['source-files'],
                                        self.SUBFOLDER_MAP_SYNTHETIC['depth-files-rectified']['folder-name'])

        prefix = os.path.basename(path_rendered_depth_file)[0:0 - len(self.SUBFOLDER_MAP_SYNTHETIC['depth-files']['postfix'])]
        output_depth_rectified_filename = (prefix + self.SUBFOLDER_MAP_SYNTHETIC['depth-files-rectified']['postfix'])
        output_depth_rectified_file = os.path.join(outlines_dir_path, output_depth_rectified_filename)

        # If file already exists, skip
        if Path(output_depth_rectified_file).is_file():
            return False

        # calculate modified depth/pixel in mtrs
        depth_img = exr_loader(path_rendered_depth_file, ndim=1)
        output = np.multiply(np.multiply(depth_img, cos_matrix_y), cos_matrix_x)
        output = np.stack((output, output, output), axis=0)

        exr_saver(output_depth_rectified_file, output, ndim=3)

        return True

    def create_seg_masks(self,variant_mask_files, json_files):
        #  Output paths and filenames
        segmentation_dir_path = os.path.join(self.NEW_DATASET_PATHS['root'], self.NEW_DATASET_PATHS['source-files'],
                                            self.SUBFOLDER_MAP_SYNTHETIC['segmentation-masks']['folder-name'])

        prefix = os.path.basename(variant_mask_files)[0:0 - len(self.SUBFOLDER_MAP_SYNTHETIC['variant-masks']['postfix'])]
        segmentation_mask_rectified_filename = (prefix + self.SUBFOLDER_MAP_SYNTHETIC['segmentation-masks']['postfix'])
        segmentation_mask_rectified_file = os.path.join(segmentation_dir_path, segmentation_mask_rectified_filename)

        # If outlines file already exists, skip
        if Path(segmentation_mask_rectified_file).is_file():
            return False

        json_file = open(json_files)
        data = json.load(json_file)

        variant_mask = exr_loader(variant_mask_files, ndim=1)

        object_id = []
        for key, values in data['variants']['masks_and_poses_by_pixel_value'].items():
            object_id.append(key)

        # create different masks
        final_mask = np.zeros(variant_mask.shape, dtype=np.uint8)

        # create mask for each instance and merge
        for i in range(len(object_id)):
            mask = np.zeros(variant_mask.shape, dtype=np.uint8)
            mask[variant_mask == int(object_id[i])] = 255
            final_mask += mask

        imageio.imwrite(segmentation_mask_rectified_file, final_mask)

        return True

if __name__ == '__main__':
    dataset_creator=DatasetCreator()
    dataset_creator.generate_outline_from_depth_normals()
    dataset_creator.data_process()
