
import argparse
import errno
import glob
import io
import os
import math
import shutil
import imageio
import oyaml
from attrdict import AttrDict
from termcolor import colored
from tensorboardX import SummaryWriter
from imgaug import augmenters as iaa
import imgaug as ia
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import tqdm
from utils.utils import exr_loader
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from model import deeplab
from utils.utils import Utils as u

class SurfaceNormalDataset(Dataset):
    '''
    用于训练表面法线估计模型的数据集类。使用imgaug进行图像增强。
    如果label_dir为空(None， ")，它将假设标签不存在，并返回一个0的张量标签。
'''
    def __init__(self,input_dir,label_dir='',mask_dir='',transform=None,input_only=None):
        super().__init__()
        self.images_dir=input_dir
        self.labels_dir=label_dir
        self.masks_dir=mask_dir
        self.transform=transform
        self.input_only=input_only

        self._datalist_input=[]#所有文件名列表
        self._datalist_label=[]#数据集重所有图片名称
        self._datalist_mask=[]
        self._extension_input=['-transparent-rgb-img.jpg','-rgb.jpg','-input-img.jpg']#输入图片的扩展名称
        self._extension_label=['-cameraNormals.exr','-normals.exr']
        self._extension_mask=['-mask.png']
        self._create_lists_filenames(self.images_dir,self.labels_dir,self.masks_dir)
    
    def __len__(self):
        return len(self._datalist_input)
    
    def __getitem__(self, index):
        image_path=self._datalist_input[index]
        _img=Image.open(image_path).convert("RGB")
        _img=np.array(_img)

        if self.labels_dir:
            label_path=self._datalist_label[index]
            _label=exr_loader(label_path,ndim=3) #（3，H，w)

        if self.masks_dir:
            mask_path=self._datalist_mask[index]
            _mask=imageio.imread(mask_path)
        
        if self.transform:
            det_tf=self.transform.to_deterministic()

            _img=det_tf.augment_image(_img)
            if self.labels_dir:
                #使所有无效像素的-1.0值标记为到0。
                #在原始数据中，无效像素被标记为(-1，-1，-1)，以便在转换为RGB时显示为黑色。
                mask=np.all(_label==-1.0,axis=0)
                _label[:,mask]=0.0

                _label=_label.transpose((1,2,0))#(H,W,3)
                _label=det_tf.augment_image(_label,hooks=ia.HooksImages(activator=self._activator_masks))
                _label=_label.transpose((2,0,1))#(3,H,W)

            if self.masks_dir:
                _mask=det_tf.augment_image(_mask,hooks=ia.HooksImages(activator=self._activator_masks))
        
        _img_tensor=transforms.ToTensor()(_img)

        if self.labels_dir:
            _label_tensor=torch.from_numpy(_label)
            _label_tensor=nn.functional.normalsize(_label_tensor,p=2,dim=0)
        else:
            _label_tensor=torch.zeros((3,_img_tensor.shape[1],_img_tensor.shape[2]),dtype=torch.float32)
        
        if self.masks_dir:
            _mask=_mask[...,np.newaxis]
            _mask_tensor=transforms.ToTensor()(_mask)
        else:
            _mask_tensor=torch.ones((1,_img_tensor.shape[1],_img_tensor.shape[2]),dtype=torch.float32)
        
        return _img_tensor,_label_tensor,_mask_tensor

    def _create_lists_filenames(self,images_dir,labels_dir,masks_dir):
        '''
        在数据集中创建图像和标签的文件名列表索引N处的标签将与索引N处的图像相匹配。
        '''
        assert os.path.isdir(images_dir),'Dataloader given images directory that does not exist: "%s"' %(images_dir)
        for ext in self._extension_input:
            image_search=os.path.join(images_dir,'*'+ext)
            image_paths=sorted(glob.glob(image_search))
            self._datalist_input=self._datalist_input+image_paths
        
        num_images=len(self._datalist_input)
        if num_images==0:
            raise ValueError('No images found in given directory. Searched in dir: {} '.format(images_dir))
        
        if labels_dir:
            assert os.path.isdir(labels_dir),'Dataloader given images directory that does not exist: "%s"' %(labels_dir)

            for ext in self._extension_label:
                label_search=os.path.join(labels_dir,'*'+ext)
                label_paths=sorted(glob.glob(label_search))
                self._datalist_label=self._datalist_label+label_paths
            
            num_labels=len(self._datalist_label)
            if num_labels==0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(image_search))
            if num_images!=num_labels:
                raise ValueError('The number of images of labels do not match. Please check data,'+'found {} images and {} labels in dirs:\n'.format(num_images,num_labels)+'images: {}\nlabels: {}\n'.format(images_dir,labels_dir))
        
        if masks_dir:
            assert os.path.isdir(labels_dir),('Dataloader given images directory that does not exist: "%s"' %(labels_dir))

            for ext in self._extension_mask:
                mask_search=os.path.join(masks_dir,'*'+ext)
                maskpaths=sorted(glob.glob(mask_search))
                self._datalist_mask=self._datalist_mask+maskpaths
            
            num_masks=len(self._datalist_mask)
            if num_masks==0:
                raise ValueError('No masks found in given directory. Searched for {}'.format(image_search))
            if num_images!=num_masks:
                raise ValueError('The number of images and masks do not match. Please check data,' +
                                 'found {} images and {} masks in dirs:\n'.format(num_images, num_masks) +
                                 'images: {}\nmasks: {}\n'.format(images_dir, masks_dir))
    
    def _activator_masks(self,images,augmenter,parents,default):
            '''imgaug一起使用，可以帮助仅对图像应用一些增强，而不是对标签应用'''
            if self.input_only and augmenter.name in self.input_only:
                return False
            else:
                return default

class SurfaceNormalDatasetMatterport(Dataset):
    '''
    用于训练表面法线估计模型的数据集类。使用imgaug进行图像增强。
    如果label_dir为空(None， ")，它将假设标签不存在，并返回一个0的张量标签。
    '''
    def __init__(self,input_dir='/matterport3d/matterport_rgb/v1/scans/',label_dir='/matterport3d/matterport_render_normal',transform=None,input_only=None):
        super().__init__()
        self.images_dir=input_dir
        self.labels_dir=label_dir
        self.transform=transform
        self.input_only=input_only
        self.rgb_folder='matterport_color_images'
        self.normal_folder='mesh_images'

        self._datalist_input=None#所有文件名列表
        self._datalist_label=None#数据集重所有图片名称
        self._extension_input['.jpg']#输入图片的扩展名称
        self._extension_label=['.exr']
        self._create_lists_filenames(self.images_dir)
    
    def __len__(self):
        return len(self._datalist_input)
    
    def __getitem__(self, index):
        '''返回数据集中给定索引处的项。如果没有指定标签目录，则将一个0张量作为标签返回。'''
        image_path=self._datalist_input[index]
        _img=Image.open(image_path).convert("RGB")
        _img=np.array(_img)

        house_id=image_path.split('/')[-4]
        normal_file_folder=os.path.join(self.labels_dir,house_id,self.normal_folder)
        normals=[]
        for (dirpath,dirnames,filenames) in os.walk(normal_file_folder):
            for filename in filenames:
                if (filename.endswith('.png')):
                    if ((filename.split('_')[0] + '_' + 'i' + filename.split('_')[1][1] + '_' + filename.split('_')[2]) == (image_path.split('/')[-1]).split('.')[0]):
                        normals.append(os.path.join(dirpath, filename))
        if len(normals)<3:
            raise ValueError('labels are not present for {}'.format(image_path))
        normals.sort()
        x=imageio.imread(normals[0])
        y=imageio.imread(normals[1])
        z=imageio.imread(normals[2])
        _label=np.stack((x,y,z),axis=2)

        # 应用图像增强并转换成张量
        if self.transform:
            det_tf=self.transform.to_deterministic()
            _img=det_tf.augment_image(_img)

            # # covert normals into an image of dtype float32 in range [0, 1] from range [-1, 1]
                # _label = (_label + 1) / 2
                # _label = _label.transpose((1, 2, 0))  # (H, W, 3)
            if self.labels_dir:
                _label=det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
                _label = _label.transpose((2, 0, 1))  # (3, H, W)
                _label = (_label / 32768) - 1
        
        _img_tensor = transforms.ToTensor()(_img)
        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label).float()
            _label_tensor = nn.functional.normalize(_label_tensor, p=2, dim=0)
        else:
            _label_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        fake_mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor, fake_mask_tensor
    
    def _create_lists_filenames(self, images_dir):
        '''在数据集中创建图像和标签的文件名列表索引N处的标签将与索引N处的图像相匹配。'''
        rgb_list = []
        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        houseid_list = [os.path.join(images_dir, name) for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
        if len(houseid_list) == 0:
            raise ValueError('No house id found in given directory. Searched for {}'.format(images_dir))

        for house in houseid_list:
            rgb_dir = os.path.join(house, os.path.basename(house), self.rgb_folder)
            assert os.path.isdir(rgb_dir), 'Dataloader given images directory that does not exist: "%s"' % (rgb_dir)
            for (dirpath, dirnames, filenames) in os.walk(rgb_dir):
                for filename in filenames:
                    if (filename.endswith(".jpg")):
                        rgb_list.append(os.path.join(rgb_dir, filename))

        self._datalist_input = rgb_list
        
    def _activator_masks(self, images, augmenter, parents, default):
        '''与imgaug一起使用，可以帮助仅对图像应用一些增强，而不是对标签应用模糊只应用于输入，而不是标签。但是，resize对两者都适用。
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

class SurfaceNormalDatasetScannet(Dataset):
    '''
    用于训练表面法线估计模型的数据集类。使用imgaug进行图像增强。
    如果label_dir为空(None， ")，它将假设标签不存在，并返回一个0的张量标签。
    '''
    def __init__(self,input_dir='/scannet-rgb/scans/',label_dir='/scannet_render_normal/',transform=None,input_only=None):
        super().__init__()
        self.images_dir=input_dir
        self.labels_dir=label_dir
        self.transform=transform
        self.input_only=input_only
        self.rgb_folder='color'
        self.normal_folder='mesh_images'

        self._datalist_input=None#所有文件名列表
        self._datalist_label=None#数据集重所有图片名称
        self._extension_input['.jpg']#输入图片的扩展名称
        self._extension_label=['.exr']
        self._create_lists_filenames(self.images_dir,self.labels_dir)
    
    def __len__(self):
        return len(self._datalist_input)
    
    def __getitem__(self, index):
        '''返回数据集中给定索引处的项。如果没有指定标签目录，则将一个0张量作为标签返回。'''
        image_path=self._datalist_input[index]
        _img=Image.open(image_path).convert("RGB")
        _img=np.array(_img)

        scene_id=image_path.split('/')[-3]
        normal_file_folder=os.path.join(self.labels_dir,scene_id,self.normal_folder)
        normals=[]
        for (dirpath,dirnames,filenames) in os.walk(normal_file_folder):
            for filename in filenames:
                if (filename.endswith('.png')):
                    if (filename.split('_')[0]) == image_path.split('/')[-1].split('.')[0].zfill(9):
                        normals.append(os.path.join(dirpath, filename))
        if len(normals)<3:
            raise ValueError('labels are not present for {}'.format(image_path))
        normals.sort()
        x=imageio.imread(normals[0])
        x = ((x / 32768) - 1)
        y=imageio.imread(normals[1])
        y = ((y / 32768) - 1)
        z=imageio.imread(normals[2])
        z = ((z / 32768) - 1)
        _label=np.stack((x,y,z),axis=0)

        # 应用图像增强并转换成张量
        if self.transform:
            det_tf=self.transform.to_deterministic()
            _img=det_tf.augment_image(_img)

            # covert normals into an image of dtype float32 in range [0, 1] from range [-1, 1]
            if self.labels_dir:
                _label=(_label + 1) / 2
                _label = _label.transpose((1, 2, 0))  # (3, H, W)
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

                _label = _label.transpose((2, 0, 1))  # (3, H, W)
                _label = (_label * 2) - 1

        _img_tensor = transforms.ToTensor()(_img)
        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label).float()
        else:
            _label_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        fake_mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor, fake_mask_tensor

    def _create_lists_filenames(self, images_dir, label_dir):
        '''在数据集中创建图像和标签的文件名列表索引N处的标签将与索引N处的图像相匹配。'''
        rgb_list = []
        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        scene_list = [os.path.join(images_dir, name) for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
        if len(scene_list) == 0:
            raise ValueError('No house id found in given directory. Searched for {}'.format(images_dir))

        # 由于normals文件的数量只是rgb文件列表的一小部分，我们迭代得到相应的rgb文件。
        for scene in scene_list:
            final_normal_dir = os.path.join(label_dir, os.path.basename(scene), self.normal_folder)
            assert os.path.isdir(final_normal_dir), 'Dataloader given images directory that does not exist: "%s"' % (final_normal_dir)
            for (dirpath, dirnames, filenames) in os.walk(final_normal_dir):
                for filename in filenames:
                    if (filename.endswith(".png")):
                        file_num = filename.split('_')[0].lstrip('0')
                        if len(file_num) == 0:
                            file_num = '0'
                        rgb_list.append(os.path.join(images_dir, scene, self.rgb_folder, file_num + '.jpg'))

        # to remove duplicate entries
        rgb_list = list(set(rgb_list))
        self._datalist_input = rgb_list
        
    def _activator_masks(self, images, augmenter, parents, default):
        '''与imgaug一起使用，可以帮助仅对图像应用一些增强，而不是对标签应用模糊只应用于输入，而不是标签。但是，resize对两者都适用。
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

class SurfaceNormal:
    def __init__(self) -> None:
        pass
    
    def train(self):
        
        #######################################################
        #########################解析参数###################
        #######################################################
        
        parser=argparse.ArgumentParser(description='Run training of outlines prediction model')
        parser.add_argument('-c','--configFile',required=True,help='Path to config yaml file',metavar='path/to/config')
        args=parser.parse_args()

        # 1.创建文件夹
        CONFIG_FILE_PATH=args.configFile
        with open(CONFIG_FILE_PATH) as fd:
            config_yaml=oyaml.load(fd)
        
        config=AttrDict(config_yaml)

        SUBDIR_RESULT='checkpoints'

        results_root_dir=config.train.logsDir
        runs=sorted(glob.glob(os.path.join(results_root_dir,'exp-*')))
        prev_run_id=int(runs[-1].split('-')[-1]) if runs else 0
        results_dir=os.path.join(results_root_dir,'exp-{:03d}'.format(prev_run_id))
        if os.path.join(results_dir,SUBDIR_RESULT):
            NUM_FILES_IN_EMPTY_FOLDER=0
            if len(os.listdir(os.path.join(results_dir,SUBDIR_RESULT))) > NUM_FILES_IN_EMPTY_FOLDER:
                prev_run_id+=1
                results_dir=os.path.join(results_dir,'exp-{:03d}'.format(prev_run_id))
                os.makedirs(results_dir)
        else:
            os.makedirs(results_dir)
        
        try:
            os.makedirs(os.path.join(results_dir,SUBDIR_RESULT))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        MODEL_LOG_DIR=results_dir
        CHECKPOINT_DIR=os.path.join(MODEL_LOG_DIR,SUBDIR_RESULT)
        shutil.copy2(CONFIG_FILE_PATH,os.path.join(results_dir,'config.yaml'))
        print('Saving results to folder: '+colored('"{}'.format(results_dir),'bule'))

        # 2.创建tensorboard
        writer=SummaryWriter(MODEL_LOG_DIR,commentt='create-graph')

        string_out=io.StringIO()
        oyaml.dump(config_yaml,string_out,default_flow_style=False)
        config_str=string_out.getvalue().split('\n')
        string=''
        for line in config_str:
            string=string+'     '+line+'\n\r'
        writer.add_text('Config',string,global_step=None)
    
        #######################################################
        #########################创建数据集###################
        #######################################################
        
        # 1、创建训练集并使用图像增强
        augs_train=iaa.Sequential([
            iaa.Resize({
                "height": config.imgHeight,
                "width": config.train_imgWidth
            },interpolation='nearest'),#几何增强

            iaa.Sometimes(
                0.1,
                iaa.blend.Alpha(
                    factor=(0.2,0.7),
                    first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5,3.0),per_channel=False),upscale_method='cubic',iterations=(1,2)),name="simplex-blend"
                )
            ),#亮度补丁

            iaa.Sometimes(
                0.3,
                iaa.OneOf([
                    iaa.Add((20,20),per_channel=0.7,name="add"),
                    iaa.Multiply((1.3,1.3),per_channel=0.7,name="mul"),
                    iaa.WithColorspace(to_colorspace="HSV",from_colorspace="RGB",children=iaa.WithChannels(0,iaa.Add((-200,200))),name="hue"),
                    iaa.WithColorspace(to_colorspace="HSV",from_colorspace="RGB",children=iaa.WithChannels(1,iaa.Add((-20,20))),name="sat"),
                    iaa.ContrastNormalization((0.5,1.5),per_channel=0.2,name="norm"),
                    iaa.Grayscale(alpha=(0.0,1.0),name="gray"),
                ])
            ),#颜色空间变化

            iaa.Sometimes(
                0.2,
                iaa.SomeOf((1,None),[
                    iaa.OneOf([iaa.MotionBlur(k=3,name="motion-blur"),iaa.GaussianBlur(sigma=(0.5,1.0),name="gaus-blur")]),
                    iaa.OneOf([iaa.AddElementwise((-5,5),per_channel=0.5,name="add-element"),iaa.MultiplyElementwise((0.95,1.05),per_channel=0.5,name="mul-element"),iaa.AdditiveGaussianNoise(scale=0.01*255,per_channel=0.5,name="guas-noise"),iaa.AdditiveLaplaceNoise(scale=(0,0.01*255),per_channel=True,name="lap-noise"),iaa.Sometimes(1.0,iaa.Dropout(p=(0.003,0.01),per_channel=0.5,name="droupout"))]),
                ],random_order=True)
            ),#模糊和噪声

            iaa.Sometimes(0.2,iaa.CoarseDropout(0.02,size_px=(4,16),per_channel=0.5,name="cdroupout")),#颜色模块
        ])

        input_only=[
            "simplex-blend","bad","mul","hue","sat","norm","gray","motion-blur","gaus-blur","add-element","mul-element","guas-noise","lap-noise","dropout","cdropout"
        ]

        db_synthetic_lst=[]

        if config.train.datasetsTrain is not None:
            for dataset in config.train.datasetsTrain:
                if dataset.images:
                    db_synthetic=SurfaceNormalDataset(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_train,input_only=input_only)
                    db_synthetic_lst.append(db_synthetic)
                db_synthetic=torch.utils.data.ConcatDataset(db_synthetic_lst)

        #注意:如果传递了多个用于matterport/scannet的文件夹，则只会接收最后一个。
        if config.train.datasetsMatterportTrain is not None:
            for dataset in config.train.datasetsMatterportTrain:
                if dataset.images:
                    db_matterport=SurfaceNormalDatasetMatterport(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_train,input_only=input_only)
                    print('Num of Matterport imgs:', len(db_matterport))
        
        if config.train.datasetsScannetTrain is not None:
            for dataset in config.train.datasetsScannetTrain:
                if dataset.images:
                    db_scannet=SurfaceNormalDatasetScannet(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_train,input_only=input_only)
                    print('Num of ScanNet imgs:', len(db_scannet))
        
        # 2、创建验证集
        augs_test=iaa.Sequential([
            iaa.Resize({
                "height": config.train.imgHeight,
                "width": config.train.imgWidth
            }, interpolation='nearest'),
        ])
        db_val_list=[]
        if config.train.datasetsVal is not None:
            for dataset in config.train.datasetsVal:
                if dataset.images:
                    db=SurfaceNormalDataset(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_test,input_only=None)
                    train_size=int(config.train.percentageDataForValidation*len(db))
                    db=torch.utils.data.Subset(db,range(train_size))
                    db_val_list.append(db)
        
        if config.train.datasetsMatterportVal is not None:
            for dataset in config.train.datasetsMatterportVal:
                if dataset.images:
                    db=SurfaceNormalDatasetMatterport(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_test,input_only=None)
                    train_size=int(config.train.percentageDataForMatterportVal*len(db))
                    db=torch.utils.data.Subset(db,range(train_size))
                    db_val_list.append(db)

        if config.train.datasetsScannetVal is not None:
            for dataset in config.train.datasetsScannetVal:
                if dataset.images:
                    db=SurfaceNormalDatasetMatterport(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_test,input_only=None)
                    train_size=int(config.train.percentageDataForScannettVal*len(db))
                    db=torch.utils.data.Subset(db,range(train_size))
                    db_val_list.append(db)

        if db_val_list:
            db_val = torch.utils.data.ConcatDataset(db_val_list)
        
        # 3 创建测试集
        db_test_list=[]
        if config.train.datasetsTestReal is not None:
            for dataset in config.train.datasetsTestReal:
                if dataset.images:
                    mask_dir=dataset.masks if hasattr(dataset,'mask') and dataset.mask else ''
                    db=SurfaceNormalDataset(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_test,input_only=None)
                    db_test_list.append(db)
            if db_test_list:
                db_test = torch.utils.data.ConcatDataset(db_test_list)

        db_test_synthetic_list=[]
        if config.train.datasetsTestSynthetic is not None:
            for dataset in config.train.datasetsTestSynthetic:
                if dataset.images:
                    db=SurfaceNormalDataset(input_dir=dataset.images,label_dir=dataset.labels,transform=augs_test,input_only=None)
                    db_test_synthetic_list.append(db)
            if db_test_synthetic_list:
                db_test_synthetic = torch.utils.data.ConcatDataset(db_test_synthetic_list)

        # 4 创建dataloader
        if db_val_list:
            assert (config.train.validationBatchSize <= len(db_val)), \
                ('validationBatchSize ({}) cannot be more than the ' +
                'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val))

            validationLoader=DataLoader(db_val,batch_size=config.train.validationBatchSize,shuffle=True,num_workers=config.train.numWorkers,drop_last=False)

        if db_test_list:
            assert (config.train.testBatchSize <= len(db_test)), \
                ('testBatchSize ({}) cannot be more than the ' +
                'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test))

            testLoader = DataLoader(db_test,
                                    batch_size=config.train.testBatchSize,
                                    shuffle=False,
                                    num_workers=config.train.numWorkers,
                                    drop_last=True)
        
        if db_test_synthetic_list:
            assert (config.train.testBatchSize <= len(db_test_synthetic)), \
                ('testBatchSize ({}) cannot be more than the ' +
                'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test_synthetic_list))

            testSyntheticLoader = DataLoader(db_test_synthetic,
                                            batch_size=config.train.testBatchSize,
                                            shuffle=True,
                                            num_workers=config.train.numWorkers,
                                            drop_last=True)

        # =======================================================
        # #######################BUILD MODEL#######################
        # =======================================================
        if config.train.model=='deeplab_xception':
            model=deeplab.DeepLab(num_classes=config.train.numClasses,backone='xception',sync_bn=True,freeze_bn=False)
        elif config.train.model=='deeplab_resnet':
            model=deeplab.DeepLab(num_classes=config.train.numClasses,backone='resnet',sync_bn=True,freeze_bn=False)
        elif config.train.model == 'drn':
            model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='drn', sync_bn=True,
                                    freeze_bn=False)
        else:
            raise ValueError('Invalid model "{}" in config file. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet"]'.format(config.train.model)) 
    
        #######################################################
        #########################设置###################
        #######################################################
        
        # 1 继续训练
        print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
        if not os.path.isfile(config.train.pathPrevCheckpoint):
            raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(config.train.pathPrevCheckpoint))
        CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')
        if 'model_state_dict' in CHECKPOINT:
            model.looad_state_dict(CHECKPOINT['model_state_dict'])
        elif 'state_dict' in CHECKPOINT:
            CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
            CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
            model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
        else:
            model.load_state_dict(CHECKPOINT)

        # 2 多GPU训练
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 3 设置优化器
        if config.train.model=='unet':
            optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config.train.optimAdam.learningRate),
                                 weight_decay=float(config.train.optimAdam.weightDecay))
        elif config.train.model=='deeplab_xception' or config.train.model=='deeplab_resnet' or config.train.model=='drn':
            optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(config.train.optimSgd.learningRate),
                                momentum=float(config.train.optimSgd.momentum),
                                weight_decay=float(config.train.optimSgd.weight_decay))
        elif config.train.model == 'refinenet' or config.train.model == 'densenet':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=float(config.train.optimSgd.learningRate),
                                        momentum=float(config.train.optimSgd.momentum),
                                        weight_decay=float(config.train.optimSgd.weight_decay))
        else:
            raise ValueError(
                'Invalid model "{}" in config file for optimizer. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet"]'
                .format(config.train.model))
        
        if not config.train.lrScheduler:
            pass
        elif config.train.lrScheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.train.lrSchedulerStep.step_size,gamma=config.train.lrScheduler.Step.gamma)
        elif config.train.lrScheduler=='ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=float(config.train.lrSchedulerPlateau.factor),patience=config.train.lrSchedulerPlateau.patience,verbose=True)
        elif config.train.lrScheduler=='lr_poly':
            pass
        else:
            raise ValueError("Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(config.train.lrScheduler))

        # 4 继续训练
        if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
            if 'optimizer_state_dict' in CHECKPOINT:
                optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
            else:
               print(colored('WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' + '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red')) 
        
        # 5 损失函数
        if config.train.lossFunc=='cosine':
            criterion=self._loss_fn_cosine
        elif config.train.lossFunc=='radians':
            criterion=self._loss_fn_radians
        else:
            raise ValueError("Invalid lossFunc from config file. Can only be ['cosine', 'radians']. " + "Value passed is: {}".format(config.train.lossFunc))
    
        #######################################################
        #########################开始训练###################
        #######################################################
        total_iter_num=0
        START_EPOCH=0
        END_EPOCH=config.train.numEpochs

        if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
            if 'optimizer_state_dict' in CHECKPOINT:
                total_iter_num=CHECKPOINT['total_iter_num']+1
                START_EPOCH=CHECKPOINT['epoch']+1
                END_EPOCH=CHECKPOINT['epoch']+config.train.numEpochs
            else:
                print(colored('Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.Starting from epoch num 0', 'red'))

        for epoch in range(START_EPOCH,END_EPOCH):
            print('\n\nEpoch {}/{}'.format(epoch,END_EPOCH-1))
            print('-'*30)

            writer.add_scalar('data/Epoch Number',epoch,total_iter_num)

            print('Train:')
            print('='*10)

            if config.train.lrScheduler=='StepLR':
                lr_scheduler.step()
            elif config.train.lrScheduler=='ReduceLROnPlateau':
                lr_scheduler.step(epoch_loss)
            elif config.train.lrScheduler=='lr_poly':
                if epoch % config.train.epochSize == config.train.epochSize-1:
                    lr_=u.lr_poly(float(config.train.optimSgd.learningRate),int(epoch-START_EPOCH),int(END_EPOCH-START_EPOCH),0.9)

                    train_params=model.parameters()
                    if model == 'drn':
                        train_params=[{'params': model.get_1x_lr_params(), 'lr': config.train.optimSgd.learningRate},{'params':model.get_10x_lr_params(),'lr':config.train.optimSgd.learningRate*10}]
                    optimizer=torch.optim.SGD(train_params,lr=lr_,momentum=float(config.train.optimSgd.momentum),weight_decay=float(config.train.optimSgd.weight_decay))
            
            # 随机分割scannet 和 matterport
            db_train_list=[]
            if config.train.datasetsTrain is not None:
                if config.train.datasetsTrain[0].images:
                    train_size_synthetic=int(config.train.percentageDataForTraining * len(db_synthetic))
                    db,_=torch.utils.data.random_split(db_synthetic,train_size_synthetic,len((db_synthetic)-train_size_synthetic))
                    db_train_list.append(db)

            if config.train.batchSizeMatterport==0 and config.train.batchSizeScannet == 0:
                if config.train.datasetsMatterportTrain is not None:
                    if config.train.datasetsMatterportTrain[0].images:
                        train_size_matterport = int(config.train.percentageDataForMatterportTraining * len(db_matterport))
                        db, _ = torch.utils.data.random_split(
                            db_matterport, (train_size_matterport, len(db_matterport) - train_size_matterport))
                        db_train_list.append(db)
                if config.train.datasetsScannetTrain is not None:
                    if config.train.datasetsScannetTrain[0].images:
                        train_size_scannet = int(config.train.percentageDataForScannetTraining * len(db_scannet))
                        db, _ = torch.utils.data.random_split(db_scannet,(train_size_scannet, len(db_scannet) - train_size_scannet))
                        db_train_list.append(db)
            else:
                # 为matterport和scannet创建单独的加载器。将手动调用它们并添加到批处理中。
                if config.train.datasetsMatterportTrain is not None:
                    trainLoaderMatterport=DataLoader(db_matterport,batch_size=config.train.batchSizeMatterport,shuffle=True,num_workers=2,drop_last=True,pin_memory=True)
                    loaderMatterport = iter(trainLoaderMatterport)
                if config.train.datasetsScannetTrain is not None:
                    trainLoaderScannet=DataLoader(db_scannet,batch_size=config.train.batchSizeScannet,shuffle=True,num_workers=2,drop_last=True,pin_memory=True)
                    loaderScannet = iter(trainLoaderScannet) 
            db_train=torch.utils.data.ConcatDataset(db_train_list)
            trainLoader=DataLoader(db_train,batch_size=config.train.batchSize,shuffle=True,num_workers=config.train.numWorkes,drop_last=True,pin_memory=True)

            model.train()

            running_loss=0.0
            runninng_mean=0.0
            running_median=0.0
            for iter_num,batch in enumerate(tqdm(trainLoader)):
                total_iter_num+=1
                # 获取数据
                if config.train.batchSizeMatterport==0 and config.train.batchSizeScannet==0:
                     inputs,labels,masks=batch
                else:
                    inputs_t,labels_t,mask_t=batch
                    inputs_m,labels_m=next(loaderMatterport)
                    inputs_s,labels_s=next(loaderScannet)
                    inputs=torch.cat([inputs_t,inputs_m,inputs_s],dim=0)
                    labels=torch.cat([labels_t,labels_m,labels_s],dim=0)
                    masks=torch.cat([mask_t,torch.ones_like(mask_t),torch.ones_like(mask_t)],dim=0)

                if config.train.model=='refinenet':
                    labels_resized=self._resize_tensor(labels,int(labels.shape[2]/4),int(labels[3]/4))
                    labels_resized=labels_resized.to(device)
                if config.train.model=='densenet':
                    labels_resized=self._resize_tensor(labels,int(labels.shape[2]/2),int(labels[3]/2))
                    labels_resized=labels_resized.to(device)

                inputs=inputs.to(device)
                labels=labels.to(device)

                # 获得模型Graph
                if epoch==0 and iter_num==0:
                    writer.add_graph(model,inputs,False)

                # 前向传递和反向传播
                optimizer.zero_grad()
                torch.set_grad_enabled(True)
                normal_vectors=model(inputs)
                normal_vectors_norm=nn.functional.normalize(normal_vectors.double(),p=2,dim=1)

                if config.train.model=='unet':
                    loss=criterion(normal_vectors_norm, labels, reduction='sum')
                elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                    loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                    loss /= config.train.batchSize
                elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                    loss = criterion(normal_vectors_norm, labels_resized, reduction='sum')
                    loss /= config.train.batchSize
                loss.backward()
                optimizer.step()

                normal_vectors_norm = normal_vectors_norm.detach().cpu()
                inputs = inputs.detach().cpu()
                labels = labels.detach().cpu()
                mask_tensor = masks.squeeze(1)

                loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = self._metric_calculator_batch(normal_vectors_norm, labels.double(), mask_tensor)
                running_mean += loss_deg_mean.item()
                running_median += loss_deg_median.item()

                # statistics
                running_loss += loss.item()
                writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)
                writer.add_scalar('data/Train Mean Error (deg)', loss_deg_mean.item(), total_iter_num)
                writer.add_scalar('data/Train Median Error (deg)', loss_deg_median.item(), total_iter_num)

                if (iter_num % config.train.saveImageIntervalIter) == 0:
                    if config.train.model == 'refinenet':
                        output = self._resize_tensor(normal_vectors_norm, int(normal_vectors_norm.shape[2] * 4),
                                            int(normal_vectors_norm.shape[3] * 4))
                    elif config.train.model == 'densenet':
                        output = self._resize_tensor(normal_vectors_norm, int(normal_vectors_norm.shape[2] * 2),
                                            int(normal_vectors_norm.shape[3] * 2))
                    else:
                        output = normal_vectors_norm

                    grid_image = u.create_grid_image(inputs, output.float(), labels, max_num_images_to_save=16)
                    writer.add_image('Train', grid_image, total_iter_num)
            # 记录
            num_samples = (len(trainLoader))
            epoch_loss = running_loss / num_samples
            writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
            print('Train Epoch Loss: {:.4f}'.format(epoch_loss))
            epoch_mean = running_mean / num_samples
            epoch_median = running_median / num_samples
            print('Train Epoch Mean Error (deg): {:.4f}'.format(epoch_mean))
            print('Train Epoch Median Error (deg): {:.4f}'.format(epoch_median))

            current_learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_learning_rate, total_iter_num)

            if (epoch % config.train.saveImageInterval) == 0:
                if config.train.model == 'refinenet':
                    output = self._resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 4),
                                        int(normal_vectors_norm.shape[3] * 4))
                elif config.train.model == 'densenet':
                    output = self._resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 2),
                                        int(normal_vectors_norm.shape[3] * 2))
                else:
                    output = normal_vectors_norm.detach().cpu()

                grid_image = u.create_grid_image(inputs.detach().cpu(),
                                                    output.float(),
                                                    labels.detach().cpu(),
                                                    max_num_images_to_save=16)
                writer.add_image('Train', grid_image, total_iter_num)

            # 保存权重
            if (epoch % config.train.saveModelInterval) == 0:
                filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
                if torch.cuda.device_count() > 1:
                    model_params = model.module.state_dict() 
                else:
                    model_params = model.state_dict()

                torch.save(
                    {
                        'model_state_dict': model_params,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'total_iter_num': total_iter_num,
                        'epoch_loss': epoch_loss,
                        'config': config_yaml
                    }, filename)

            #######################################################
            #########################开始验证###################
            #######################################################
            if db_val_list:
                print('\nValidation:')
                print('=' * 10)

                model.eval()

                running_loss = 0.0
                running_mean = 0
                running_median = 0
                for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
                    inputs, labels, masks = sample_batched

                    # Forward pass of the mini-batch
                    if config.train.model == 'refinenet':
                        labels_resized = self._resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                        labels_resized = labels_resized.to(device)
                    if config.train.model == 'densenet':
                        labels_resized = self._resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                        labels_resized = labels_resized.to(device)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        normal_vectors = model(inputs)

                    normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

                    if config.train.model == 'unet':
                        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                    elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                        loss /= config.train.batchSize
                    elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                        normal_vectors_norm = normal_vectors_norm.detach().cpu()
                        labels_resized = labels_resized.detach().cpu()
                        # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
                        loss = criterion(normal_vectors_norm, labels_resized, reduction='sum')
                        loss /= config.train.batchSize

                    running_loss += loss.item()

                    loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = self._metric_calculator_batch(
                        normal_vectors_norm, labels.double())
                    running_mean += loss_deg_mean.item()
                    running_median += loss_deg_median.item()

                # Log Epoch Loss
                num_samples = (len(validationLoader))
                epoch_loss = running_loss / num_samples
                writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
                print('Validation Epoch Loss: {:.4f}'.format(epoch_loss))
                epoch_mean = running_mean / num_samples
                epoch_median = running_median / num_samples
                print('Val Epoch Mean: {:.4f}'.format(epoch_mean))
                print('Val Epoch Median: {:.4f}'.format(epoch_median))
                writer.add_scalar('data/Val Epoch Mean Error (deg)', epoch_mean, total_iter_num)
                writer.add_scalar('data/Val Epoch Median Error (deg)', epoch_median, total_iter_num)

                # Log 10 images every N epochs
                if (epoch % config.train.saveImageInterval) == 0:
                    if config.train.model == 'refinenet':
                        output = self._resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 4),
                                            int(normal_vectors_norm.shape[3] * 4))
                    elif config.train.model == 'densenet':
                        output = self._resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 2),
                                            int(normal_vectors_norm.shape[3] * 2))
                    else:
                        output = normal_vectors_norm.detach().cpu()
                    grid_image = u.create_grid_image(inputs.detach().cpu(),
                                                        output.float(),
                                                        labels.detach().cpu(),
                                                        max_num_images_to_save=10)
                    writer.add_image('Validation', grid_image, total_iter_num)

            #######################################################
            #########################开始验证###################
            #######################################################
            if db_test_list:
                print('\nTesting:')
                print('=' * 10)

                model.eval()

                running_mean = 0
                running_median = 0
                running_d1 = 0
                img_tensor_list = []
                output_tensor_list = []
                label_tensor_list = []
                for iter_num, sample_batched in enumerate(tqdm(testLoader)):
                    inputs, labels, masks = sample_batched

                    # Forward pass of the mini-batch
                    if config.train.model == 'refinenet':
                        labels_resized = self._resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                        labels_resized = labels_resized.to(device)
                    if config.train.model == 'densenet':
                        labels_resized = self._resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                        labels_resized = labels_resized.to(device)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        normal_vectors = model(inputs)

                    normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)

                    # Save output images, one at a time, to results
                    img_tensor = inputs.detach().cpu()
                    output_tensor = normal_vectors_norm.detach().cpu()
                    label_tensor = labels.detach().cpu()
                    mask_tensor = masks.squeeze(1)

                    img_tensor_list.append(img_tensor)
                    output_tensor_list.append(output_tensor)
                    label_tensor_list.append(label_tensor)

                    for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor, mask_tensor)):
                        img, output, label, mask = sample_batched
                        # Calc metrics
                        loss_deg_mean, loss_deg_median, percentage_1, _, _, _ = self._metric_calculator(output,
                                                                                                                label,
                                                                                                                mask=mask)
                        running_mean += loss_deg_mean.item()
                        running_median += loss_deg_median.item()
                        running_d1 += percentage_1.item()

                    # statistics
                    running_loss += loss.item()
                    writer.add_scalar('data/Test Real Mean Error (deg)', loss_deg_mean.item(), total_iter_num)
                    writer.add_scalar('data/Test Real Median Error (deg)', loss_deg_median.item(), total_iter_num)

                # Log Epoch Loss
                num_samples = len(testLoader.dataset)
                epoch_mean = running_mean / num_samples
                epoch_median = running_median / num_samples
                epoch_d1 = running_d1 / num_samples
                print('Test Real Epoch Mean: {:.4f}'.format(epoch_mean))
                print('Test Real Epoch Median: {:.4f}'.format(epoch_median))
                writer.add_scalar('data/Test Real Epoch Mean Error (deg)', epoch_mean, total_iter_num)
                writer.add_scalar('data/Test Real Epoch Median Error (deg)', epoch_median, total_iter_num)
                writer.add_scalar('data/Test Real Epoch d1 Error (deg)', epoch_d1, total_iter_num)

                # Log 30 images every N epochs
                if (epoch % config.train.saveImageInterval) == 0:
                    grid_image = u.create_grid_image(torch.cat(img_tensor_list, dim=0),
                                                        torch.cat(output_tensor_list, dim=0),
                                                        torch.cat(label_tensor_list, dim=0),
                                                        max_num_images_to_save=200)
                    writer.add_image('Test Real', grid_image, total_iter_num)

            #######################################################
            #########################开始测试###################
            #######################################################
            if db_test_synthetic_list:
                print('\nTest Synthetic:')
                print('=' * 10)

                model.eval()

                running_loss = 0.0
                running_mean = 0
                running_median = 0
                for iter_num, sample_batched in enumerate(tqdm(testSyntheticLoader)):
                    inputs, labels, masks = sample_batched

                    # Forward pass of the mini-batch
                    if config.train.model == 'refinenet':
                        labels_resized = self._resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                        labels_resized = labels_resized.to(device)
                    if config.train.model == 'densenet':
                        labels_resized = self._resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                        labels_resized = labels_resized.to(device)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        normal_vectors = model(inputs)

                    normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

                    if config.train.model == 'unet':
                        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                    elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                        loss /= config.train.batchSize
                    elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                        # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
                        loss = criterion(normal_vectors_norm, labels_resized.double(), reduction='sum')
                        loss /= config.train.batchSize

                    running_loss += loss.item()

                    loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = self._metric_calculator_batch(
                        normal_vectors_norm, labels.double())
                    running_mean += loss_deg_mean.item()
                    running_median += loss_deg_median.item()

                num_samples = (len(testSyntheticLoader))
                epoch_loss = running_loss / num_samples
                writer.add_scalar('data/Test Synthetic Epoch Loss', epoch_loss, total_iter_num)
                print('\Test Synthetic Epoch Loss: {:.4f}'.format(epoch_loss))
                epoch_mean = running_mean / num_samples
                epoch_median = running_median / num_samples
                print('Test Synthetic Epoch Mean: {:.4f}'.format(epoch_mean))
                print('Test Synthetic Epoch Median: {:.4f}'.format(epoch_median))
                writer.add_scalar('data/Test Synthetic Epoch Mean Error (deg)', epoch_mean, total_iter_num)
                writer.add_scalar('data/Test Synthetic Epoch Median Error (deg)', epoch_median, total_iter_num)

                if (epoch % config.train.saveImageInterval) == 0:
                    if config.train.model == 'refinenet':
                        output = self._resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 4),
                                            int(normal_vectors_norm.shape[3] * 4))
                    elif config.train.model == 'densenet':
                        output = self._resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 2),
                                            int(normal_vectors_norm.shape[3] * 2))
                    else:
                        output = normal_vectors_norm.detach().cpu().float()

                    grid_image = u.create_grid_image(inputs.detach().cpu(),
                                                        output,
                                                        labels.detach().cpu(),
                                                        max_num_images_to_save=16)
                    writer.add_image('Test Synthetic', grid_image, total_iter_num)

        writer.close()

    def _resize_tensor(input_tensor,height,width):
        augs_label_resize=iaa.Sequential([iaa.Resize({"height": height,"width": width},interpolation='nearest')])
        det_tf=augs_label_resize.to_deterministic()
        input_tensor=input_tensor.numpy().transpose(0,2,3,1)
        resized_array=det_tf.augment_images(input_tensor)
        resized_array=torch.from_numpy(resized_array.transpose(0,3,1,2))
        resized_array=resized_array.type(torch.DoubleTensor)

        return resized_array

    def _loss_fn_cosine(self,input_vec,target_vec,reduction='sum'):
        '''用于曲面法线估计的余弦损失函数。计算两个向量之间的余弦损失。两者的尺寸应该相同。'''
        
        cos=nn.CosineSimilarity(dim=1,eps=1e-6)
        loss_cos=1.0-cos(input_vec,target_vec)

        # 仅在合法像素上计算损失
        mask_invalid_pixel=torch.all(target_vec==-1,dim=1) & torch.all(target_vec==0,dim=1)

        loss_cos[mask_invalid_pixel]=0.0
        loss_cos_sum=loss_cos.sum()
        total_valid_pixel=(~mask_invalid_pixel).sum()
        error_output=loss_cos_sum/total_valid_pixel

        if reduction=='elementwise_mean':
            loss_cos=error_output
        elif reduction=='sum':
            loss_cos=loss_cos_sum
        elif reduction=='none':
            loss_cos=loss_cos
        else:
            raise Exception('Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

        return loss_cos

    def _metric_calculator_batch(self,input_vec,target_vec,mask=None):
        '''计算预测值与实际值之间的平均值、中位数和角度误差'''
        if len(input_vec.shape)!=4:
            raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(input_vec.shape))
        if len(target_vec.shape) != 4:
            raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(target_vec.shape))
        
        INVALID_PIXEL_VALUE=0
        mask_valid_pixels=~(torch.all(target_vec==INVALID_PIXEL_VALUE,dim=1))
        if mask is not None:
            mask_valid_pixels=(mask_valid_pixels.float()*mask).byte()
        total_valid_pixels=mask_valid_pixels.sum()
        if (total_valid_pixels==0):
            print('[WARN]: Image found with ZERO valid pixels to calc metrics')
            return torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0),mask_valid_pixels
        
        cos=nn.CosineSimilarity(dim=1,eps=1e-6)
        loss_cos=cos(input_vec,target_vec)

        eps=1e-10
        loss_cos=torch.clamp(loss_cos,(-1.0+eps),(1.0-eps))
        loss_rad=torch.acos(loss_cos)
        loss_deg=loss_rad*(180.0/math.pi)

        loss_deg=loss_deg[mask_valid_pixels]
        loss_deg_mean=loss_deg.mean()
        loss_deg_median=loss_deg.median()

        percentage_1=((loss_deg<11.25).sum().float()/total_valid_pixels)*100
        percentage_2=((loss_deg<22.5).sum().float()/total_valid_pixels)*100
        percentage_3=((loss_deg<30).sum().float()/total_valid_pixels)*100

        return loss_deg_mean,loss_deg_median,percentage_1,percentage_2,percentage_3


    def _metric_calculator(self,input_vec,target_vec,mask=None):
        '''计算预测值与实际值之间的平均值、中位数和角度误差'''
        if len(input_vec.shape)!=3:
            raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(input_vec.shape))
        if len(target_vec.shape) != 3:
            raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(target_vec.shape))
        
        INVALID_PIXEL_VALUE=0
        mask_valid_pixels=~(torch.all(target_vec==INVALID_PIXEL_VALUE,dim=0))
        if mask is not None:
            mask_valid_pixels=(mask_valid_pixels.float()*mask).byte()
        total_valid_pixels=mask_valid_pixels.sum()
        if (total_valid_pixels==0):
            print('[WARN]: Image found with ZERO valid pixels to calc metrics')
            return torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0),mask_valid_pixels
        
        cos=nn.CosineSimilarity(dim=1,eps=1e-6)
        loss_cos=cos(input_vec,target_vec)

        eps=1e-10
        loss_cos=torch.clamp(loss_cos,(-1.0+eps),(1.0-eps))
        loss_rad=torch.acos(loss_cos)
        loss_deg=loss_rad*(180.0/math.pi)

        loss_deg=loss_deg[mask_valid_pixels]
        loss_deg_mean=loss_deg.mean()
        loss_deg_median=loss_deg.median()

        percentage_1=((loss_deg<11.25).sum().float()/total_valid_pixels)*100
        percentage_2=((loss_deg<22.5).sum().float()/total_valid_pixels)*100
        percentage_3=((loss_deg<30).sum().float()/total_valid_pixels)*100

        return loss_deg_mean,loss_deg_median,percentage_1,percentage_2,percentage_3

    def _loss_fn_radians(self,input_vec,target_vec,reduction='sum'):
        '''表面法线估计的损失函数。计算两个向量之间的夹角通过求cos的倒数。'''
        
        cos=nn.CosineSimilarity(dim=1,eps=1e-6)
        loss_cos=cos(input_vec,target_vec)
        loss_rad=torch.acos(loss_cos)
        if reduction=='elementwise_mean':
            loss_rad=torch.mena(loss_rad)
        elif reduction=='sum':
            loss_cos=torch.sum(loss_rad)
        elif reduction=='none':
            pass
        else:
            raise Exception('Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

        return loss_rad

    def _coss_entropy2d(self,logit,target,ignore_index=255,weight=None,batch_average=True):
        n,c,h,w=logit.shape
        target=target.squeeze(1)

        if weight is None:
            criterion=nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_index,reduction='sum')
        else:
            criterion=nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32),ignore_index=ignore_index,reduction='sum')
        
        loss=criterion(logit,target.long())

        if batch_average:
            loss/=n
            
        return loss


if __name__ == '__main__':
    surface_normal=SurfaceNormal()
    surface_normal.train()
