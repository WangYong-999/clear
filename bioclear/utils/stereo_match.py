
import glob
import os
import time
import cv2
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

import numpy as np

RENDERING_PATH=os.path.dirname(os.path.abspath(__file__))
setting=yaml.load(open(os.path.join(RENDERING_PATH,"setting.yaml")),Loader=yaml.FullLoader)
class BioclearDataset(Dataset):
    def __init__(self,root,scale) -> None:
        super(BioclearDataset,self).__init__()
        self.root=root
        self.scale=scale
        self.ir_l_list=[]
        self.ir_r_list=[]
        self.depth_list=[]
        for img_id in glob.glob(self.root+"/*/*_ir_l.png"):
            if not os.path.exists(img_id[:-8]+"simDepthImage.exr"):
                path=img_id[:-8]
                self.ir_l_list.append(path+"ir_l.png")
                self.ir_r_list.append(path+"ir_r.png")
                self.depth_list.append(path+"depth_120.exr")
        print(self.ir_l_list,self.ir_r_list,self.depth_list)
    
    def __getitem__(self, index):
        '''读取ir和深度图像并转换成tensor'''
        ir_l_path=self.ir_l_list[index]
        ir_r_path=self.ir_r_list[index]
        imageL=Image.open(ir_l_path).convert("L")
        imageR=Image.open(ir_r_path).convert("L")#"L代表转换为灰度模式"
        imageL=np.array(imageL.resize((int(imageL.size[0]*self.scale),int(imageL[1]*self.scale)))).astype(np.float32)
        imageR=np.array(imageR.resize((int(imageR.size[0]*self.scale),int(imageR[1]*self.scale)))).astype(np.float32)
        imageL=np.array([imageL]).transpose(1,2,0)
        imageR=np.array([imageR]).transpose(1,2,0)
        imageLTensor=torch.from_numpy(imageL.transpose(2,0,1))
        imageRTensor=torch.from_numpy(imageR.transpose(2,0,1))

        # 读取深度转换为张量
        depth_image_path=self.depth_list[index]
        image=cv2.imread(depth_image_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_array=image[:,:,0]
        depthTensor=torch.from_numpy(depth_array).unsqueeze(0)
        return imageLTensor,imageRTensor,ir_l_path,ir_r_path,depthTensor
    
    def __len__(self):
        return len(self.ir_l_list)


class StereoMatch(nn.Module):
    def __init__(self,maxDisp=60,minDisp=1,blockSize=9,eps=1e-6,subPixel=True,bilateralFilter=True) -> None:
        super(StereoMatch,self).__init__()
        self.maxDisp=maxDisp
        self.minDisp=minDisp
        self.blockSize=blockSize
        self.eps=eps
        self.subPixel=subPixel
        self.bilateralFilter==bilateralFilter
    
    def forward(self,imageL,imageR,f,blDis,beta,sigmaColor=0.05,sigmaSpace=5):
        print("forward start")
        beginTime=time.time()
        B,C,H,W=imageL.shape
        D=self.maxDisp-self.minDisp+1
        device=imageL.device

        if self.maxDisp>=imageR.shape[3]:
            raise RuntimeError("The max disparity must be smaller than the width of input image")
        
        # 正态分布噪声
        mu=torch.full(imageL.size(),setting["mu"])
        std=torch.full(imageL.size(),setting["sigma"])
        eps_l=torch.randn_like(std)
        eps_r=torch.randn_like(std)

        if imageL.is_cuda():
            mu=mu.cuda()
            std=std.cuda()
            eps_l=eps_l.cuda()
            eps_r=eps_r.cuda()
        
        delta_img_receiver_l=eps_l.mul(std).add_(mu)
        delta_img_receiver_r=eps_r.mul(std).add_(mu)
        delta_img_receiver_l[delta_img_receiver_l<0]=0
        delta_img_receiver_r[delta_img_receiver_r>255]=255
        
        dispVolume=torch.zeros([D,B,1,H,W],device=device)#B,C,H,W,D
        costVolumeL=torch.zeros([D,B,1,H,W],device=device)#B,C,H,W,D
        costVolumeR=torch.zeros([D,B,1,H,W],device=device)#B,C,H,W,D

        filters=Variable(torch.ones(1,C,self.blockSize,self.blockSize,dtype=imageL.dtype,device=device))
        padding=(self.blockSize//2,self.blockSize//2)

        imageLSum=F.conv2d(imageL,filters,stride=1,padding=padding)
        imageLAve=imageLSum/(self.blockSize*self.blockSize*C)
        imageLAve2=imageLAve.pow(2)
        imageL2=imageL.pow(2)
        imageL2Sum=F.conv2d(imageL2,filters,stride=1,padding=padding)

        imageRSum=F.conv2d(imageR,filters,stride=1,padding=padding)
        imageRAve=imageRSum/(self.blockSize*self.blockSize*C)
        imageRAve2=imageRAve.pow(2)
        imageR2=imageR.pow(2)
        imageR2Sum=F.conv2d(imageR2,filters,stride=1,padding=padding)

        cacheImageL=[imageL,imageLSum,imageLAve,imageLAve2,imageL2,imageL2Sum]
        cacheImageR=[imageR,imageRSum,imageRAve,imageLAve2,imageR2,imageR2Sum]

        def CorrL(i,cacheImageL,cacheImageR,filters,padding,blockSize,eps):
            '''计算左协方差'''
            imageL,imageLSum,imageLAve,imageLAve2,imageL2,imageL2Sum=cacheImageL
            imageR,imageRSum,imageRAve,imageRAve2,imageR2,imageLR2Sum=cacheImageR
            B,C,H,W=imageL.shape
            
            # 裁剪图像
            cropedR=imageR.narrow(3,0,W-i)
            cropedRSum=imageRSum.narrow(3,0,W-i)
            cropedR2Sum=imageR2Sum.narrow(3,0,W-i)
            cropedRAve=imageRAve.narrow(3,0,W-i)
            cropedRAve2=imageRAve2.narrow(3,0,W-i)

            # pad图像
            shifted=F.pad(cropedR,(i,0,0),"constant",0.0)
            shiftedSum=F.pad(cropedRSum,(i,0,0,0),"constant",0.0)
            shifted2Sum=F.pad(cropedR2Sum,(i,0,0,0),"constant",0.0)
            shiftedAve=F.pad(cropedRAve,(i,0,0,0),"constant",0.0)
            shiftedAve2=F.pad(cropedRAve2,(i,0,0,0),"constant",0.0)

            LShifted=imageL*shifted
            LShiftedSum=F.conv2d(LShifted,filters,stride=1,padding=padding).double()
            LAveShifted=imageLAve*shiftedSum
            shiftedAveL=shiftedAve*imageLSum
            LAveShiftedAve=imageLAve*shiftedAve
            productSum=LShiftedSum-LAveShifted-shiftedAveL+blockSize*blockSize*C*LAveShiftedAve

            sqrtL=(imageL2Sum-2*imageLAve*imageLSum+blockSize*blockSize*C*LAveShiftedAve+1e-5).sqrt()
            sqrtShifted = (shifted2Sum - 2 * shiftedAve * shiftedSum + blockSize * blockSize * C * shiftedAve2 + 1e-5).sqrt()

            corrL = (productSum + eps) / (sqrtL * sqrtShifted + eps)
            corrL[:, :, :, :i] = 0

            return corrL
        
        def CorrR(i, cacheImageL, cacheImageR, filters, padding, blockSize, eps):
            '''计算右协方差'''
            imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum = cacheImageL
            imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum = cacheImageR
            B, C, H, W = imageL.shape

            cropedL = imageL.narrow(3, i, W - i)                
            cropedLSum = imageLSum.narrow(3, i, W - i)          
            cropedL2Sum = imageL2Sum.narrow(3, i, W - i)       
            cropedLAve = imageLAve.narrow(3, i, W - i)         
            cropedLAve2 = imageLAve2.narrow(3, i, W - i)       

            shifted = F.pad(cropedL, (0, i, 0, 0), "constant", 0.0)
            shiftedSum = F.pad(cropedLSum, (0, i, 0, 0), "constant", 0.0)
            shifted2Sum = F.pad(cropedL2Sum, (0, i, 0, 0), "constant", 0.0)
            shiftedAve = F.pad(cropedLAve, (0, i, 0, 0), "constant", 0.0)
            shiftedAve2 = F.pad(cropedLAve2, (0, i, 0, 0), "constant", 0.0)

            RShifted = imageR * shifted
            RShiftedSum = F.conv2d(RShifted, filters, stride=1, padding=padding).double()
            RAveShifted = imageRAve * shiftedSum
            shiftedAveR = shiftedAve * imageRSum
            RAveShiftedAve = imageRAve * shiftedAve
            productSum = RShiftedSum - RAveShifted - shiftedAveR + blockSize * blockSize * C * RAveShiftedAve

            sqrtR = (imageR2Sum - 2 * imageRAve * imageRSum + blockSize * blockSize * C * imageRAve2 + 1e-5).sqrt()
            sqrtShifted = (shifted2Sum - 2 * shiftedAve * shiftedSum + blockSize * blockSize * C * shiftedAve2 + 1e-5).sqrt()
            
            corrR = (productSum + eps) / (sqrtR * sqrtShifted + eps)
            corrR[:, :, :, W - i:] = 0

            return corrR

        def CostToDisp(costVolume,dispVolume,beta,eps,subPixel):
            if subPixel==True:
                D,B,C,H,W=costVolume.shape
                costVolumePad=torch.full((1,B,1,H,W),0,device=costVolume.device)

                dispVolume=(dispVolume+torch.cat((costVolume)))

        # 计算costVolume
        testBeginTime=time.time()
        for i in range(self.minDisp,D+1,1):
            costVolumeL[i-self.minDisp]=CorrL(i,cacheImageL,cacheImageR,filters,padding,self.blockSize,self.eps)
            costVolumeR[i-self.minDisp]=CorrR(i,cacheImageL,cacheImageR,filters,padding,self.blockSize,self.eps)
            dispVolume[i-self.minDisp]=torch.full_like(costVolumeL[0],i)
        
        testEndTime=time.time()
        print("costVolume Time: ", testEndTime - testBeginTime)

        # 计算视差图
        testBeginTime=time.time()
        dispL=CostToDisp(costVolumeL,dispVolume,beta,self.eps,self.subPixel)

def run(self):

    # 1 初始化cudnn
    dev=torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(32,32,32,32,device=dev),torch.zeros(32,32,32,32,device=dev))
    
    # 2 参数设置
    scale=0.5                           #图像缩放
    beta=torch.tensor(100.)             #softmax的beta参数
    blockSize=11                        #双目匹配参数
    maxDisp=110                         #左右图像最大差距
    minDisp=3                           #左右图像最小差距
    f=torch.tensor(446.31)              #传感器的焦距
    baselineDis=torch.tensor(0.055)     #传感器基线距离
    subPixel=True                       #是否计算亚像素
    bilateralFilter=True                #是否使用双边滤波
    simgaColor=torch.tensor(0.02)       #双边滤波的颜色参数
    simgaSpace=torch.tensor(3.)         #双边滤波的空间参数
    gpu_id=0
    batch_size=8
    input_root = "./rendered_output"
    save_root = "./rendered_output"

    # 3 加载数据集
    torch.cuda.set_device(gpu_id)
    bioclear_dataset=BioclearDataset(root=input_root,scale=scale)
    print("Len: ",len(bioclear_dataset))
    dataloader=torch.utils.data.DataLoader(bioclear_dataset,batch_size=batch_size,shuffle=False,num_workers=10)

    beta=torch.autograd.Variable(beta)
    simgaColor=torch.autograd.Variable(simgaColor)
    simgaSpace=torch.autograd.Variable(simgaSpace)

    object=StereoMatch()