import sys
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn

class Utils:
    def __init__(self) -> None:
        pass

    def lr_poly(self,base_lr,iter_,max_iter=100,power=0.9):
        return base_lr*((1-float(iter_)/max_iter)**power)
    
    def normal_to_rgb(self,normals_to_convert):
        camera_normal_rgb = (normals_to_convert + 1) / 2
        return camera_normal_rgb

    def create_grid_image(self,inputs, outputs, labels, max_num_images_to_save=3):

        img_tensor = inputs[:max_num_images_to_save]

        output_tensor = outputs[:max_num_images_to_save]
        output_tensor_rgb = self.normal_to_rgb(output_tensor)

        label_tensor = labels[:max_num_images_to_save]
        mask_invalid_pixels = torch.all(label_tensor == 0, dim=1, keepdim=True)
        mask_invalid_pixels = (torch.cat([mask_invalid_pixels] * 3, dim=1)).byte()

        label_tensor_rgb = self.normal_to_rgb(label_tensor)
        label_tensor_rgb[mask_invalid_pixels] = 0

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        x = cos(output_tensor, label_tensor)
        # loss_cos = 1.0 - x
        loss_rad = torch.acos(x)

        loss_rad_rgb = np.zeros((loss_rad.shape[0], 3, loss_rad.shape[1], loss_rad.shape[2]), dtype=np.float32)
        for idx, img in enumerate(loss_rad.numpy()):
            error_rgb = api_utils.depth2rgb(img,
                                            min_depth=0.0,
                                            max_depth=1.57,
                                            color_mode=cv2.COLORMAP_PLASMA,
                                            reverse_scale=False)
            loss_rad_rgb[idx] = error_rgb.transpose(2, 0, 1) / 255
        loss_rad_rgb = torch.from_numpy(loss_rad_rgb)
        loss_rad_rgb[mask_invalid_pixels] = 0

        mask_invalid_pixels_rgb = torch.ones_like(img_tensor)
        mask_invalid_pixels_rgb[mask_invalid_pixels] = 0

        images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb, loss_rad_rgb, mask_invalid_pixels_rgb), dim=3)
        grid_image = make_grid(images, 1, normalize=False, scale_each=False)

        return grid_image
