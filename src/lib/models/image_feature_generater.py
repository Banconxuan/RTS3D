# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import cv2
import numpy as np
import matplotlib.pyplot as plt
BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class PoseResNet(nn.Module):

    def __init__(self, block, layers,opt, **kwargs):

        self.inplanes = 64
        self.deconv_with_bias = False


        super(PoseResNet, self).__init__()
        self.opt = opt
        self.conv0 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1,
                               bias=False),
                                nn.BatchNorm2d(3, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.reduce = nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def generate_gridpoint(self, dim, pos, ori, calib_l, calib_r, trans_output_l, trans_output_r):  # dim B,K,3
        '''
           generate grid point coordinates, the image featuremap coordinates corresponding the grid point.
           return:
                image_xy_l: left image featuremap coordinates corresponding the grid point.
                image_xy_r: right image featuremap coordinates corresponding the grid point.
                xyz_norm: the grid point coordinates in the object coordinate system
                xyz: the grid point coordinates in the camera coordinate system
        '''

        h = dim[0]
        w = dim[1]
        l = dim[2]
        x_axi = -torch.linspace(-l / 2., l / 2., self.opt.R_l).cuda()
        y_axi = torch.linspace(0, -h, self.opt.R_h).cuda()
        z_axi = -torch.linspace(-w / 2., w / 2., self.opt.R_w).cuda()
        xx, yy, zz = torch.meshgrid(x_axi, y_axi, z_axi)
        xyz = torch.stack([xx, yy, zz], 0).view((3, -1))  # 3,resl***2
        R = ori
        xyz = R.mm(xyz)
        xyz_norm = xyz.clone()
        xyz[0, :] += pos[0]
        xyz[1, :] += pos[1]
        xyz[2, :] += pos[2]
        ones = torch.ones((1, xyz.size(1))).cuda()
        xyz_hom = torch.cat((xyz, ones), dim=0)
        image_xy_hom_l = calib_l.mm(xyz_hom)
        image_xy_hom_l = image_xy_hom_l / image_xy_hom_l[2, :]

        image_xy_hom_r = calib_r.mm(xyz_hom)
        image_xy_hom_r = image_xy_hom_r / image_xy_hom_r[2, :]
        image_xy_l = []
        image_xy_r = []
        for py in range(self.opt.pynum):
            image_xy_l.append(trans_output_l[py].mm(image_xy_hom_l))
            image_xy_r.append(trans_output_r[py].mm(image_xy_hom_r))

        image_xy_l = torch.stack(image_xy_l,dim=0)
        image_xy_r = torch.stack(image_xy_r, dim=0)
        return image_xy_l, image_xy_r, xyz_norm, xyz

    def featuremap2gridpoint(self, outputs_l, outputs_r, batch, phase='train'):
        '''
           image featuremap to gridpoint
        '''
        batch_for_point = {}
        batch_for_point['dim'] = []
        batch_for_point['pos'] = []
        batch_for_point['ori'] = []
        batch_for_point['dim_real'] = []
        batch_for_point['pos_real'] = []
        batch_for_point['ori_real'] = []
        batch_for_point['dim_est'] = []
        batch_for_point['pos_est'] = []
        batch_for_point['ori_est_scalar'] = []
        batch_for_point['reg_mask'] = []


        B = outputs_l[0].size(0)
        ## *_est represent monocular 3D detector results.
        dim = batch['dim_est']
        pos = batch['pos_est']
        ori = batch['ori_est']
        calib_l = batch['calib_l']
        calib_r = batch['calib_r']
        ## trans_output_* represent the transformation from 3D grid point to image featuremap.
        trans_output_l = batch['trans_output_l']
        trans_output_r = batch['trans_output_r']

        pointNet_input_list_r = []
        pointNet_input_list_l = []
        pointNet_input_list_xyz_abs = []
        pointNet_input_consis = []
        reg_mask = batch['reg_mask']
        obj_num=[]
        for b in range(B):
            index_box_l = []
            index_box_r = []
            volume_xyz_list = []
            volume_xyz_abs_list = []
            mask = torch.nonzero(reg_mask[b])
            K = mask.size(0)
            obj_num.append(K)
            for k in range(K):#range(self.opt.max_objs):
                #k_index = mask[k, 0]
                index_l, index_r, xyz, xyz_abs = self.generate_gridpoint(dim[b, k], pos[b, k],
                                                                          ori[b, k], calib_l[b],
                                                                          calib_r[b], trans_output_l[b],
                                                                          trans_output_r[b])
                index_box_l.append(index_l)
                index_box_r.append(index_r)
                volume_xyz_list.append(xyz)
                volume_xyz_abs_list.append(xyz_abs)
            index_box_l = torch.stack(index_box_l, 0).transpose(3,2).unsqueeze(0)  # 1,K,3,2,resl***2
            index_box_r = torch.stack(index_box_r, 0).transpose(3,2).unsqueeze(0)

            volume_xyz_list = torch.stack(volume_xyz_list, 0)  # m(<=K),3,resl***2
            volume_xyz_abs_list = torch.stack(volume_xyz_abs_list, 0)
            volume_from_heatmap_l = []
            volume_from_heatmap_r = []
            for py in range(self.opt.pynum):
                grid_l = index_box_l[:,:,py,:,:]  #1, K,resl***2,2
                grid_r = index_box_r[:,:,py,:,:]  #1, K,resl***2,2
                featuremap_l = outputs_l[py]
                featuremap_r = outputs_r[py]
                lx = 2 * (grid_l[:, :, :, 0] / featuremap_l.size(3) - 0.5)
                ly = 2 * (grid_l[:, :, :, 1] / featuremap_l.size(2) - 0.5)
                rx = 2 * (grid_r[:, :, :, 0] / featuremap_r.size(3) - 0.5)
                ry = 2 * (grid_r[:, :, :, 1] / featuremap_r.size(2) - 0.5)

                grid_l = torch.stack((lx, ly),dim=3)
                grid_r = torch.stack((rx, ry), dim=3)

                volume_from_heatmap_l.append(torch.nn.functional.grid_sample(featuremap_l[b:b + 1], grid_l))  # 1,64,16K,resl***2
                volume_from_heatmap_r.append(torch.nn.functional.grid_sample(featuremap_r[b:b + 1], grid_r))  # 1,64,16K,resl***2

            volume_from_heatmap_l = torch.cat(volume_from_heatmap_l,dim=1)   # 1,mm,K,resl***2
            volume_from_heatmap_r = torch.cat(volume_from_heatmap_r, dim=1)  # 1,mm,K,resl***2

            volume_from_heatmap_l = volume_from_heatmap_l[0].transpose(1, 0)
            volume_from_heatmap_r = volume_from_heatmap_r[0].transpose(1, 0)


            volume_from_heatmap = volume_from_heatmap_l[:,:128,:] - volume_from_heatmap_r[:,:128,:]

            BRF=(volume_from_heatmap_l[:,128:256,:] +volume_from_heatmap_r[:,128:256,:])/2
            semantic = (volume_from_heatmap_l[:, 256:, :] + volume_from_heatmap_r[:, 256:, :]) / 2
            volume_from_heatmap=torch.exp(-(volume_from_heatmap**2)*(BRF**2))

            volume_depth=torch.norm(volume_xyz_abs_list,p=2,dim=1,keepdim=True)
            volume_from_heatmap = torch.cat([volume_from_heatmap,volume_xyz_list,volume_depth,semantic], dim=1)

            if phase=='train':
                batch_for_point['dim'].append(batch['dim'][b])
                batch_for_point['pos'].append(batch['pos'][b])
                batch_for_point['ori'].append(batch['ori'][b])
                batch_for_point['dim_real'].append(batch['dim_real'][b])
                batch_for_point['pos_real'].append(batch['pos_real'][b])
                batch_for_point['ori_real'].append(batch['ori_real'][b])
            batch_for_point['reg_mask'].append(batch['reg_mask'][b])
            batch_for_point['dim_est'].append(batch['dim_est'][b])
            batch_for_point['pos_est'].append(batch['pos_est'][b])
            batch_for_point['ori_est_scalar'].append(batch['ori_est_scalar'][b])
            pointNet_input_list_l.append(volume_from_heatmap_l)
            pointNet_input_list_r.append(volume_from_heatmap_r)
            pointNet_input_list_xyz_abs.append(volume_xyz_abs_list)
            pointNet_input_consis.append(volume_from_heatmap)

        pointNet_input_tensor_l = torch.cat(pointNet_input_list_l, dim=0)
        pointNet_input_tensor_r = torch.cat(pointNet_input_list_r, dim=0)
        pointNet_input_tensor_consis = torch.cat(pointNet_input_consis, dim=0)
        pointNet_input_tensor_xyz_abs = torch.cat(pointNet_input_list_xyz_abs, dim=0)

        input_model = {}
        input_model['input_feat_l'] = pointNet_input_tensor_l
        input_model['input_feat_r'] = pointNet_input_tensor_r
        input_model['input_feat_xyz_abs'] = pointNet_input_tensor_xyz_abs
        input_model['input_feat_consis'] = pointNet_input_tensor_consis
        if phase == 'train':
            batch_for_point['dim'] = torch.cat(batch_for_point['dim'], dim=0)
            batch_for_point['pos'] = torch.cat(batch_for_point['pos'], dim=0)
            batch_for_point['ori'] = torch.cat(batch_for_point['ori'], dim=0)
            batch_for_point['dim_real'] = torch.cat(batch_for_point['dim_real'], dim=0)
            batch_for_point['pos_real'] = torch.cat(batch_for_point['pos_real'], dim=0)
            batch_for_point['ori_real'] = torch.cat(batch_for_point['ori_real'], dim=0)

        batch_for_point['dim_est'] = torch.cat(batch_for_point['dim_est'], dim=0)
        batch_for_point['pos_est'] = torch.cat(batch_for_point['pos_est'], dim=0)
        batch_for_point['ori_est_scalar'] = torch.cat(batch_for_point['ori_est_scalar'], dim=0)
        batch_for_point['reg_mask'] = torch.cat(batch_for_point['reg_mask'], dim=0)
        input_model['input_batch'] = batch_for_point
        #input_model['obj_num']=obj_num
        return input_model
    def forward(self, batch, phase='train'):
        xl1, xr1 = batch['input'], batch['input_r']
        xl2 = self.relu(self.bn1(self.conv1(xl1)))
        xl3 = self.layer1(self.maxpool(xl2))
        xl4 = self.layer2(xl3)
        xl5 = self.layer3(xl4)
        #xl6 = self.layer4(xl5)


        xl5=self.reduce(xl5)
        xr2 = self.relu(self.bn1(self.conv1(xr1)))
        xr3 = self.layer1(self.maxpool(xr2))
        xr4=self.layer2(xr3)
        xr5 = self.layer3(xr4)
        #xr6 = self.layer4(xr5)
        xr5 = self.reduce(xr5)

        xl = [xl2,xl3,xl4,xl5]
        xr = [xr2,xr3,xr4,xr5]
        ret = {}
        ret['left_image_feature'] = xl
        ret['right_image_feature'] = xr
        # input_data = self.featuremap2gridpoint(xl, xr, batch,phase=phase)
        return ret
    def draw_features(self,width, height, x, savename):

        fig = plt.figure(figsize=(40, 20))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.00001, hspace=0.00003)
        for i in range(width * height):
            plt.subplot(height, width, i + 1)
            plt.axis('off')
            img = x[0, i, :, :]
            pmin = np.min(img)
            pmax = np.max(img)
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
            img = img.astype(np.uint8)  # 转成unit8
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
            img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
            plt.imshow(img)
            print("{}/{}".format(i, width * height))
        fig.savefig(savename, dpi=200)
        fig.clf()
        plt.close()
    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')

            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            #raise ValueError('imagenet pretrained model does not exist')
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_image_feature(num_layers,opt):
  block_class, layers = resnet_spec[num_layers]
  model = PoseResNet(block_class, layers,opt=opt)
  model.init_weights(num_layers, pretrained=True)
  return model
