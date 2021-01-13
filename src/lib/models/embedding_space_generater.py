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
def generate_gridpoint(dim, pos, ori, calib_l, calib_r, trans_output_l, trans_output_r, opt=None):  # dim B,K,3
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
    x_axi = -torch.linspace(-l / 2., l / 2., opt.R_l).cuda()
    y_axi = torch.linspace(0, -h, opt.R_h).cuda()
    z_axi = -torch.linspace(-w / 2., w / 2., opt.R_w).cuda()
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
    for py in range(opt.pynum):
        image_xy_l.append(trans_output_l[py].mm(image_xy_hom_l))
        image_xy_r.append(trans_output_r[py].mm(image_xy_hom_r))

    image_xy_l = torch.stack(image_xy_l,dim=0)
    image_xy_r = torch.stack(image_xy_r, dim=0)
    return image_xy_l, image_xy_r, xyz_norm, xyz

def featuremap2gridpoint(batch, phase='train', opt = None):
    '''
       image featuremap to gridpoint
    '''
    outputs_l, outputs_r = batch['left_image_feature'], batch['right_image_feature']
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
            index_l, index_r, xyz, xyz_abs = generate_gridpoint(dim[b, k], pos[b, k],
                                                                      ori[b, k], calib_l[b],
                                                                      calib_r[b], trans_output_l[b],
                                                                      trans_output_r[b], opt)
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
        for py in range(opt.pynum):
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

        if phase == 'train' or phase == 'val':
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
    if phase == 'train' or phase =='val':
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


