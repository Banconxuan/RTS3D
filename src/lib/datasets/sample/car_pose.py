from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import random

class CarPoseDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    def read_calib(self,calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def E2R(self,Ry):
        '''Combine Euler angles to the rotation matrix (right-hand)

            Inputs:
                Ry, Rx, Rz : rotation angles along  y, x, z axis
                             only has Ry in the KITTI dataset
            Returns:
                3 x 3 rotation matrix

        '''
        R_yaw = np.array([[math.cos(Ry), 0, math.sin(Ry)],
                          [0, 1, 0],
                          [-math.sin(Ry), 0, math.cos(Ry)]])

        return R_yaw
    def __getitem__(self, index):
        img_id = self.images[index]
        if img_id<7481:
            img_id_r=img_id+7481
        else:
            img_id_r = img_id - 7481
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        file_name_r="{:06d}".format(img_id_r)+'.png'
        img_path = os.path.join(self.img_dir, file_name)
        img_path_r = os.path.join(self.img_dir, file_name_r)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        img_r = cv2.imread(img_path_r)
        img_shape = img.shape[:2]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0
        c_r = np.array([img_r.shape[1] / 2., img_r.shape[0] / 2.], dtype=np.float32)
        s_r = max(img_r.shape[0], img_r.shape[1]) * 1.0


        trans_input_l = get_affine_transform(
            c, s, rot, [self.opt.input_w, self.opt.input_h])
        trans_input_r = get_affine_transform(
            c_r, s_r, rot, [self.opt.input_w, self.opt.input_h])

        inp = cv2.warpAffine(img, trans_input_l,
                             (self.opt.input_w, self.opt.input_h),
                             #(self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        inp_r = cv2.warpAffine(img_r, trans_input_r,
                             (self.opt.input_w, self.opt.input_h),
                             # (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp_r = (inp_r.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp_r, self._eig_val, self._eig_vec)
        inp_r = (inp_r - self.mean) / self.std
        inp_r = inp_r.transpose(2, 0, 1)
        trans_output_l = np.zeros((self.opt.pynum, 2, 3), dtype=np.float32)
        trans_output_r = np.zeros((self.opt.pynum, 2, 3), dtype=np.float32)
        for j in range(self.opt.pynum):
            down_ratio = math.pow(2,j+1)
            trans_output_l[j,:,:] = get_affine_transform(
                c, s, rot, [self.opt.input_w // down_ratio, self.opt.input_h // down_ratio])
            trans_output_r[j,:,:] = get_affine_transform(
                c_r, s_r, rot, [self.opt.input_w // down_ratio, self.opt.input_h // down_ratio])
        dim = np.zeros((self.max_objs, 3), dtype=np.float32)
        ori = np.zeros((self.max_objs), dtype=np.float32)
        pos = np.zeros((self.max_objs, 3), dtype=np.float32)

        dim_real = np.zeros((self.max_objs, 3), dtype=np.float32)
        ori_real = np.zeros((self.max_objs), dtype=np.float32)
        pos_real = np.zeros((self.max_objs, 3), dtype=np.float32)

        dim_est = np.zeros((self.max_objs, 3), dtype=np.float32)
        ori_est = np.zeros((self.max_objs, 3, 3), dtype=np.float32)
        ori_est_scalar = np.zeros((self.max_objs), dtype=np.float32)
        pos_est = np.zeros((self.max_objs, 3), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        calib = np.array(anns[0]['calib_l'], dtype=np.float32)
        calib = np.reshape(calib, (3, 4))
        calib_r = np.array(anns[0]['calib_r'], dtype=np.float32)
        calib_r = np.reshape(calib_r, (3, 4))
        if self.split == 'val':
            for k in range(self.max_objs):
                if k + 1 > num_objs:
                    kk = random.randint(0, num_objs - 1)
                    ann = anns[kk]
                else:

                    ann = anns[k]
                reg_mask[k] = 1
                dim_est[k][0] = ann['dim'][0] + random.uniform(-0.8, 0.8)
                dim_est[k][1] = ann['dim'][1] + random.uniform(-0.8, 0.8)
                dim_est[k][2] = ann['dim'][2] + random.uniform(-0.8, 0.8)
                ori[k] = random.uniform(-0.3, 0.3)
                ori_est_scalar[k] = ann['rotation_y'] - ori[k]
                ori_est[k] = self.E2R(ann['rotation_y'] - ori[k])
                pos_est[k][0] = ann['location'][0] + random.uniform(-1, 1)
                pos_est[k][1] = ann['location'][1] + random.uniform(-0.5, 0.5)
                pos_est[k][2] = ann['location'][2] + random.uniform(-2, 2)

                dim[k][0] = ann['dim'][0] - dim_est[k][0]
                dim[k][1] = ann['dim'][1] - dim_est[k][1]
                dim[k][2] = ann['dim'][2] - dim_est[k][2]

                pos[k][0] = ann['location'][0] - pos_est[k][0]
                pos[k][1] = ann['location'][1] - pos_est[k][1]
                pos[k][2] = ann['location'][2] - pos_est[k][2]

                dim_real[k][0] = ann['dim'][0]
                dim_real[k][1] = ann['dim'][1]
                dim_real[k][2] = ann['dim'][2]

                pos_real[k][0] = ann['location'][0]
                pos_real[k][1] = ann['location'][1]
                pos_real[k][2] = ann['location'][2]
                ori_real[k] = ann['rotation_y']
        if self.split =='train':
            for k in range(self.max_objs):
                if k+1>num_objs:
                    kk=random.randint(0,num_objs-1)
                    ann = anns[kk]
                else:
                    ann = anns[k]
                reg_mask[k]=1
                if np.random.random() < 0.7:
                    dim_est[k][0] = ann['dim'][0] + random.uniform(-1.5, 1.5)
                    dim_est[k][1] = ann['dim'][1] + random.uniform(-1.5, 1.5)
                    dim_est[k][2] = ann['dim'][2] + random.uniform(-1.5, 1.5)
                    ori[k] = random.uniform(-0.6, 0.6)
                    ori_est_scalar[k] = ann['rotation_y'] - ori[k]
                    ori_est[k] = self.E2R(ann['rotation_y'] - ori[k])
                    pos_est[k][0] = ann['location'][0] + random.uniform(-2, 2)
                    pos_est[k][1] = ann['location'][1] + random.uniform(-0.8, 0.8)
                    pos_est[k][2] = ann['location'][2] + random.uniform(-3, 3)
                else:
                    dim_est[k][0] = ann['dim'][0] + random.uniform(-0.5, 0.5)
                    dim_est[k][1] = ann['dim'][1] + random.uniform(-0.5, 0.5)
                    dim_est[k][2] = ann['dim'][2] + random.uniform(-0.5, 0.5)
                    ori[k] = random.uniform(-0.3, 0.3)
                    ori_est_scalar[k] = ann['rotation_y'] - ori[k]
                    ori_est[k] = self.E2R(ann['rotation_y'] - ori[k])
                    pos_est[k][0] = ann['location'][0] + random.uniform(-0.8, 0.8)
                    pos_est[k][1] = ann['location'][1] + random.uniform(-0.3, 0.3)
                    pos_est[k][2] = ann['location'][2] + random.uniform(-1, 1)
                dim[k][0] = ann['dim'][0]-dim_est[k][0]
                dim[k][1] = ann['dim'][1]-dim_est[k][1]
                dim[k][2] = ann['dim'][2]-dim_est[k][2]

                pos[k][0] = ann['location'][0]- pos_est[k][0]
                pos[k][1] = ann['location'][1]- pos_est[k][1]
                pos[k][2] = ann['location'][2]- pos_est[k][2]

                dim_real[k][0] = ann['dim'][0]
                dim_real[k][1] = ann['dim'][1]
                dim_real[k][2] = ann['dim'][2]

                pos_real[k][0] = ann['location'][0]
                pos_real[k][1] = ann['location'][1]
                pos_real[k][2] = ann['location'][2]
                ori_real[k] = ann['rotation_y']

            #reg_mask[k]=1

        meta = {}
        meta['img_shape'] = img_shape
        meta['num_objs'] = num_objs
        meta['img_name'] = file_name
        ret = {'input': inp,'input_r':inp_r,'dim':dim,'ori':ori,'pos':pos,'dim_real':dim_real,'ori_real':ori_real,'pos_real':pos_real,'dim_est':dim_est,'ori_est':ori_est,
               'pos_est':pos_est,'ori_est_scalar':ori_est_scalar,'calib_l':calib,'calib_r':calib_r,'trans_output_l':trans_output_l,'trans_output_r':trans_output_r,
               'reg_mask':reg_mask,'meta':meta}
        return ret
