from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
import math
from models.embedding_space_generater import featuremap2gridpoint
try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

from utils.post_process import car_pose_post_process


from .base_detector import BaseDetector


class RTS3DDetector(BaseDetector):
    def __init__(self, opt):
        super(RTS3DDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx
        self.opt = opt
    def process(self, meta, return_time=False):
        dets=[]
        with torch.no_grad():
            image_ret = self.model_image(meta, phase='val')
            torch.cuda.synchronize()
            start = time.time()
            meta.update(image_ret)
            for i in range(self.opt.interation):
                point_data = featuremap2gridpoint(meta, 'demo', self.opt)
                output = self.model_point(point_data)

                reg_mask = meta['reg_mask']
                dim_est = meta['dim_est']
                dim_est_mask = dim_est[reg_mask]
                dim_est[reg_mask] = dim_est_mask + output[:, 3:6]
                meta['dim_est'] = dim_est

                pos_est = meta['pos_est']
                pos_est_mask = pos_est[reg_mask]
                pos_est[reg_mask] = pos_est_mask + output[:, 0:3]
                meta['pos_est'] = pos_est

                ori_est_scalar = meta['ori_est_scalar']
                ori_est_scalar_mask = ori_est_scalar[reg_mask]
                ry = output[:, 6] + ori_est_scalar_mask
                ori_est_scalar[reg_mask] = ry
                meta['ori_est_scalar'] = ori_est_scalar

                R_yaw = meta['ori_est'].new_zeros(ry.size(0), 3, 3)
                R_yaw[:, 0, 0] = torch.cos(ry)
                R_yaw[:, 0, 2] = torch.sin(ry)
                R_yaw[:, 1, 1] = 1
                R_yaw[:, 2, 0] = -torch.sin(ry)
                R_yaw[:, 2, 2] = torch.cos(ry)

                ori_est = meta['ori_est']
                ori_est[reg_mask] = R_yaw
                meta['ori_est'] = ori_est

            torch.cuda.synchronize()
            forward_time = time.time() - start
            output = output.detach().cpu().numpy()
            for i in range(output.shape[0]):
                meta['ori_DATA'][i][8] = meta['dim_est'][0,i,0].item()
                meta['ori_DATA'][i][9] = meta['dim_est'][0,i,1].item()
                meta['ori_DATA'][i][10] = meta['dim_est'][0,i,2].item()
                meta['ori_DATA'][i][11] = meta['pos_est'][0,i,0].item()
                meta['ori_DATA'][i][12] = meta['pos_est'][0,i,1].item()
                meta['ori_DATA'][i][13] = meta['pos_est'][0,i,2].item()
                meta['ori_DATA'][i][14] = meta['ori_est_scalar'][0, i].item()
                meta['ori_DATA'][i][15] = 1 / (1 + math.exp(-output[i, 7]))
                dets.append(meta['ori_DATA'])
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = car_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 46)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:23] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image_l, image_r, results, calib):
        calib = calib.cpu().numpy()[0]
        debugger.add_img(image_l, img_id='car_pose_l')
        #debugger.add_img(image_r, img_id='car_pose_r')
        for bbox in results[0]:
                debugger.add_3d_detection(bbox, calib, img_id='car_pose_l', )
                debugger.add_bev(bbox, img_id='car_pose_l')
                debugger.save_kitti_formet(bbox, self.image_path, self.opt)
        if self.opt.vis:
            debugger.show_all_imgs(pause=self.pause)
