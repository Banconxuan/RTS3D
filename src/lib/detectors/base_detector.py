from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import math
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
import os

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model_image, self.model_point = create_model(opt)
    self.model_image = load_model(self.model_image, opt.load_model, 'image_model')
    self.model_image = self.model_image.to(opt.device)
    self.model_image.eval()

    self.model_point = load_model(self.model_point, opt.load_model, 'point_model')
    self.model_point = self.model_point.to(opt.device)
    self.model_point.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
    self.image_path=' '
    self.max_objs=32
    const = torch.Tensor(
      [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
       [-1, 0], [0, -1], [-1, 0], [0, -1]])
    self.const = const.unsqueeze(0).unsqueeze(0)
    self.const=self.const.to(self.opt.device)

  def pre_process(self, image, scale, meta=None):
      height, width = image.shape[0:2]
      new_height = int(height * scale)
      new_width  = int(width * scale)
      if self.opt.fix_res:
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
      else:
        inp_height = (new_height | self.opt.pad) + 1
        inp_width = (new_width | self.opt.pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
      resized_image = cv2.resize(image, (new_width, new_height))
      inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
      inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

      images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
      if self.opt.flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
      images = torch.from_numpy(images)

      meta = {'c': c, 's': s,
              'out_height': inp_height // self.opt.down_ratio,
              'out_width': inp_width // self.opt.down_ratio}
      trans_output_l = np.zeros((self.opt.pynum, 2, 3), dtype=np.float32)
      for j in range(self.opt.pynum):
          down_ratio = math.pow(2, j + 1)
          trans_output_l[j, :, :] = get_affine_transform(
              c, s, 0, [self.opt.input_w // down_ratio, self.opt.input_h // down_ratio])

      trans_output_l = torch.from_numpy(trans_output_l)
      trans_output_l = trans_output_l.unsqueeze(0)
      trans_output_r = trans_output_l

      meta['trans_output_l'] = trans_output_l
      meta['trans_output_r'] = trans_output_r

      return images, meta

  def process(self, meta, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image_l,image_r, results):
   raise NotImplementedError

  def read_clib(self,calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
      if i == 2:
        calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        calib = calib.reshape(3, 4)
        return calib
  def read_clib3(self,calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
      if i == 3:
        calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        calib = calib.reshape(3, 4)
        return calib

  def E2R(self, Ry):
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
  def read_est_from_mono(self,LABEL_PATH,meta):
    detection_data = open(LABEL_PATH, 'r')
    detections = detection_data.readlines()
    dim_est = np.zeros((self.max_objs, 3), dtype=np.float32)
    ori_est = np.zeros((self.max_objs, 3, 3), dtype=np.float32)
    pos_est = np.zeros((self.max_objs, 3), dtype=np.float32)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    ori_DATA=[]
    ori_est_scalar  = np.zeros((self.max_objs), dtype=np.float32)
    for object_index in range(len(detections)):
      data_str = detections[object_index]
      data_list = data_str.split()
      # if data_list[0] != 'Car':
      #   continue
      dim_est[object_index,0] = float(data_list[8])
      dim_est[object_index, 1] = float(data_list[9])
      dim_est[object_index, 2] = float(data_list[10])
      pos_est[object_index,0]=float(data_list[11])
      pos_est[object_index, 1] = float(data_list[12])
      pos_est[object_index, 2] = float(data_list[13])
      ori_est_scalar[object_index] = float(data_list[14])
      ori_est[object_index] = self.E2R(float(data_list[14]))
      reg_mask[object_index] = 1
      ori_DATA_s = [data_list[0]]
      ori_DATA_s = ori_DATA_s + [float(ii) for ii in data_list[1:]]
      ori_DATA.append(ori_DATA_s)
      # The orientation definition is inconsitent with right-hand coordinates in kitti

    meta['dim_est'] = torch.from_numpy(dim_est).unsqueeze(0).to(self.opt.device)
    meta['pos_est'] = torch.from_numpy(pos_est).unsqueeze(0).to(self.opt.device)
    meta['ori_est'] = torch.from_numpy(ori_est).unsqueeze(0).to(self.opt.device)
    meta['ori_est_scalar'] = torch.from_numpy(ori_est_scalar).unsqueeze(0).to(self.opt.device)
    meta['reg_mask'] = torch.from_numpy(reg_mask).unsqueeze(0).to(self.opt.device)

    meta['ori_DATA']=ori_DATA
  def run(self, image_or_path_or_tensor_l,image_or_path_or_tensor_r, mono_est,meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor_l, np.ndarray):
      image = image_or_path_or_tensor_l
    elif type(image_or_path_or_tensor_l) == type (''):
      self.image_path=image_or_path_or_tensor_l
      image_l = cv2.imread(image_or_path_or_tensor_l)
      image_r = cv2.imread(image_or_path_or_tensor_r)

      calib_path=os.path.join(self.opt.calib_dir,image_or_path_or_tensor_l[-10:-3]+'txt')
      calib=self.read_clib(calib_path)
      calib=torch.from_numpy(calib).unsqueeze(0).to(self.opt.device)
      calib3 = self.read_clib3(calib_path)
      calib3 = torch.from_numpy(calib3).unsqueeze(0).to(self.opt.device)
    else:
      image = image_or_path_or_tensor_l['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor_l
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    # cv2.imshow('s',image_l)
    # cv2.waitKey(0)
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images_l, meta = self.pre_process(image_l, scale, meta)
        images_r, _ = self.pre_process(image_r, scale, meta)
        meta['imag_name']=image_or_path_or_tensor_l[-10:-3]
        meta['trans_output_l']=meta['trans_output_l'].to(self.opt.device)
        meta['trans_output_r'] = meta['trans_output_r'].to(self.opt.device)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      meta['calib_l']=calib
      meta['calib_r'] = calib3
      images_l = images_l.to(self.opt.device)
      images_r = images_r.to(self.opt.device)
      self.read_est_from_mono(mono_est,meta)
      torch.cuda.synchronize()


      meta['input']=images_l
      meta['input_r']=images_r
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      output, dets, forward_time = self.process(meta,return_time=True)
      net_time += forward_time #- pre_process_time
      torch.cuda.synchronize()

      decode_time = time.time()
      dec_time += decode_time - pre_process_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images_l, dets, output, scale)
      
      #dets = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    #results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image_l, image_r, dets, calib)
    
    return {'results': dets, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}