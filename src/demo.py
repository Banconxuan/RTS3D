from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from detectors.RTS3D_infer import RTS3DDetector
from opts import opts
import shutil
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net']

def demo(opt):
    #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.faster=False
    Detector = RTS3DDetector
    detector = Detector(opt)
    if os.path.exists(opt.results_dir):
        shutil.rmtree(opt.results_dir, True)
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      if opt.demo[-3:]=='txt':
          with open(opt.demo,'r') as f:
              lines = f.readlines()
          ls = os.listdir(opt.mono_path)
          image_l_names=[os.path.join(opt.data_dir+'/kitti/image/',img[:6]+'.png') for img in ls]
          image_r_names = [os.path.join(opt.data_dir + '/kitti/image/', "{:06d}".format(int(float(img[:6])+7481))+'.png') for img in ls]
          mono_est= [os.path.join(opt.mono_path,img[:6]+'.txt') for img in ls]
      else:
        image_names = [opt.demo]
    time_tol = 0
    num = 0
    for (image_name_l,image_name_r,mono_est) in zip(image_l_names,image_r_names,mono_est):
      num+=1
      ret = detector.run(image_name_l,image_name_r,mono_est)
      time_str = ''
      for stat in time_stats:
          time_tol=time_tol+ret[stat]
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
      print(time_str)
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
