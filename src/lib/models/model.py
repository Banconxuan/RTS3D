from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from models.image_feature_generater import get_image_feature
from models.pointNet import PointNetDetector


def create_model(opt):
  get_model = get_image_feature
  model_image = get_model(num_layers=18,opt=opt)
  model_cloudpoint = PointNetDetector(8,opt=opt)
  return model_image, model_cloudpoint

def load_model(model, model_path, model_name, optimizer=None, resume=False,
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  if model_name == 'image_model':
      checkpoint = checkpoint['image_model']
  elif model_name == 'point_model':
      checkpoint = checkpoint['point_model']
  else:
      print('no existing model')
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model_image, optimizer_image=None, model_point=None, optimizer_point=None):
  if isinstance(model_image, torch.nn.DataParallel):
    state_dict_image = model_image.module.state_dict()
  else:
    state_dict_image = model_image.state_dict()
  data_image = {'epoch': epoch,
          'state_dict': state_dict_image}
  if not (optimizer_image is None):
    data_image['optimizer'] = optimizer_image.state_dict()
  if isinstance(model_point, torch.nn.DataParallel):
    state_dict_point = model_point.module.state_dict()
  else:
    state_dict_point = model_point.state_dict()
  data_point = {'epoch': epoch,
                'state_dict': state_dict_point}
  if not (optimizer_point is None):
      data_point['optimizer'] = optimizer_point.state_dict()
  data={'image_model':data_image,'point_model':data_point}
  torch.save(data, path)

