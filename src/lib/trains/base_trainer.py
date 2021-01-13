from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from models.embedding_space_generater import featuremap2gridpoint
from models.utils import _transpose_and_gather_feat
import numpy as np
import copy
def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper
class ModelWithLoss(torch.nn.Module):
  def __init__(self, model_image, model_point, loss):
    super(ModelWithLoss, self).__init__()
    self.model_image = model_image
    self.model_point = model_point
    self.loss = loss
    self.time_up=exp_rampup(50)

  def exp_rampup(self, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""

        def warpper(epoch):
            if epoch < rampup_length:
                epoch = np.clip(epoch, 0.0, rampup_length)
                phase = 1.0 - epoch / rampup_length
                return float(np.exp(-5.0 * phase * phase))
            else:
                return 1.0

        return warpper
  def forward(self, batch, phase, epoch=None, opt=None):#point_data, phase, epoch=epoch, self.opt
    image_ret = self.model_image(batch)
    batch.update(image_ret)
    point_data = featuremap2gridpoint(batch, phase, opt)
    loss_batch = point_data['input_batch']
    outputs = self.model_point(point_data)

    loss, loss_stats, next_est = self.loss(outputs, loss_batch, epoch)
    return  loss, loss_stats, next_est
class BaseTrainer(object):
  def __init__(
    self, opt, model_image, model_point, optimizer_image, optimizer_point):
    self.opt = opt
    self.optimizer_image = optimizer_image
    self.optimizer_point = optimizer_point

    self.loss_stats, self.loss= self._get_losses(opt)
    self.model_image = model_image
    self.model_point = ModelWithLoss(model_image, model_point, self.loss)#model_point
  def set_device(self, gpus, chunk_sizes, device,model_select):
    if len(gpus) > 1:
      if model_select=='image':
        self.model_image = DataParallel(
          self.model_image, device_ids=gpus,
          chunk_sizes=chunk_sizes).to(device)
      elif model_select=='point':
        self.model_point = DataParallel(
          self.model_point, device_ids=gpus,
          chunk_sizes=chunk_sizes).to(device)
      else:
        print("no support model")
    else:
      if model_select=='image':
        self.model_image = self.model_image.to(device)
      elif model_select=='point':
        self.model_point = self.model_point.to(device)
      else:
        print("no support model")

    if model_select == 'image':
      for state in self.optimizer_image.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.to(device=device, non_blocking=True)
    elif model_select == 'point':
      for state in self.optimizer_point.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.to(device=device, non_blocking=True)
    else:
      print("no support model")


  def run_epoch(self, phase, epoch, data_loader):
    model_image = self.model_image
    model_point = self.model_point
    if phase == 'train':
      model_image.train()
      model_point.train()
    else:
      model_image.eval()
      model_point.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format("3D detection", opt.exp_id), max=num_iters)
    end = time.time()

    for iter_id, batch in enumerate(data_loader):

      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      if iter_id == 8:
        a=1
      loss, loss_stats, next_est = model_point(batch, phase, epoch = epoch, opt=self.opt)#loss, loss_stats
      next_est = next_est.view(batch['reg_mask'].size(0),batch['reg_mask'].size(1),-1).detach()
      loss = loss.mean()
      if phase == 'train':
        self.optimizer_image.zero_grad()
        self.optimizer_point.zero_grad()
        loss.backward()
        self.optimizer_image.step()
        self.optimizer_point.step()

        batch['pos_est'] = next_est[:, :, :3]
        batch['dim_est'] = next_est[:, :, 3:6]
        ry = next_est[:, :, 6]
        R_yaw = batch['pos_est'].new_zeros(next_est.size(0), next_est.size(1), 3, 3)
        R_yaw[:, :, 0, 0] = torch.cos(ry)
        R_yaw[:, :, 0, 2] = torch.sin(ry)
        R_yaw[:, :, 1, 1] = 1
        R_yaw[:, :, 2, 0] = -torch.sin(ry)
        R_yaw[:, :, 2, 2] = torch.cos(ry)
        batch['ori_est'] = R_yaw
        batch['ori_est_scalar'] = ry
        loss, loss_stats, next_est = model_point(batch, phase, epoch=epoch, opt=self.opt)
        loss = 2*loss.mean()
        self.optimizer_image.zero_grad()
        self.optimizer_point.zero_grad()
        loss.backward()
        self.optimizer_image.step()
        self.optimizer_point.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()

      del loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)