from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset

from trains.RTS3D_trainer import RTS3DTrainer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset()
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  logger = Logger(opt)

  # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model_image,model_point = create_model(opt)
  optimizer_image = torch.optim.Adam(model_image.parameters(), opt.lr)
  opt.lr_c=opt.lr
  optimizer_point = torch.optim.Adam(model_point.parameters(), opt.lr_c)
  start_epoch = 0
  if opt.load_model != '':
    model_image, optimizer_image, start_epoch = load_model(
      model_image, opt.load_model, 'image_model', optimizer_image, opt.resume, opt.lr, opt.lr_step)
    model_cloudpoint, optimizer_cloudpoint, _= load_model(
        model_point, opt.load_model, 'point_model', optimizer_point, opt.resume, opt.lr, opt.lr_step)


  image_chunk = opt.chunk_sizes
  batch_size = 0
  cloud_chunk = []
  for i in image_chunk:
    batch_size += i
    cloud_chunk.append(i*opt.max_objs)
  #cloud_chunk=[image_chunk[0]*opt.max_objs,image_chunk[1]*opt.max_objs]
  Trainer = RTS3DTrainer
  trainer = Trainer(opt, model_image,model_point, optimizer_image,optimizer_point)
  trainer.set_device(opt.gpus, image_chunk, opt.device,'image')
  trainer.set_device(opt.gpus, cloud_chunk, opt.device,'point')


  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'),
      batch_size = batch_size,
      shuffle = True,
      num_workers = opt.num_workers,
      pin_memory = True,
      drop_last = True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size = batch_size,
      shuffle = True,
      num_workers = opt.num_workers,
      pin_memory = True,
      drop_last = True
  )

  print('Starting training...')
  best = 1e10
  iter_num = [0]
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    # with torch.no_grad():
    #       log_dict_val, preds = trainer.val(epoch, val_loader)

    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                 epoch, model_image, optimizer_image, model_point, optimizer_point)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    else:
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model_image, optimizer_image, model_point, optimizer_point)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                   epoch, model_image, optimizer_image, model_point, optimizer_point)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      lr_c = opt.lr_c * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer_image.param_groups:
          param_group['lr'] = lr
      for param_group in optimizer_point.param_groups:
          param_group['lr'] = lr_c
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)