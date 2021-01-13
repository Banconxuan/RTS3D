from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np



from models.focalloss import BCEFocalLoss

from .base_trainer import BaseTrainer
import torch.nn.functional as F
import iou3d_cuda
from utils import kitti_utils_torch as kitti_utils

class CornerLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CornerLoss, self).__init__()

        self.opt = opt
        self.iou_loss=torch.nn.BCEWithLogitsLoss()
        self.iou_loss =BCEFocalLoss()
        self.coners_const = torch.Tensor(
            [[0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5,0],
             [0, 0, 0, 0, -1, -1, -1, -1,-0.5],
             [0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,0]]
        )

        self.rampup_coor = self.exp_rampup(50)

    def exp_rampup(self,rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""

        def warpper(epoch):
            if epoch < rampup_length:
                epoch = np.clip(epoch, 0.0, rampup_length)
                phase = 1.0 - epoch / rampup_length
                return float(np.exp(-5.0 * phase * phase))
            else:
                return 1.0

        return warpper
    def boxes_iou_bev(self,boxes_a, boxes_b):
        """
        :param boxes_a: (M, 5)
        :param boxes_b: (N, 5)
        :return:
            ans_iou: (M, N)
        """
        ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

        iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

        return ans_iou

    def boxes_iou3d_gpu(self,boxes_a, boxes_b):
        """
        :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
        :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
        :return:
            ans_iou: (M, N)
        """
        boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
        boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

        # bev overlap
        overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

        # height overlap
        boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
        boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
        boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
        boxes_b_height_max = boxes_b[:, 1].view(1, -1)

        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

        # 3d iou
        overlaps_3d = overlaps_bev * overlaps_h

        vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
        vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

        iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

        return iou3d

    def param2corner(self,pos,h,w,l,ry):
        #pos=pos.transpose(1,0)#3,K
        # h = h.transpose(1, 0)#1,K
        # w = w.transpose(1, 0)#1,K
        # l = l.transpose(1, 0)#1,K
        pos = pos.unsqueeze(2).expand(l.size(0),3,9)
        dim=torch.cat([l,h,w],dim=1).unsqueeze(2)
        dim=dim.expand(l.size(0),3,9)#K,3,9
        corner = self.coners_const.cuda().unsqueeze(0).expand(l.size(0),3,9)  # K,3,9
        corner=dim*corner

        R=pos.new_zeros(pos.size(0),3,3)
        R[:,0,0]=torch.cos(ry)
        R[:, 0, 2] = torch.sin(ry)
        R[:, 1, 1] = 1
        R[:, 2, 0] = -torch.sin(ry)
        R[:, 2, 1] = torch.cos(ry)#K,3,3
        corner=R.bmm(corner)+pos#K,3,9

        return corner


    def forward(self, outputs, batch, epoch=None):


        dim_real = batch['dim_real'][:, :]
        pos_real = batch['pos_real'][:, :]
        ori_real = batch['ori_real'][:].unsqueeze(1)

        dim_est = batch['dim_est'][:, :]
        pos_est = batch['pos_est'][:, :]
        ori_est = batch['ori_est_scalar'][:].unsqueeze(1)


        iou3d_input_gt = torch.cat([pos_real,dim_real,ori_real],dim=1)
        ES_EST = torch.cat([pos_est, dim_est, ori_est], dim=1)

        iou3d_input_est = ES_EST +outputs[:,:7]

        next_est = iou3d_input_est
        box_score = self.boxes_iou3d_gpu(iou3d_input_est.detach(), iou3d_input_gt.detach())
        box_score = torch.diag(box_score)

        pos_pre = iou3d_input_est[:,:3]
        h_pre = iou3d_input_est[:,3:4]
        w_pre = iou3d_input_est[:, 4:5]
        l_pre = iou3d_input_est[:, 5:6]
        ry_pre = iou3d_input_est[:,6]
        corner_pre = self.param2corner(pos_pre, h_pre, w_pre, l_pre, ry_pre)

        pos_g = iou3d_input_gt[:, :3]
        h_g = iou3d_input_gt[:, 3:4]
        w_g = iou3d_input_gt[:, 4:5]
        l_g = iou3d_input_gt[:, 5:6]
        ry_g = iou3d_input_gt[:, 6]
        corner_g = self.param2corner(pos_g, h_g, w_g, l_g, ry_g)

        l2 = corner_g - corner_pre#K,3,9
        l2 = torch.norm(l2, p=2, dim=1)#K,9
        l2 = torch.log(l2 + 1)
        loss_reg = l2.mean()

        box_score = box_score.detach()
        box_score = 2*box_score-0.5
        box_score = torch.clamp(box_score,0,1)
        loss_cls = self.iou_loss(outputs[:,7],box_score)
        loss = self.rampup_coor(epoch)*loss_cls+loss_reg

        loss_pos = F.l1_loss(iou3d_input_est[:,:3],iou3d_input_gt[:,:3])
        loss_dim = F.l1_loss(iou3d_input_est[:, 3:6], iou3d_input_gt[:, 3:6])
        loss_ori = F.l1_loss(iou3d_input_est[:, 6], iou3d_input_gt[:, 6])
        loss_stats = {'loss': loss,'loss_cls':loss_cls,'loss_reg':loss_reg,'box_score':box_score,'loss_pos':loss_pos,'loss_dim':loss_dim,'loss_ori':loss_ori}

        return loss, loss_stats,next_est
class RTS3DTrainer(BaseTrainer):
    def __init__(self, opt, model_image,model_cloudpoint, optimizer_image,optimizer_cloudpoint):
        super(RTS3DTrainer, self).__init__(opt, model_image,model_cloudpoint, optimizer_image,optimizer_cloudpoint)

    def _get_losses(self, opt):
        loss_states = ['loss','loss_cls','loss_reg','box_score','loss_pos','loss_dim','loss_ori']
        loss = CornerLoss(opt)
        return loss_states, loss

