from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.car_pose import CarPoseDataset
from .dataset.kittihp import KITTIHP

def get_dataset():
  class Dataset(KITTIHP, CarPoseDataset):
    pass
  return Dataset
  
