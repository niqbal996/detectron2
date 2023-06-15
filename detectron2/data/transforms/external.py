import numpy as np
import sys
from typing import Tuple
import torch
from fvcore.transforms.transform import NoOpTransform
from PIL import Image

from .augmentation import Augmentation, _transform_to_aug
from .transform import ExtentTransform, ResizeTransform, RotationTransform
from albumentations import *

class AlbumentationsTX(Augmentation):
    def __init__(self, augmentation, p=0.5):
        self.aug = augmentation
        self.prob = p
        self.transform = augmentation
    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            return self.transform(image)
        else:
            return NoOpTransform()