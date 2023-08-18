from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.maize import dataloader
from ..common.models.retinanet import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
model.backbone.bottom_up.freeze_at = 2
optimizer.lr = 0.001

model.num_classes = 2
dataloader.train.total_batch_size = 50  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 40
train.output_dir = "/netscratch/naeem/retinanet_maize_no_augmentation_synthetic_data_frozen"
# train.output_dir = "/netscratch/naeem/retinanet_maize_no_augmentation"
train.max_iter = 30000
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
