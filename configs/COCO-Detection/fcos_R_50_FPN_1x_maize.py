from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
# from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 1e-4

# model.backbone.bottom_up.freeze_at = 2
model.num_classes = 2
# dataloader.train.total_batch_size = 70  # 28 on 40GB and 80 on 80 GB
# dataloader.train.num_workers = 50
# dataloader.test.batch_size = 60
# dataloader.test.num_workers = 40


dataloader.train.total_batch_size = 28  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 28
dataloader.test.batch_size = 20
dataloader.test.num_workers = 20

train.output_dir = "/netscratch/naeem/fcos_pretrained_not_frozen_transformed_1e-4"
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.max_iter = 15000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=10)
