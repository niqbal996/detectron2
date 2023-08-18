from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
# from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.001

# model.backbone.bottom_up.freeze_at = 2
model.num_classes = 2
# dataloader.train.total_batch_size = 70  # 28 on 40GB and 80 on 80 GB
# dataloader.train.num_workers = 50
# dataloader.test.batch_size = 60
# dataloader.test.num_workers = 40


dataloader.train.total_batch_size = 70  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 60
dataloader.test.batch_size = 60
dataloader.test.num_workers = 50

train.output_dir = "/netscratch/naeem/fcos_r50_FPN_1x_maize_scratch_not_frozen_transformed_short"
# train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.max_iter = 20000
train.eval_period = 500
train.log_period = 10
train.checkpointer = dict(period=2000, max_to_keep=10)
