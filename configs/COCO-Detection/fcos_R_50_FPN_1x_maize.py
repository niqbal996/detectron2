from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
# from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.001

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 2
dataloader.train.total_batch_size = 70  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 50
dataloader.test.batch_size = 70
dataloader.test.num_workers = 50

train.output_dir = "/netscratch/naeem/fcos_r50_FPN_1x_maize_pretrained_no_freeze_synthetic_adam_optimizer"
train.max_iter = 20000
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
# train.init_checkpoint = "/media/niqbal/T7/trainers/gil_paper/fcos_R_1600_valid_500/model_final.pth"
