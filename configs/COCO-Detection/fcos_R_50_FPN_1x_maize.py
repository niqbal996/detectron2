from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.001

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 2
dataloader.train.total_batch_size = 12
dataloader.train.num_workers = 10

dataloader.test.batch_size = 8
dataloader.test.num_workers = 6

train.max_iter = 20000
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.output_dir = "/home/niqbal/trainers/detectron2/fcos_validation_loss_best_model"