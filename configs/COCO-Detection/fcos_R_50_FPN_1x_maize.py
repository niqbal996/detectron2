from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train
# from detectron2.config import LazyCall as L
# solver = L()
dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 2
dataloader.train.total_batch_size = 4



train.init_checkpoint = "/media/niqbal/T7/trainers/gil_paper/fcos_R_1600_valid_500/model_final.pth"
# print('hold')