# from ..common.optim import SGD as optimizer
from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
# from ..common.data.maize import dataloader
from ..common.data.al_maize import dataloader_dict 
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

optimizer.lr = 1e-4
# dataloader.train.mapper.use_instance_mask = False

train_batch_size = 30
train_workers = 30
test_batch_size = 20
test_workers = 20

for al_iter in dataloader_dict:
    dataloader_dict[al_iter].train.total_batch_size = train_batch_size
    dataloader_dict[al_iter].train.num_workers = train_workers
    dataloader_dict[al_iter].test.batch_size = test_batch_size
    dataloader_dict[al_iter].test.num_workers = test_workers

# model.backbone.bottom_up.freeze_at = 2
model.roi_heads.num_classes = 2
del model.roi_heads.mask_in_features
del model.roi_heads.mask_pooler
del model.roi_heads.mask_head

train.max_iter = 5000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=10)
train.output_dir = '/netscratch/naeem/frcnn_syn_maize_AL_random_sampler'

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"