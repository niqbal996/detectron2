from detrex.config import get_config
from .models.dino_swin_tiny_224 import model

# get default config
dataloader = get_config("common/data/maize.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "/home/niqbal/git/detrex/pretrained/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"
# train.init_checkpoint = "/home/niqbal/git/detrex/output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune/model_final.pth"
train.output_dir = "./output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune_with_augmentations_p_80"

# max training iterations
train.max_iter = 90000

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 2 # weed, maize
# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
