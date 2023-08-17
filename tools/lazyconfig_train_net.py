#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    # optim = torch.optim.AdamW(model.parameters(), lr=0.01)   # TODO: REmove afterDEBUG
    train_loader = instantiate(cfg.dataloader.train)
    val_loader = instantiate(cfg.dataloader.test)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    # trainer.register_hooks(
    #     [
    #         hooks.IterationTimer(),
    #         hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
    #         hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
    #         if comm.is_main_process()
    #         else None,
    #         hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
    #         hooks.PeriodicWriter(
    #             default_writers(cfg.train.output_dir, cfg.train.max_iter),
    #             period=cfg.train.log_period,
    #         )
    #         if comm.is_main_process()
    #         else None,
    #     ]
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, 
                                                           mode='min',      # Validation loss has to be reduced for scheduler hence min mode. 
                                                           factor=0.1,     
                                                           patience=500, # TODO chcck this based on validation loss
                                                           threshold_mode='rel',
                                                           cooldown=2000,   # TODO check this based on validation loss
                                                           min_lr=0,
                                                           verbose=True)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            # hooks.LRScheduler(scheduler=scheduler, metric=LossEvalHook(100,  # TODO Set = 1000 after debugging
            #                                                             model, 
            #                                                             val_loader)),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer),
            hooks.BestCheckpointer(eval_period=100, checkpointer=checkpointer, 
                                   val_metric='validation_loss', mode='min', 
                                   file_prefix='model_best_mAP50')
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.LossEvalHook(99,  # TODO Set = 1000 after debugging
                         model, 
                         val_loader),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            # hooks.TorchProfiler(
            #  lambda trainer: 10 < trainer.iter < 20, cfg.train.output_dir
            # )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def register_dataset():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("maize_syn_v2_train", {},
                            "/netscratch/naeem/maize_syn_v3/instances_train_2022.json",
                            "/netscratch/naeem/maize_syn_v3/data")
    register_coco_instances("maize_real_v2_val", {},
                            "/netscratch/naeem/maize_real_all_days/coco_annotations/all_data.json",
                            "/netscratch/naeem/maize_real_all_days/data")


def main(args):
    cfg = LazyConfig.load(args.config_file) 
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    register_dataset()

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
