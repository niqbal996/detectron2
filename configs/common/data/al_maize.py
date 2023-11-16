from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
# import albumentations as A
from detectron2.data import transforms as T
# from detectron2.data.transforms import AlbumentationsWrapper
from detectron2.evaluation import COCOEvaluator
from detectron2.data.samplers import RandomSubsetTrainingSampler
dataloader_dict = {}
for percentage in range(10, 100, 10):
    idx = int(percentage / 10)
    dataloader_dict[idx] = OmegaConf.create()
    dataloader_dict[idx].train = L(build_detection_train_loader)(
        dataset=L(get_detection_dataset_dicts)(names="maize_syn_v2_train"),
        mapper=L(DatasetMapper)(
            is_train=True,
            augmentations=[
                L(T.ResizeShortestEdge)(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    sample_style="choice",
                    max_size=1333,
                ),
            ],
            image_format="BGR",
            use_instance_mask=True,
        ),
        sampler=L(RandomSubsetTrainingSampler)(
            size=1000, # TODO add dynamically
            subset_ratio=(percentage/100), # subset size 0.1 x 1000 = 100 images
            seed_shuffle=1,
            seed_subset=1,
        ),
        total_batch_size=16,
        num_workers=8,
    )

    dataloader_dict[idx].test = L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="maize_real_v2_val", filter_empty=True),
        mapper=L(DatasetMapper)(
            is_train=True,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
            ],
            image_format="${...train.mapper.image_format}",
        ),
    )

    dataloader_dict[idx].evaluator = L(COCOEvaluator)(
        dataset_name="${..test.dataset.names}",
        max_dets_per_image=500,
    )
