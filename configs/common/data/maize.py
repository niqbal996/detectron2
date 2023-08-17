from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
import albumentations as A
from detectron2.data import transforms as T
from detectron2.data.transforms import AlbumentationsWrapper
from detectron2.evaluation import COCOEvaluator
import torchvision
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_syn_v2_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            # AlbumentationsWrapper(A.HorizontalFlip(p=0.1)),
<<<<<<< Updated upstream
            # AlbumentationsWrapper(A.MotionBlur(blur_limit=13, p=0.8)),
            # AlbumentationsWrapper(A.Blur(blur_limit=13, p=0.8)),
            # AlbumentationsWrapper(A.MedianBlur(blur_limit=13, p=0.8)),
            # AlbumentationsWrapper(A.ToGray(p=0.01)),
            # AlbumentationsWrapper(A.CLAHE(p=0.3)),
            # AlbumentationsWrapper(A.RandomBrightnessContrast(p=0.01)),
            # AlbumentationsWrapper(A.ImageCompression(quality_lower=75, p=0.5))
=======
            # AlbumentationsWrapper(A.MotionBlur(blur_limit=13, p=0.2)),
            # # AlbumentationsWrapper(A.Blur(blur_limit=13, p=0.8)),
            # # AlbumentationsWrapper(A.MedianBlur(blur_limit=13, p=0.8)),
            # AlbumentationsWrapper(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.1)),
            # AlbumentationsWrapper(A.ToGray(p=0.01)),
            # AlbumentationsWrapper(A.CLAHE(p=0.3)),
            # AlbumentationsWrapper(A.RandomBrightnessContrast(p=0.01)),
            # AlbumentationsWrapper(A.ImageCompression(quality_lower=75, p=0.2))
>>>>>>> Stashed changes
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
<<<<<<< Updated upstream
    num_workers=8,
=======
    num_workers=20,
>>>>>>> Stashed changes
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_real_v2_val", filter_empty=True),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=500,
)
