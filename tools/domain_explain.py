# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import umap
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob
import os, json, cv2, random
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

data_dir = '/media/naeem/T7/datasets/custom_coco'
# dataframe = pd.read_csv("./filter_activations.csv")
# Initialize detectron model and load weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

umap_transform = umap.UMAP(low_memory=True, metric='euclidean')

# Initialize empty activations dictionary
layer_name = 'backbone.bottom_up.res5.2.conv2'
activations = {}
activations[layer_name] = []

# Register layer to fetch activation maps from it when predicting samples
def save_activation(name, mod, inp, out):
    out_put = out.detach().cpu().numpy().copy()
    # This is hardcoded at the moment
    activations[name].append(np.resize(out_put, (512, 10, 25)).ravel())

for name, m in predictor.model.named_modules():
    if name == layer_name:
        m.register_forward_hook(partial(save_activation, name))

# Iterate through class folders in custom dataset
for class_id in os.listdir(data_dir):
    print('[INFO] Processing class ID: {}'.format(class_id))
    images = glob(os.path.join(data_dir, class_id, 'data', '*.jpg'))
    print('[INFO] Found {} number of samples from class ID: {}'.format(len(images), class_id))
    for file_name in images:
        image = cv2.imread(file_name)
        outputs = predictor(image)
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # if len(outputs['instances']) != 0:
            # boxes = outputs["instances"]._fields['pred_boxes'].to("cpu").tensor.numpy()[0]
            # out = v.draw_box(boxes)
            # cv2.imshow('fig', out.get_image())
            # cv2.waitKey()


embeddings = np.zeros((len(activations[layer_name]), 512 * 10 * 25), dtype=float)

for item, sample in zip(activations[layer_name],
                        range(len(activations[layer_name]))):
    embeddings[sample, :] = item

np.save('x.npy', embeddings)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# pca_result = pca.fit(embeddings)
print('[INFO] Loading embeddings from numpy array . . .')
embeddings = np.load('x.npy')
palette = ['green', 'orange', 'brown']
print('[INFO] Calculating UMAP transform . . .')
result = umap_transform.fit_transform(embeddings)
sns.scatterplot(result[:, 0], result[:, 1],
                s=2,
                palette=palette)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection', fontsize=12)
plt.show()
print('hold')
