import random
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torchvision
import albumentations as A

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    w, h = bbox[2], bbox[3]
    x_min, y_min = int(bbox[0] - w/2), int(bbox[1] - h/2)
    x_max, y_max = int(bbox[0] + w/2), int(bbox[1] + h/2)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, orig=True):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    # plt.imshow(img)
    if orig:
        plt.imsave('./orig.png', img)
    else:
        plt.imsave('./motion_blurred.png', img)
    # plt.show()

image = cv2.imread('/home/niqbal/0000.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
width = image.shape[1]
height = image.shape[0]

with open('/home/niqbal/0000.txt', 'r') as f:
    bboxes = [line.rstrip() for line in f]
    category_ids = [0] * len(bboxes)
    # convert each box to a list
    for box, i in zip(bboxes, range(len(bboxes))):
        bboxes[i] = list(map(float, box.split()))[1:]
        category_ids[i] = int(box[0])

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {0: 'weeds', 1: 'maize', 2: 'bark'}
tmp = [[i[0] * width, i[1] * height, i[2] * width, i[3] * height] for i in bboxes]
visualize(image, tmp, category_ids, category_id_to_name, orig=True)
transform = A.Compose([
     # A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2),
     # A.Blur(blur_limit=(7, 9), always_apply=True),
     # A.AdvancedBlur(blur_limit=(7, 9), always_apply=True),
     A.MotionBlur(blur_limit=(7, 9), always_apply=True, allow_shifted=True)
],
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
)
t = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET)
pil_image = Image.fromarray(np.uint8(image)).convert('RGB')

for i in range(20):
    x = t(pil_image)
    x = np.array(x)
    plt.figure()
    plt.imshow(x)
    plt.show()
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
height = transformed['image'].shape[0]
width = transformed['image'].shape[1]
tmp = [[i[0] * width, i[1] * height, i[2] * width, i[3] * height] for i in transformed['bboxes']]
visualize(
    transformed['image'],
    tmp,
    transformed['category_ids'],
    category_id_to_name,
    orig=False
)
# plt.show()