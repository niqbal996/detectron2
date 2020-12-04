#!/usr/bin/env python3
import cv2
import numpy as np
import datetime
import json
import re
import fnmatch
import random
from PIL import Image

from os.path import join, split, basename, splitext
from os import makedirs, walk
from scipy.spatial import distance
from glob import glob
from pycococreatortools import pycococreatortools

class Sugarbeet2Coco:
    def __init__(self,
                 root='/home/robot/datasets/structured_cwc/',
                 data_split='valid'):
        self.root = root
        self.split = data_split
        self.classes = ['crop', 'weed']
        self.rgb_dir = join(self.root, self.split, 'img/*.png')
        self.label_dir = join(self.root, self.split, 'lbl/*.png')
        self.coco_instances = join(self.root, self.split, 'lbl_coco')
        makedirs(self.coco_instances, exist_ok=True)
        self.list_labels = glob(self.label_dir)
        self.INFO = None
        self.LICENSES = None
        self.CATEGORIES = None
        self.coco_config_template()

    def coco_config_template(self):
        self.INFO = {
            "description": "SugarBeet 2016 Dataset",
            "url": "https://www.ipb.uni-bonn.de/data/sugarbeets2016/",
            "version": "0.1.0",
            "year": 2016,
            "contributor": "https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chebrolu2017ijrr.pdf",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        self.LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-sa/4.0/"
            }
        ]

        self.CATEGORIES = [
            {
                'id': 1,
                'name': 'maize',
                'supercategory': 'crop',
            },
            {
                'id': 2,
                'name': 'dicot weeds',
                'supercategory': 'weed',
            },
        ]

    def filter_for_jpeg(self, root, files):
        file_types = ['*.jpeg', '*.jpg']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_png(self, root, files):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_annotations(self, root, files, image_filename):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = splitext(basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '.*'
        files = [join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(file_name_prefix, splitext(basename(f))[0])]

        return files

    def generate_coco_annotations(self):
        coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1
        classes = [x['supercategory'] for x in self.CATEGORIES]

        for root, _, files in walk(split(self.rgb_dir)[0]):
            image_files = self.filter_for_png(root, files)

            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, basename(image_filename), image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in walk(self.coco_instances):
                    annotation_files = self.filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        # print(annotation_filename)
                        for class_label in classes:
                            if class_label in annotation_filename:
                                class_id = classes.index(class_label) + 1
                        # class_id = [x['id'] for x in CATEGORIES][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        # category_info = {'id': 1, 'is_crowd': 0}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=0)

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

        with open(join(self.root, 'instances_{}2016.json'.format(self.split)), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

    def contour2corners(self, contour):
        top_left_x, top_left_y, width, height = cv2.boundingRect(contour)
        return [top_left_x, top_left_y, top_left_x + width, top_left_y + height]

    def dist(self, p1, p2):
        distance.euclidean(p1, p2)

    def rect_distance(self, contour1, contour2):
        """
        Calculates the minimum distance between two given contours
        Args:
            contour2:
            contour1:
        Returns:
            Returns minimum distance between contour1 and contour2
        """
        R1 = self.contour2corners(contour1)
        R2 = self.contour2corners(contour2)

        x1, y1, x1b, y1b = R1
        x2, y2, x2b, y2b = R2
        left = x2b < x1
        right = x1b < x2
        bottom = y2b < y1
        top = y1b < y2
        if top and left:
            return self.dist((x1, y1b), (x2b, y2))
        elif left and bottom:
            return self.dist((x1, y1), (x2b, y2b))
        elif bottom and right:
            return self.dist((x1b, y1), (x2, y2b))
        elif right and top:
            return self.dist((x1b, y1b), (x2, y2))
        elif left:
            return x1 - x2b
        elif right:
            return x2 - x1b
        elif bottom:
            x = y1 - y2b
            return y1 - y2b
        elif top:
            return y2 - y1b
        else:  # rectangles intersect
            return 0.

    # https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    @staticmethod
    def is_overlap(self, contour1, contour2):
        """
        Checks the two contours and returns True if they are overlapping
        Args:
            contour1:
            contour2:
        Returns:
            boolean output, True if overlapping, False if not
        """
        R1 = self.contour2corners(contour1)
        R2 = self.contour2corners(contour2)
        # If one rectangle is on left side of other
        if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
            return False
        else:
            return True

    def check_contours(self, contours):
        """
        Args:
            contours: contours generated by cv2.findcontours for segmentation masks for individual classes
        Returns:
            Unified contours i.e. the contours which are too close or overlapping are grouped into one contour
            TODO this can be further improved to generate better instance masks
        """
        def find_if_close(cnt1, cnt2):
            row1, row2 = cnt1.shape[0], cnt2.shape[0]
            for i in range(row1):
                for j in range(row2):
                    dist = np.linalg.norm(cnt1[i] - cnt2[j])
                    if abs(dist) < 50:
                        return True
                    elif i == row1 - 1 and j == row2 - 1:
                        return False

        LENGTH = len(contours)
        status = np.zeros((LENGTH, 1))

        for i, cnt1 in enumerate(contours):
            x = i
            if i != LENGTH - 1:
                for j, cnt2 in enumerate(contours[i + 1:]):
                    x = x + 1
                    dist = find_if_close(cnt1, cnt2)
                    if dist == True:
                        val = min(status[i], status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x] == status[i]:
                            status[x] = i + 1

        unified = []
        maximum = int(status.max()) + 1
        for i in range(maximum):
            pos = np.where(status == i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                hull = cv2.convexHull(cont)
                unified.append(hull)

        return unified

    def generate_instances(self):
        """
        Takes the semantic segmentation masks and converts them into instance masks
        for each class and image
        Returns: None, Generates instance masks for each segmentation mask in "lbl_coco" folder
        """
        for image_path, count in zip(self.list_labels, range(len(self.list_labels))):
            image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
            rgb_path = join(split(split(image_path)[0])[0], 'img', split(image_path)[1])
            # rgb_image = cv2.imread(rgb_path)
            color = (0, 0, 0)
            class_image = np.zeros_like(image)
            print('[INFO] Processed {} percent of the data \r'.format(round((count / len(self.list_labels)) * 100), 2),
                  flush=True)
            for class_label in self.classes:
                if class_label == 'crop':
                    class_image[image == 255] = image[image == 255]
                    # color = (0, 255, 0)
                else:
                    class_image[image == 128] = image[image == 128]
                    # color = (0, 0, 255)
                if np.count_nonzero(class_image) > 0:
                    contours, _ = cv2.findContours(class_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # reset
                    class_image = np.zeros_like(image)
                    overlap = False
                    # filter extra neighbouring contours and unify them into one contour
                    valid_contours = self.check_contours(contours)
                    for contour, instance in zip(valid_contours, range(len(valid_contours))):
                        instance_mask = np.zeros(image.shape, dtype=np.uint8)
                        x, y, w, h = cv2.boundingRect(contour)
                        instance_mask[y:y + h, x:x + w] = image[y:y + h, x:x + w]
                        dest = join(split(split(image_path)[0])[0],
                                    'lbl_coco',
                                    (basename(image_path)[:-4] + '_' + class_label + '_' + str(instance) + '.png'))
                        cv2.imwrite(dest, instance_mask)
                        # print('[INFO] Split the segmentation mask into instance: \n {}'.format(dest))
                        # cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)
                        # cv2.putText(rgb_image, class_label, (x + w + 10, y + h), 0, 0.3, color)
                else:
                    print('[INFO] No classes found in the current picture')

    def visualize_samples(self, number):
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        from detectron2.data.catalog import DatasetCatalog
        from detectron2.data.datasets import register_coco_instances

        register_coco_instances('sugar_beet_{}'.format(self.split), {},
                                join(self.root, 'instances_{}2016.json'.format(self.split)),
                                join(self.root, self.split, 'img'))
        # register_coco_instances('sugar_beet_{}'.format(self.split), {},
        #                         join(self.root, 'instances_{}2016.json'.format(self.split)),
        #                         join(self.root, self.split, 'img'))
        # register_coco_instances("sugar_beet_test", {},
        #                         "/home/robot/datasets/structured_cwc/instances_test2016.json",
        #                         "/home/robot/datasets/structured_cwc/test/img/")

        # visualize training data
        my_dataset_train_metadata = MetadataCatalog.get('sugar_beet_{}'.format(self.split))
        dataset_dicts = DatasetCatalog.get('sugar_beet_{}'.format(self.split))

        for d in random.sample(dataset_dicts, number):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow(vis.get_image()[:, :, ::-1])
            cv2.imshow('image', vis.get_image())
            cv2.waitKey()


if __name__ == '__main__':
    coco_converter = Sugarbeet2Coco(data_split='valid')
    coco_converter.generate_instances()
    coco_converter.generate_coco_annotations()
    # coco_converter.visualize_samples(5)



