#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from configs import CATEGORIES
from configs import CLASSES
import cv2



ROOT_DIR = '/home/kobe/maskrcnn-benchmark/datasets/ACRV/'

IMAGE_DIR = os.path.join(ROOT_DIR, "frames", '000003/')
Data_DIR = os.path.join(ROOT_DIR, "ground_truth", '000003/' )


INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]




def filter_for_image(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def mask2polygon(mask, Color_list):

    polygonlist = {}
    for index, color in enumerate(Color_list):
        lower = np.array(color, dtype="uint8")
        mask_temp = mask.copy()
        mask_temp = cv2.inRange(mask_temp, lower, lower)
        if(mask_temp.sum()):
            binary_mask = np.asarray(mask_temp).astype(np.uint8)
            polygonlist[index + 1] = binary_mask

    return polygonlist



def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1


    ACRVdata_path = Data_DIR + 'labels.json'

    ana_id = 1
    image_id = 1

    width = 640
    height = 480

    with open(ACRVdata_path) as fp:
        labels = json.load(fp)
    for image_index, image_name in sorted((int(l), l) for l in labels.keys()):
        image_data = labels[image_name]
        image_name = image_name + '.png'
        image_info = pycococreatortools.create_image_info( image_id, image_name, (width, height))
        coco_output["images"].append(image_info)

        if len(image_data) > 0:
            for instance_name in sorted(image_data.keys()):
                if not instance_name.startswith('_'):
                    detection_data = image_data[instance_name]
                    if detection_data['class'] == 'none':
                        continue
                    class_id = CLASSES.index(detection_data['class'])

                    category_info = {'id': class_id, 'is_crowd': False}

                    boxtemp = [detection_data['bounding_box'][0], detection_data['bounding_box'][1], detection_data['bounding_box'][2] - detection_data['bounding_box'][0], detection_data['bounding_box'][3] - detection_data['bounding_box'][1]]

                    annotation_info = pycococreatortools.create_annotation_info_withoutmask( ana_id, image_id, category_info, bounding_box = boxtemp, image_size = (width, height), tolerance=2)
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    ana_id = ana_id + 1

        image_id = image_id + 1


    with open('{}/ACRV_000003.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
