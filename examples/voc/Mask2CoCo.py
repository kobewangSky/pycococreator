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
import cv2

ROOT_DIR = 'train/voc/VOC2007/'

IMAGE_DIR = os.path.join(ROOT_DIR, "JPEGImages")
SEGMENT_DIR = os.path.join(ROOT_DIR, "SegmentationClass")

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


    for root, _, files in os.walk(SEGMENT_DIR):
        files = sorted(files)
        Segment_files = filter_for_image(root, files)

        for Segment_filename in Segment_files:
            image_filename =  Segment_filename.replace('SegmentationClass', 'JPEGImages').replace('.png', '.jpg')

            image = cv2.imread(image_filename)
            mask = cv2.imread(Segment_filename)
            height, width, channels = mask.shape


            image_info = pycococreatortools.create_image_info( image_id, os.path.basename(image_filename), (width, height))
            coco_output["images"].append(image_info)


            Color_list = [x['color'] for x in CATEGORIES ]
            binarymask_list = mask2polygon(mask, Color_list)

            for class_id, binarymask in binarymask_list.items():
                category_info = {'id': class_id, 'is_crowd': False}
                annotation_info = pycococreatortools.create_annotation_info( segmentation_id, image_id, category_info, binarymask, (width, height), tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/VOC2007.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
