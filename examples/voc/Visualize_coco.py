
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import cv2

image_directory = '/home/kobe/maskrcnn-benchmark/datasets/ACRV/frames/000000/'
annotation_file = '/home/kobe/maskrcnn-benchmark/datasets/ACRV/ACRV_000000.json'

# image_directory = './train/voc/VOC2007/JPEGImages/'
# annotation_file = './train/voc/VOC2007/VOC2007.json'


example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])

category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)

for id,img in enumerate(image_ids):

    # image_data = example_coco.loadImgs(img)[0]
    #
    # # load and display instance annotations
    # image = io.imread(image_directory + image_data['file_name'])
    # plt.imshow(image); plt.axis('off')
    # pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    # annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    # annotations = example_coco.loadAnns(annotation_ids)
    # example_coco.showAnns(annotations)
    # name = str(id) + '.jpg'
    # name = os.path.join('./test', name)
    # plt.savefig(name)
    # print()

    image_data = example_coco.loadImgs(img)[0]
    image = cv2.imread(image_directory + image_data['file_name'])
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)

    for ann in annotations:
        x1 = ann['bbox'][0]
        y1 = ann['bbox'][1]
        w = ann['bbox'][2]
        h = ann['bbox'][3]

        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255,0,0), 2)
        name = str(id) + '.jpg'
        name = os.path.join('./test', name)
        cv2.imwrite(name, image)
        print()


