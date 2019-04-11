import os
import glob

import json
import shutil
import cv2
from pycococreatortools import pycococreatortools
import datetime
from configs import CATEGORIES
from configs import CLASSES


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


IDTransform = {'plant': 'potted plant',
               'oven': 'oven',
                'apple' : 'apple',
               'banana':'banana',
               'bed':'bed',
               'book': 'book',
               'bottle':'bottle',
               'bowl':'bowl',
               'chair':'chair',
               'clock':'clock',
               'cup':'cup',
               'fork':'fork',
               'keyboard':'keyboard',
               'knife':'knife',
               'laptop':'laptop',
               'microwave':'microwave',
               'mouse':'mouse',
               'orange':'orange',
               'person':'person',
               'refrigerator':'refrigerator',
               'sink':'sink',
               'sofa':'couch',
               'spoon':'spoon',
               'table':'dining table',
               'toaster':'toaster',
               'wine':'wine glass',
               'toilet':'toilet',
               'tv':'television',

               }




_testDir = os.path.join(os.getcwd(), 'Testimage')

def removecoverdata(boundingboxlist, ymin, xmin, ymax, xmax, lower):

    needremove = []
    Covered = False
    for it in boundingboxlist:
        if it['xmin'] < xmin and it['ymin'] < ymin and it['xmax'] > xmax and it['ymax'] > ymax:
            Covered = True
            break
        if it['xmin'] > xmin and it['ymin'] > ymin and it['xmax'] < xmax and it['ymax'] < ymax:
            needremove.append(it)
            continue

    if Covered == False:
        boundingboxlist.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'class': lower})

    for it in needremove:
        boundingboxlist.remove(it)

    return boundingboxlist

clamp = lambda n, minn, maxn :max(min(maxn, n), minn)

if __name__ == '__main__':

    ROOT_DIR = '/home/kobe/maskrcnn-benchmark/datasets/Virtual/'

    RawDir = os.path.join(ROOT_DIR, 'Raw')

    ImageTargetDir = os.path.join(ROOT_DIR, 'image')

    listdir = os.listdir(RawDir)

    FileNameIndex = 0

    for listdir_temp in listdir:
        TempDir = os.path.join(RawDir, listdir_temp)

        if os.path.isdir(TempDir) != True:
            continue

        #Imagegroup = sorted(glob.glob(os.path.join(ImageTargetDir, '*.png')))
        #Imagegroup = sorted(Imagegroup, key=lambda f: int(os.path.basename(f).split('.')[0]))

        Imagegroup = sorted(glob.glob(os.path.join(TempDir, '??????.MainViewpoint.png')))
        Jsongroup = sorted(glob.glob(os.path.join(TempDir, '??????.MainViewpoint.json')))

        if len(Imagegroup) != len(Jsongroup):
            assert 'Data Fail'
            break

        FileNameIndextemp = FileNameIndex

        print("Start Image Copy")

        for Filepath in Imagegroup:
            TargetPathTemp = os.path.join(ImageTargetDir, os.path.basename(Filepath))
            shutil.copy2(Filepath, TargetPathTemp)
            FileNameIndextemp = FileNameIndextemp + 1
            print(FileNameIndextemp)
        print("End Image Copy")

        FileNameIndextemp = FileNameIndex

        TrainLabelList = []

        ana_id = 1
        image_id = 1

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        for i in range(len(Jsongroup)):


            jsonFilepath = Jsongroup[i]
            json_file = open(jsonFilepath)
            Data = json.load(json_file)

            img = cv2.imread(Imagegroup[i])
            # img = img.copy()

            img_h, img_w, img_c = img.shape

            image_name = os.path.basename(Imagegroup[i])
            image_info = pycococreatortools.create_image_info(image_id, image_name, (img_w, img_h))
            coco_output["images"].append(image_info)

            boundingboxlist = []

            for p in Data['objects']:

                strname = p['class']
                strname = strname.lower()

                ymin = p['bounding_box']["top_left"][0]
                ymin = int(clamp(ymin, 0, img_h))

                xmin = p['bounding_box']["top_left"][1]
                xmin = int(clamp(xmin, 0, img_w))

                ymax = p['bounding_box']["bottom_right"][0]
                ymax = int(clamp(ymax, 0, img_h))

                xmax = p['bounding_box']["bottom_right"][1]
                xmax = int(clamp(xmax, 0, img_w))

                if ymin == ymax or xmin == xmax:
                    continue
                lower = None

                for key in IDTransform:
                    tempvalue = strname.find(key)
                    if tempvalue > 0:
                        lower = CLASSES.index(IDTransform[key])
                        break


                if lower is None:
                    print('error have other object')
                    continue

                #boundingboxlist = removecoverdata(boundingboxlist, ymin, xmin, ymax, xmax, lower)

                boundingboxlist.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'class': lower})


            output = []
            for it in boundingboxlist:
                xmin = it['xmin']
                ymin = it['ymin']
                xmax = it['xmax']
                ymax = it['ymax']
                id = it['class']

                class_id = id

                category_info = {'id': class_id, 'is_crowd': False}

                boxtemp = [xmin, ymin,
                           xmax - xmin,
                           ymax - ymin]

                annotation_info = pycococreatortools.create_annotation_info_withoutmask(ana_id, image_id, category_info,
                                                                                        bounding_box=boxtemp,
                                                                                        image_size=(img_w, img_h),
                                                                                        tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                ana_id = ana_id + 1

            image_id = image_id + 1


                # cv2.circle(img, (xmin, ymin), 2, (0, 255, 0), -1)
                # cv2.circle(img, (xmax, ymax), 2, (0, 255, 0), -1)

            # imagename = os.path.basename(Imagegroup[i])
            # temp = os.path.join(_testDir, imagename)
            # cv2.imwrite(temp, img)

        jsonpath = f'{ROOT_DIR}/ACRV_Virtual.json'
        with open(jsonpath, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

        print("Finish ImageSets")
