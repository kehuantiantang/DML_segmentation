# coding=utf-8
import hashlib
import os
import pprint
import random
from collections import OrderedDict, defaultdict

from PIL import Image, ImageOps
from torch.utils.data import Dataset
import os.path as osp

from tqdm import tqdm

from json_polygon import JsonLoader
from utils.util import load_pkl
from utils.sober_logger import SoberLogger

class CropPatchObject(object):
    def __init__(self, filename, category_name, bbox, score):
        self.filename = filename
        self.category_name = category_name
        self.bbox = self.bbox_float2int(bbox)
        self.score = score

    def bbox_float2int(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def get_item(self):
        return self.filename, self.category_name, self.bbox, self.score



class PinePKLDataset(Dataset):

    def __init__(self, image_root, pkl_path, transforms = None, random_enlarge = None, only_return_dataLabels = True,
                 tp_fp_threshold = 0.1, img_extension = 'jpg'):

        self.transforms = transforms
        self.image_root = image_root
        self.pkl_path = pkl_path
        self.tp_fp_threshold = tp_fp_threshold
        self.random_enlarge = random_enlarge
        self.img_extension = img_extension

        self.only_return_dataLabels = only_return_dataLabels
        self.class2index_dict = OrderedDict({'tp': 0, 'gt':0, 'fp_repeat':0, 'fp':1})
        self.target_class_names = list(set([k.split('|')[0] for k in sorted(self.class2index_dict.keys())]))
        SoberLogger.debug('Target_class_names:', self.target_class_names)

        self.classes = list(sorted(set([v for k, v in self.class2index_dict.items()])))
        SoberLogger.debug('Classes index:', self.classes)

        self.all_objects, self.counter = [], defaultdict(lambda : 0)
        self.load_patch_coordinate(pkl_path)

    def __add_item(self, value, key, classname, filename):
        category_name = '%s|%s' % (classname, key)
        if category_name in self.class2index_dict.keys():
            for *a, xmin, ymin, xmax, ymax in value[key]:
                score = a[0] if key != 'gt' else 2.0
                self.all_objects.append(CropPatchObject(filename, category_name, [xmin, ymin, xmax, ymax],
                                                  score))

                self.counter[category_name] += 1



    def load_patch_coordinate(self, pkl_path):
        assert osp.exists(pkl_path), 'Pkl file is not exit ! %s'%pkl_path
        pkl_content = load_pkl(pkl_path)
        for filename in tqdm(sorted(pkl_content), total=len(pkl_content.keys()), desc='Load pkl '
                                                                                      'file'):
            value = pkl_content[filename]
            tps, fps, fp_repeats, gts = value['tp'], value['fp'], value['fp_repeat'], value['gt']
            all_polygons = [*tps, *fps, *fp_repeats]

            for ap in all_polygons:
                score, polygon = ap
                xmin, ymin, xmax, ymax = min(polygon[:, 0]), min(polygon[:, 1]), max(polygon[:, 0]), max(polygon[:, 1])
                if score > self.tp_fp_threshold:
                    category_name = 'tp'
                else:
                    category_name = 'fp'

                self.counter[category_name] += 1
                self.all_objects.append(CropPatchObject(filename, category_name, [xmin, ymin, xmax, ymax], score))

            for g in gts:
                score, polygon = g
                xmin, ymin, xmax, ymax = min(polygon[:, 0]), min(polygon[:, 1]), max(polygon[:, 0]), max(polygon[:, 1])

                self.counter['gt'] += 1
                self.all_objects.append(CropPatchObject(filename, 'tp', [xmin, ymin, xmax, ymax], score))

        pprint.pprint(self.counter, indent = 4)


    def __getitem__(self, index):
        while True:
            try:
                filename, label, (xmin, ymin, xmax, ymax), score = self.all_objects[index].get_item()
                img_path = osp.join(self.image_root, '%s.%s'%(filename, self.img_extension))

                assert osp.exists(img_path), 'Image is not exit ! %s'%img_path
                f = open(img_path, 'rb')
                image = Image.open(f)
                image = ImageOps.exif_transpose(image)
                height, width = image.size

                if self.random_enlarge is not None:
                    assert isinstance(self.random_enlarge, list) and len(self.random_enlarge) == 2
                    enlarge = random.randint(self.random_enlarge[0], self.random_enlarge[1])
                    xmin, ymin, xmax, ymax = max(xmin - enlarge, 0), max(ymin - enlarge, 0), min(xmax + enlarge,
                                                                                                 width), \
                                             min(ymax + enlarge, height)

                patch = image.crop((xmin, ymin, xmax, ymax))
                f.close()


                if self.transforms:
                    patch = self.transforms(patch)

                if self.only_return_dataLabels:
                    return patch, self.class2index_dict[label]
                else:
                    return patch, self.class2index_dict[label], filename, score
            except Exception as e:
                from utils.sober_logger import SoberLogger

                SoberLogger.critical('error', filename, index, e)
                index = random.randint(0, len(self.all_objects) - 1)

    def __len__(self):
        return len(self.all_objects)


class PineSegJsonDataset(Dataset):
    def __init__(self, image_root, json_path, transforms = None, random_enlarge = None, only_return_dataLabels =
    True, tp_fp_threshold = 0.50, img_extension = 'jpg'):
        '''
        Using unet segmentation result '*.json', '*.jpg' to generate dataset
        :param image_root: the image path
        :param json_path: predict unet json path
        :param transforms:
        :param random_enlarge: whether to enlarge the bbox in order to get more background information
        :param only_return_dataLabels: whether to return the filename and score
        :param tp_fp_threshold: the threshold of tp and fp in terms of score (predict polygon overlap with gt IoU value)
        :param img_extension: the image suffix extension
        '''

        self.transforms = transforms
        self.image_root = image_root
        self.json_path = json_path
        self.random_enlarge = random_enlarge
        self.tp_fp_threshold = tp_fp_threshold
        self.only_return_dataLabels = only_return_dataLabels
        self.img_extension = img_extension

        self.class2index_dict = OrderedDict({'tp': 0, 'fp':1})
        self.target_class_names = list(set([k.split('|')[0] for k in sorted(self.class2index_dict.keys())]))
        SoberLogger.debug('Target_class_names:', self.target_class_names)

        self.classes = list(sorted(set([v for k, v in self.class2index_dict.items()])))
        SoberLogger.debug('Classes index:', self.classes)

        self.all_objects, self.counter = [], defaultdict(lambda : 0)
        self.load_patch_coordinate(json_path)



    def load_patch_coordinate(self, json_path):
        assert osp.exists(json_path), 'Json file is not exit ! %s'%json_path
        jl = JsonLoader()
        for root, _, filenames in os.walk(json_path):
            for filename in tqdm(sorted(filenames), total=len(filenames), desc='Load json file'):
                if filename.endswith('.json'):
                    content = jl.load_json(osp.join(root, filename))
                    objs = jl.get_objects(content)
                    for bbox, score in zip(objs['bboxes'], objs['score']):
                        # IoU overlap score with gt smaller than threshold if fp, otherwise tp
                        if score > self.tp_fp_threshold:
                            category_name = 'tp'
                        else:
                            category_name = 'fp'

                        self.all_objects.append(
                            CropPatchObject(filename.split('.')[0], category_name, bbox, score))

                        self.counter[category_name] += 1

        pprint.pprint(self.counter, indent = 4)


    def __getitem__(self, index):
        while True:
            try:
                filename, label, (xmin, ymin, xmax, ymax), score = self.all_objects[index].get_item()
                img_path = osp.join(self.image_root, '%s.%s'%(filename, self.img_extension))

                assert osp.exists(img_path), 'Image is not exit ! %s'%img_path
                f = open(img_path, 'rb')
                image = Image.open(f)
                image = ImageOps.exif_transpose(image)
                height, width = image.size

                if self.random_enlarge is not None:
                    assert isinstance(self.random_enlarge, list) and len(self.random_enlarge) == 2
                    enlarge = random.randint(self.random_enlarge[0], self.random_enlarge[1])
                    xmin, ymin, xmax, ymax = max(xmin - enlarge, 0), max(ymin - enlarge, 0), min(xmax + enlarge,
                                                                                                 width), \
                                             min(ymax + enlarge, height)

                patch = image.crop((xmin, ymin, xmax, ymax))
                f.close()

                if self.transforms:
                    patch = self.transforms(patch)

                if self.only_return_dataLabels:
                    return patch, self.class2index_dict[label]
                else:
                    return patch, self.class2index_dict[label], filename, score
            except Exception as e:
                from utils.sober_logger import SoberLogger

                SoberLogger.critical('error', filename, index, e)
                index = random.randint(0, len(self.all_objects) - 1)

    def __len__(self):
        return len(self.all_objects)


if __name__ == '__main__':
    # dt = PineDTDataset('/dataset/khtt/dataset/pine_total_data_2021_deadTree/img',
    #               '/dataset/khtt/dataset/pine_total_data_2021_deadTree/xml_rename', cache_key = 'train')
    pkl = load_pkl('/dataset/khtt/dataset/pine2022/ECOM/7.evaluations/unet_deploy_small_20230120_150735/tp_fp_gt.pkl')
    for k, value in pkl.items():
        tps, fps, fp_repeats, gts = value['tp'], value['fp'], value['fp_repeat'], value['gt']
        all = [*tps, *fps, *fp_repeats, *gts]
        print(k, value)


