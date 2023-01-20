# coding=utf-8
# coding=utf-8
import glob
from collections import defaultdict

import os
from jinja2 import Environment, PackageLoader, FileSystemLoader
import xml.etree.ElementTree as ET
import numpy as np
from xml.etree.ElementTree import parse, Element

from tqdm import tqdm


class Writer(object):
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=FileSystemLoader('../resource'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')
        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': round(float(xmin)),
            'ymin': round(float(ymin)),
            'xmax': round(float(xmax)),
            'ymax': round(float(ymax)),
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def addObjectByBox(self, bbox, name):
        xmin, ymin, xmax, ymax = [round(float(b)) for b in bbox]
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': 'Unspecified',
            'truncated': 0,
            'difficult': 0,
        })

    def addBboxes(self, bboxes, names, difficults= None):
        if not isinstance(names, list):
            names = [names for _ in range(len(bboxes))]
        if difficults is None:
            difficults = [0 for _ in range(len(names))]
        for box, name, difficult in zip(bboxes, names, difficults):
            xmin, ymin, xmax, ymax = map(round, map(float, box))
            self.addObject(name, xmin, ymin, xmax, ymax, difficult = difficult)


    def save(self, annotation_path):
        if not annotation_path.endswith('.xml'):
            annotation_path = annotation_path.split('.')[0] + '.xml'

        with open(annotation_path, 'w', encoding= 'utf-8') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content.encode('utf-8').decode('utf-8'))


class Reader(object):
    def __init__(self, xml_path):
        self.xml_path = xml_path
        # assert os.path.exists(self.xml_path)
        if os.path.exists(self.xml_path):
            tree = parse(xml_path)
            self.root = tree.getroot()
        else:
            self.root = None

        self.name_pattern = os.path.split(xml_path)[-1].split('_')[-1].split(
            '.')[0]

    def text2int(self, text):
        return int(round(float(text)))

    def get_objects(self):
        # objects = self.root.getElementsByTagName("object")
        width = float(self.root.find("size").find('width').text)
        height = float(self.root.find("size").find('height').text)
        path = self.root.find("path").text
        obj_dicts = {'name':[], 'bboxes':[], 'category_name':[], 'difficult':[],
                     'name_pattern': '', 'height':height, 'width':width, 'path':path, 'database': self.root.find(
                "source").find('database').text}

        if self.root is not None:
            for object in self.root.iter('object'):
                difficult = int(object.find('difficult').text)

                box = object.find('bndbox')
                y_min = self.text2int(box.find("ymin").text)
                x_min = self.text2int(box.find("xmin").text)
                y_max = self.text2int(box.find("ymax").text)
                x_max = self.text2int(box.find("xmax").text)
                bbox = [x_min, y_min, x_max, y_max]

                name = object.find('name').text
                obj_dicts['name'].append(name)
                obj_dicts['category_name'].append(name)
                obj_dicts['bboxes'].append(bbox)
                obj_dicts['difficult'].append(difficult)

                obj_dicts['name_pattern'] = self.name_pattern
        return obj_dicts

    def get_objectByName(self, name):
        objects = self.root.getElementsByTagName("object")
        obj_dicts = defaultdict(list)

        for object in objects:
            box = object.find('bndbox')

            y_min = int(box.find("ymin").text)
            x_min = int(box.find("xmin").text)
            y_max = int(box.find("ymax").text)
            x_max = int(box.find("xmax").text)
            bbox = [x_min, y_min, x_max, y_max]

            current_name = object.find('name').text
            if current_name == name:
                obj_dicts['category_name'].append(current_name)
                obj_dicts['bboxes'].append(bbox)
                obj_dicts['name_pattern'].append(self.name_pattern)
        return obj_dicts

class Modify(object):
    def __init__(self, xml_path, target_path):
        self.xml_path = xml_path
        self.target_path = target_path

    def modify(self):


        # tags = ['xmin', 'ymin', 'xmax', 'ymax']
        # for tag in tags:
        #     for x in self.root.iter(tag=tag):
        #         x.text = str(round(float(x.text)))
        #
        # return True
        tags = ['name']
        for tag in tags:
            for x in self.root.iter(tag=tag):
                if x.text.lower().strip() == 'neg':
                    x.text = 'disease'
                    # print(self.xml_path)
        return True

    def init(self):
        self.doc = parse(self.xml_path)
        self.root = self.doc.getroot()

    def rewrite(self):
        self.doc.write(self.target_path, encoding='utf-8')

    def run(self):
        self.init()
        flag = self.modify()
        if flag:
            self.rewrite()
            # print(self.xml_path)





def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    # np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def cal_anchor(path, CLUSTERS = 9):
    dataset = []
    for xml_file in tqdm(glob.glob("{}/*.xml".format(path))):
        tree = ET.parse(xml_file)

        height = float(tree.findtext("./size/height"))
        width = float(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = float(obj.findtext("bndbox/xmin")) / width
            ymin = float(obj.findtext("bndbox/ymin")) / height
            xmax = float(obj.findtext("bndbox/xmax")) / width
            ymax = float(obj.findtext("bndbox/ymax")) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            dataset.append([xmax - xmin, ymax - ymin])

    dataset = np.array(dataset)
    out = kmeans(dataset, k = CLUSTERS)
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(dataset, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))




if __name__ == '__main__':
    # for root, _, filenames in tqdm(os.walk(
    #         '/Users/sober/Downloads/new/참나무시들음병/xml')):
    #     if 'xml' in root and 'xml1' not in root:
    #         for filename in filenames:
    #             if filename.endswith('.xml'):
    #                 s_p = os.path.join(root, filename)
    #                 t_r = root.replace('xml', 'xml1')
    #                 os.makedirs(t_r, exist_ok=True)
    #                 t_p = os.path.join(t_r, filename)
    #                 Modify(s_p, t_p).run()

    # path = '/Users/sober/Downloads/data_split/train_valid/large_xml/vis_00000676.xml'
    # read = Reader(path)
    # print(read.get_objects())

    # cal_anchor('/Users/sober/Downloads/pine/test/xml')


    # for root, _, filenames in tqdm(os.walk(
    #         '/Users/sober/Downloads/pine')):
    #     if 'xml' in root :
    #         for filename in filenames:
    #             if filename.endswith('.xml'):
    #                 s_p = os.path.join(root, filename)
    #
    #                 # info = Reader(s_p).get_objects()
    #                 # for bbox in info['bboxes']:
    #                 #     if bbox[0] == 0 and bbox[1] == 0:
    #                 #         print(root, filename)
    #
    #                 # t_r = root.replace('xml', 'xml1')
    #                 # os.makedirs(t_r, exist_ok=True)
    #                 # t_p = os.path.join(t_r, filename)
    #
    #                 Modify(s_p, s_p).run()


    for root, _, filenames in os.walk(
            '/Users/sober/Downloads/pos+neg/train/large_neg_xml'):
        for filename in tqdm(filenames):
            if filename.endswith('.xml'):
                t_p = os.path.join('/Users/sober/Downloads/pos+neg/train/merge_disneg', filename)
                s_p = os.path.join(root, filename)
                Modify(s_p, t_p).run()

    # Reader('/Volumes/dataset 1/make_data/all pine_rgb dataset/data_split/train_valid/large_xml/vis_00002162.xml').get_objects()

