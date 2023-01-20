# coding=utf-8
import json
from types import SimpleNamespace
import os
import os.path as osp
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class JsonLoader(object):
    color_dict = {'red':(0, 0, 255), 'green':(0, 255, 0), 'blue':(255, 0,
                                                                  0),
                  'yellow': (255, 255, 0), 'cyan':(0, 255, 255), 'sliver':(
            192, 192, 192), 'black': (255, 255, 255)}

    def __init__(self, get_object_method = None):
        self.get_object_method = get_object_method

    def get_color(self, c):
        if isinstance(c, str):
            return JsonLoader.color_dict[c]
        else:
            return c

    def load_json(self, path, replace_pair = [('class', 'class_id'), ('Class', 'Class_id')]):
        with open(path, encoding='utf-8') as f:
            context = f.read()
            if replace_pair:
                for old, new in replace_pair:
                    context = context.replace(old, new)

            context = json.loads(context, object_hook=lambda d: SimpleNamespace(**d))
            return context

    def get_objects(self, context):
        if self.get_object_method != None:
            return self.get_object_method(context)
        else:
            width =  int(context.imageWidth)
            height = int(context.imageHeight)
            path = context.imagePath
            obj_dicts = {'name':[], 'bboxes':[], 'category_name':[], 'score':[],
                         'name_pattern': '', 'height':height, 'width':width, 'path':path, 'polygons':[],
                         'filename':path}


            for polygon in context.shapes:
                label = polygon.label
                points = polygon.points
                for i in range(len(points)):
                    points[i][0] = int(points[i][0])
                    points[i][1] = int(points[i][1])

                # points = list(map(int, points))
                group_id = polygon.group_id
                shape_type = polygon.shape_type
                score = float(polygon.score) / 100.0
                flags = polygon.flags

                hs, ws = [], []
                for point in points:
                    h, w = point
                    hs.append(h)
                    ws.append(w)

                if len(hs) > 0 or len(ws) > 0:

                    xmin, ymin, xmax, ymax = min(hs), min(ws), max(hs), max(ws)

                    if shape_type == 'polygon':
                        obj_dicts['name'].append('disease')
                        obj_dicts['bboxes'].append([xmin, ymin, xmax, ymax])
                        obj_dicts['category_name'].append('disease')
                        obj_dicts['score'].append(score)
                        obj_dicts['polygons'].append(points)


            return obj_dicts
    #

    def draw_box(self, img, bb, name, c):
        color = self.get_color(c)
        bb = [int(float(b)) for b in bb]
        img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)

        text_pos = [bb[0], bb[1] - 10]
        # 'xmin', 'ymin', 'xmax', 'ymax'
        if text_pos[1] < 0:
            text_pos = [bb[2], bb[3] - 10]
        img = cv2.putText(img, str(name), tuple(text_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, JsonLoader.color_dict["green"], 1, cv2.LINE_AA)
        return img

    def draw_bboxes(self, img, obj_dicts):
        assert np.max(img) > 1.0
        img = np.array(img)
        for box, name in zip(obj_dicts['bboxes'], obj_dicts['name']):
            img = self.draw_box(img, box, name, 'blue')
        return img


    def draw_polygon(self, img, polygon, name, c):
        color = self.get_color(c)
        img = cv2.polylines(img, [np.array(polygon, np.int32)], True, color, 1)
        img = cv2.putText(img, str(name), (int(polygon[0][0]), int(polygon[0][1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, JsonLoader.color_dict["green"], 1, cv2.LINE_AA)
        return img

    # def draw_polygons(self, img, obj_dicts):
    #     assert np.max(img) > 1.0
    #     img = np.array(img)
    #     for polygon, name in zip(obj_dicts['polygons'], obj_dicts['name']):
    #         img = self.draw_polygon(img, polygon, name, 'blue')
    #     return img

    def draw_polygons(self, img, obj_dicts):
        assert np.max(img) > 1.0
        img = np.array(img)
        for polygon, name in zip(obj_dicts['polygons'], obj_dicts['name']):
            img = self.draw_polygon(img, polygon, name, 'blue')
        return img

    def draw_mask(self, img, obj_dicts, c = 'black', single_channel = False):
        '''
        draw semantic segmentation mask
        :param img:  raw gis image
        :param obj_dicts:  polygon point [n, m, 2], n object, m point, (x, y)
        :param c:  color (0, 0, 0),  (1, 1, 1)
        :param single_channel: if true return segmentation mask [0, 1, 0, 0, 2, 1],
        else return rgb visualization mask
        :return:
        '''
        mask = np.zeros_like(img)
        polygons = [np.array(polygon) for polygon in obj_dicts['polygons']]
        color = self.get_color(c) # (r, g, b)
        mask = cv2.fillPoly(mask, polygons, color)
        # mask = cv2.fillPoly(mask, polygons, 1)
        # return mask

        if single_channel:
            mask = mask[:, :, 0]
            return mask
        else:
            return mask


    def object_counter_bbox_polygon(self, path):
        counter, bboxes, polygons = defaultdict(int), defaultdict(list), defaultdict(list)

        for root, _, filenames in os.walk(path):
            for filename in tqdm(filenames):
                if filename.endswith('json'):
                    p = osp.join(root, filename)

                    context = JsonLoader.load_json(p)
                    obj_dicts = JsonLoader.get_objects(context)

                    for cls, bbox, polygon in zip(obj_dicts['name'], obj_dicts['bboxes'], obj_dicts['polygons']):
                        counter[cls] += 1
                        bboxes[cls].append(bboxes)
                        polygons[cls].append(polygon)

        return counter, bboxes, polygons

