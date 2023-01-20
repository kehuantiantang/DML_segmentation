# coding=utf-8
import json
import os
import sys
import time
import zipfile
import os.path as osp
from pathlib import Path
import hdfdict as h5d
import h5py
import yaml

def beauty_argparse(args):
    import prettytable as pt
    if not isinstance(args, dict):
        args = vars(args)

    tb = pt.PrettyTable()
    for key in sorted(args.keys()):
        tb.field_names = ["Name", "Params"]
        tb.add_row([key, args[key]])
    tb.align = 'l'
    print(tb)
    print()
    return tb.get_string()

def save_hyperparams(path, context):
    with open(osp.join(path, 'hyper_params.txt'), 'a+') as file:
        file.write("%s\n"%time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime()))
        file.write(context)
        file.write('%s%s'%('='*80, '\n'))

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best_str = None, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    import torch
    from utils.sober_logger import SoberLogger

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    SoberLogger.debug('Save to %s'%filepath)
    if is_best_str is not None:
        import shutil
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_%s.pth.tar' % is_best_str))
        SoberLogger.info('Save best to %s' % os.path.join(checkpoint, 'model_%s.pth.tar') % is_best_str)


def load_h5(path, key):
    array = None
    try:
        res = h5d.load(path)
        array = res[key]
    except Exception as e:
        print('\nERROR:', path, e)
        sys.exit(-1)
    if array is None:
        print('\nERROR:', path)
        sys.exit(-1)
    return array

def save_h5(fname, data):
    def _check_hdf_file(hdf):
        """Returns h5py File if hdf is string (needs to be a path)."""
        if isinstance(hdf, str):
            hdf = h5py.File(hdf, 'x')
        return hdf

    def dump(d, hdf, compression = 'lzf'):
        """Adds keys of given dict as groups and values as datasets
        to the given hdf-file (by string or object) or group object.

        Parameters
        ----------
        d : dict
            The dictionary containing only string keys and
            data values or dicts again.
        hdf : string (path to file) or `h5py.File()` or `h5py.Group()`

        Returns
        -------
        hdf : obj
            `h5py.Group()` or `h5py.File()` instance
        """

        hdf = _check_hdf_file(hdf)

        def _recurse(d, h):
            for k, v in d.items():
                if isinstance(v, dict):
                    g = h.create_group(k)
                    _recurse(v, g)
                else:
                    h.create_dataset(name=k, data=v, compression=compression)

        _recurse(d, hdf)
        return hdf

    dump(data, fname)


def load_json(fname):
    with open(fname, "r", encoding= 'utf8') as json_file:
        d = json.load(json_file)
        return d


def save_json(fname, data):
    with open(fname, "w", encoding = 'utf8') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True, ensure_ascii=False)


import pickle
def save_pkl(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding='utf8')

def load_weights(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        print('Successfully loaded model', '!'*100)
    except:
        model_dict = model.state_dict()
        # 从与训练的网络中读取相应的权重，如果在现在的网络中拥有，就保留，否则忽略
        prediction_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                prediction_dict[k] = v
            else:
                print('Layer %s not load, Pretrain %s: Define %s'%(k, v.shape, model_dict[k].shape))
        model_dict.update(prediction_dict)
        model.load_state_dict(model_dict)

        print('Load part weight loaded model', '!'*100)
    return model




def hyperparams2yaml(path, context, backup_file = True):

    params = vars(context)
    os.makedirs(osp.join(path, 'yaml'), exist_ok=True)
    f_time = get_format_time()
    with open(osp.join(path, 'yaml', '%s.yaml'%f_time), 'w',
              encoding="UTF-8") as f:
        yaml.dump(params, f, sort_keys=False, allow_unicode = True, indent =4)

    if backup_file:
        backup(osp.join(path, 'yaml', '%s.zip'%f_time))
    from utils.sober_logger import SoberLogger
    SoberLogger.info('Backup:', osp.join(path, 'yaml', '%s'%f_time), '.'*50)
    return osp.join(path, 'yaml', '%s'%f_time)

def backup(path, base_path = None, suffixs = ['py', 'yaml'], exclude_folder_names = ['output', '.idea']):
    base_path = osp.join(osp.dirname(osp.abspath(__file__)), '../') if base_path is None else base_path

    zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
    for root, _, filenames in os.walk(base_path):
        r = osp.abspath(root)
        if sum([e in r for e in exclude_folder_names]) == 0:
            for filename in filenames:
                if filename.split('.')[-1] in suffixs:
                    zipf.write(os.path.join(root, filename),
                               os.path.relpath(os.path.join(root, filename),
                                               os.path.join(base_path, '..')))
    zipf.close()


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_format_time(timezone_str = 'Asia/Seoul', time_format = '%Y%m%d-%H%M%S'):
    from pytz import timezone
    from datetime import datetime
    KST = timezone(timezone_str)
    return datetime.now(tz = KST).strftime(time_format)

if __name__ == '__main__':
    # import numpy as np
    # save_h5(os.path.join('/media/HDD3/khtt/weight/dml/test/resnet.resnet34_plant_village_0.01_eb1000',
    #                      'embedding_123.h5'), {'a':np.random.randint(0, 100, (100))})
    # 最后real test 数据/media/HDD1/khtt/strawberry_dml_final_val/test_data1/
    # count_bbox('/dataset/khtt/dataset/strawberry/real_test_annotation')
    print(get_project_root())