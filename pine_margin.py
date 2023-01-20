# coding=utf-8
import os
from utils.gpu_utils import auto_select_gpu
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu(10000)
import sys
sys.path.append(".")
sys.path.append("..")

import traceback
import warnings
warnings.filterwarnings("ignore")

import argparse
import importlib
import os.path as osp
import random
from collections import OrderedDict
import umap
from efficientnet_pytorch import EfficientNet
from pytorch_metric_learning import losses, miners, distances
from pytorch_metric_learning.utils import common_functions
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from libs.accuracy_calculator import MyAccuracyCalculator
from models.base import MLP
from predict import knn_test
from utils.util import beauty_argparse, AverageMeter, save_checkpoint, save_hyperparams, load_weights,  get_format_time, hyperparams2yaml
from visualization import faster_tsne
from pine_pkl_dataset import PinePKLDataset, PineSegJsonDataset
# from pine_pkl_dataset import PinePKLDataset as PineSegJsonDataset
from utils.sober_logger import SoberLogger



class Main(object):
    def __init__(self, args, hyperparams, data_classes):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_classes = data_classes


        self.checkpoint = self.args.checkpoint
        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)

        if not self.args.evaluate:
            save_hyperparams(self.checkpoint, hyperparams)
            SoberLogger.debug('Set distance and mining loss: ', self.checkpoint)

            ### pytorch-metric-learning stuff ###
            distance = distances.CosineSimilarity()


            self.loss_func = losses.TripletMarginLoss(margin=self.args.margin, distance=distance)
            self.mining_func = miners.TripletMarginMiner(margin=self.args.margin, distance=distance,
                                                         type_of_triplets="all") #semihard

        self.accuracy_calculator = MyAccuracyCalculator(
            include=('classification_report', 'NMI', 'AMI', 'precision_at_k', 'confusion_matrix'), k=self.args.k)

    def train(self, model, train_loader, optimizers, epoch):
        model.train()
        avg_losses = AverageMeter()

        pbar = tqdm(train_loader)
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            for _, optimizer in optimizers.items():
                optimizer.zero_grad()
            embeddings = model(data)
            indices_tuple = self.mining_func(embeddings, labels)
            loss = self.loss_func(embeddings, labels, indices_tuple)
            # loss = self.loss_func(embeddings, labels)
            loss.backward()

            for _, optimizer in optimizers.items():
                optimizer.step()


            # write record
            avg_losses.update(loss.data.cpu().numpy(), data.size(0))

            pbar.set_description_str("Loss:%.8f" % (avg_losses.avg))
            self.writer.add_scalar('train/loss_iter', loss, epoch * len(
                train_loader) // self.args.train_batch + batch_idx)

            # if batch_idx == 5:
            #     break

        SoberLogger.info("Average loss for epoch %s is %.8f" % (epoch, avg_losses.avg))
        self.writer.add_scalar('train/loss_epoch', avg_losses.avg, epoch)

    ### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
    def test(self, train_set, test_set, model, epoch):
        accuracies, features = knn_test(train_set, test_set, model, self.accuracy_calculator,
                                        batch_size=self.args.test_batch)
        # print()
        SoberLogger.debug(accuracies)
        for k, v in accuracies.items():
            print(k)
            print(v)
        test_features, test_label = features['test']
        test_features, test_label = test_features.cpu().numpy(), test_label.cpu().numpy().reshape((-1, ))
        umap_embeddings =  umap.UMAP().fit_transform(test_features)
        fig = faster_tsne(umap_embeddings, test_label, self.class_names)
        self.writer.add_figure('test/visual', fig, epoch)
        plt.close()

        return accuracies["precision_at_k"], (accuracies["confusion_matrix"][0][0],accuracies["confusion_matrix"][1][0])

    def prepare_data(self):

        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])


        train_transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.RandomRotation((0, 90)), transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomVerticalFlip(p=0.5), transforms.CenterCrop((224, 224)),
             transforms.ToTensor(),
             normalize])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])


        train_dataset = PineSegJsonDataset(self.args.train_image_root, self.args.train_json_path, transforms=train_transform,
                                       random_enlarge = self.args.random_enlarge,
                                          tp_fp_threshold=self.args.tp_fp_threshold, img_extension=self.args.img_extension)
        SoberLogger.debug('train_dataset', train_dataset.counter)

        if self.args.enlarge_inVal:
            val_random_enlarge = self.args.random_enlarge
        else:
            val_random_enlarge = None

        train_eval_dataset = PineSegJsonDataset(self.args.train_image_root, self.args.train_json_path,
                                            transforms=test_transform, random_enlarge = val_random_enlarge,
                                          tp_fp_threshold=self.args.tp_fp_threshold, img_extension=self.args.img_extension)
        SoberLogger.debug('train_eval_dataset:', train_eval_dataset.counter,
                                          self.args.tp_fp_threshold)


        test_dataset = PineSegJsonDataset(self.args.val_image_root, self.args.val_json_path,
                                          transforms=test_transform, random_enlarge= val_random_enlarge,
                                          tp_fp_threshold=self.args.tp_fp_threshold, img_extension=self.args.img_extension)
        SoberLogger.debug('test_dataset:', test_dataset.counter,
                                          self.args.tp_fp_threshold)

        predict_dataset = PineSegJsonDataset(self.args.val_image_root, self.args.val_json_path,
                                         transforms=test_transform, random_enlarge = val_random_enlarge,
                                          tp_fp_threshold=self.args.tp_fp_threshold, img_extension=self.args.img_extension)
        SoberLogger.debug('predict_dataset:', predict_dataset.counter,
                                          self.args.tp_fp_threshold)


        train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch,
                                  num_workers=self.args.workers, pin_memory=True)

        self.class_names = train_dataset.classes
        return train_loader, None, train_eval_dataset, test_dataset, predict_dataset


    def build_model(self):
        # Model,构建你想要使用的网络
        SoberLogger.debug("==> creating model '{}'".format(self.args.model_name))

        if self.args.model_name.startswith('efficientnet'):
            model = EfficientNet.from_pretrained(self.args.model_name, num_classes=args.nb_classes,
                                                 in_channels=args.in_channels)
        else:
            filename, model_name = self.args.model_name.split('.')
            module = getattr(importlib.import_module('models.' + filename), model_name)
            model = module(pretrained=True, progress=True, start_nb_channels=args.in_channels,
                           num_classes=args.feature_dim)


        embedder_layer_size = [model.fc.in_features] + self.args.embedder_layers
        embedder = MLP(embedder_layer_size, final_relu=True).to(self.device)
        model.fc = common_functions.Identity()

        return {'trunk':model, 'embedder':embedder}

    def build_optimizer(self, models):
        trunk_optimizer = optim.Adam(models['trunk'].parameters(), lr=self.args.lr[0])
        embedder_optimizer = optim.Adam(models['embedder'].parameters(), lr=self.args.lr[1])
        optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}

        trunk_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trunk_optimizer, self.args.epochs)
        embedder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(embedder_optimizer, self.args.epochs)
        schedulers = {"trunk_scheduler": trunk_scheduler, "embedder_scheduler": embedder_scheduler}

        return optimizers, schedulers


    def run(self):
        model = self.build_model()

        optimizers, schedulers = self.build_optimizer(model)
        model = nn.Sequential(model['trunk'], model['embedder']).to(self.device)

        train_loader, _, train_dataset, test_dataset, predict_dataset = self.prepare_data()


        best_accuracy, best_tp, best_fn = -1.0, -1, sys.maxsize

        if self.args.resume:
            # Load checkpoint
            SoberLogger.info('==> Resuming from checkpoint %s' % args.resume)
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            self.writer = SummaryWriter(os.path.join(self.checkpoint, 'tf'))
            cp = torch.load(args.resume)


            model = load_weights(model, state_dict=cp['state_dict'])

            best_accuracy = cp['best_accuracy'] if 'best_accuracy' in cp.keys() else -1
            self.args.start_epoch = cp['epoch']

            SoberLogger.info('Current best accuracy: %.5f'%best_accuracy)

            if self.args.evaluate:
                SoberLogger.debug('Evaluating the model %s'%self.args.resume)
                # test_dataset
                accuracies, embedding = knn_test(train_dataset, predict_dataset, model, self.accuracy_calculator,
                                                 batch_size=self.args.test_batch)
                for k, v in accuracies.items():
                    print(k)
                    print(v)
                results = self.accuracy_calculator.get_results()
                embedding['knn_labels'], embedding['knn_distances'], embedding['query_labels'], embedding[
                    'reference_labels'] = results['knn_labels'].cpu().numpy(), results['knn_distances'].cpu().numpy(), \
                                          results['query_labels'].cpu().numpy(), results[
                                              'reference_labels'].cpu().numpy()
                return


        else:
            # 日志文件地址
            cp = self.checkpoint
            self.writer = SummaryWriter(os.path.join(cp, 'tf'))


        for epoch in range(args.start_epoch, args.epochs + 1):
            self.train(model, train_loader, optimizers, epoch)


            save_dict = {'best_accuracy': best_accuracy, 'state_dict': model.state_dict(), 'epoch': epoch}

            if epoch % self.args.eval_interval == 0:
                curr_accuracy, (curr_tp, curr_fn) = self.test(train_dataset, test_dataset, model, epoch)

                if curr_accuracy > best_accuracy:
                    best_accuracy = curr_accuracy
                    save_dict['best_accuracy'] = best_accuracy
                    save_checkpoint(save_dict, is_best_str='acc_best', checkpoint=self.checkpoint,
                                    filename='latest.pth.tar')

                if curr_tp > best_tp:
                    save_checkpoint(save_dict, is_best_str='tp_best', checkpoint=self.checkpoint,
                                    filename='latest.pth.tar')

                if curr_fn < best_fn:
                    save_checkpoint(save_dict, is_best_str='fn_best', checkpoint=self.checkpoint,
                                    filename='latest.pth.tar')


            if epoch % self.args.save_interval == 0:
                save_checkpoint(save_dict, is_best_str=False, filename='epoch_%03d.pth.tar' % epoch,
                                checkpoint=self.checkpoint)

            for _, scheduler in schedulers.items():
                scheduler.step()


if __name__ == '__main__':
    data_classes = OrderedDict({'tp': 0, 'fp':1})
    parser = argparse.ArgumentParser(description='PyTorch ForeBack Training')

    # epochs
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on '
                                                                                'restarts)')

    # model
    parser.add_argument('--model_name', type=str, default='resnet.resnet18')

    # dataset
    parser.add_argument('--train_batch', default=64*2, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--val_batch', default=64*3, type=int, metavar='N', help='validate batchsize')
    parser.add_argument('--test_batch', default=64*3, type=int, metavar='N', help='test batchsize')



    parser.add_argument('--img_extension', default='tif', type=str, help='image extension')



    base = '/dataset/khtt/dataset/pine2022/elcom/2.labled'
    train_img = osp.join(base, 'CONTOUR_V3_20221122_145813_R12345_25000_Tag_70+7000')
    test_img = osp.join(base, 'CONTOUR_V3_20221122_145813_R12345_25000_Tag_seg_30')


    parser.add_argument('--train_image_root', default=train_img, type=str)
    parser.add_argument('--val_image_root', default=test_img, type=str)

    base = '/dataset/khtt/dataset/pine2022/elcom/7.evaluations'
    train_json = osp.join(base, 'unet_deploy_CONTOUR_V3_20221122_145813_R12345_25000_Tag_70+7000_20230120_152443')
    test_json = osp.join(base, 'unet_deploy_CONTOUR_V3_20221122_145813_R12345_25000_Tag_seg_30_20230120_152454')
    parser.add_argument('--train_json_path', default=train_json, type=str)
    parser.add_argument('--val_json_path', default=test_json, type=str)


    parser.add_argument('--tp_fp_threshold', default=0.5, type=float, help='threshold for filter tp and fp')


    # hyper-parameter
    parser.add_argument('--lr', '--learning_rate', default=[0.00001, 0.0001], nargs = '+', metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--k', default=1, type=int, metavar='M', help='momentum')


    parser.add_argument('-c', '--checkpoint', default="./outputs/test", type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default=None, help='path to latest checkpoint (default: none)')



    parser.add_argument('--pretrain', default=True, action='store_true', help='Using pretrain model or not')
    parser.add_argument('--nb_classes', default=2, type=int, metavar='W', help='number of '
                                                                               'classes')
    parser.add_argument('--feature_dim', default=256, type=int, metavar='W', help='number of classes')
    parser.add_argument('--in_channels', default=3, type=int, metavar='W', help='number of input channel')
    parser.add_argument('--embedder_layers', default=[256], nargs = '+', help='number of embedder-layers')
    parser.add_argument('--random_enlarge', default=[0, 0], nargs = '+', help='increase the bbox size to obtain more '
                                                                              'background information')

    parser.add_argument('--enlarge_inVal', action='store_false')



    parser.add_argument('--margin', type=float, default=0.001,help = 'Triple loss margin')
    parser.add_argument('--comment', type=str, default='_tp')

    parser.add_argument('--save_interval', default=100, type=int, metavar='W', help='number of save interval')
    parser.add_argument('--eval_interval', default=10, type=int, metavar='W', help='number of evaluation interval')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on validation '
                                                                       'set')
    parser.add_argument('--predict_tsne', action='store_true', help='predict model on validation '
                                                                    'set', default=False)


    parser.add_argument('--seed', type=int, help='manual seed', default=0)
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('-logfile_level', default = 'debug')
    parser.add_argument('-stdout_level', default = 'debug')

    args = parser.parse_args()
    args.checkpoint = os.path.join(args.checkpoint, args.model_name + args.comment)

    SoberLogger.init(logfile_level=args.logfile_level, stdout_level=args.stdout_level,
                     log_file=osp.join(args.checkpoint, 'log', '%s.log'%get_format_time()), rewrite=False)

    args.predict_classes_dict = data_classes

    hyperparams = beauty_argparse(args)
    SoberLogger.debug(hyperparams)

    hyperparams2yaml(args.checkpoint, args)

    # 让网络实验可以重复
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        Main(args, hyperparams, data_classes).run()
    except Exception:
        SoberLogger.critical(str(traceback.format_exc()))
