import argparse
from ast import arg
from distutils.command.config import config
import time
import datetime
import os
import shutil
import sys
from webbrowser import get
import numpy as np
from tqdm import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from models.model_zoo import get_segmentation_model

from utils.distributed import *
from losses.loss import SegCrossEntropyLoss, CriterionKD
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.datasets import CSTrainValSet
from utils.flops import cal_multi_adds, cal_param_size
from PIL import Image
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument('--pixel-memory-size', type=int, default=20000)
    parser.add_argument('--region-memory-size', type=int, default=2000)
    parser.add_argument('--region-contrast-size', type=int, default=1024)
    parser.add_argument('--pixel-contrast-size', type=int, default=4096)

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--lambda-kd", type=float, default=1., help="lambda_kd")
    parser.add_argument("--ohem_thresh", type=float, default=0.7, help="lambda memory-based region")
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--warmup_iter', type=int, default=8000,
                        help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--EDD_method', type=str, default='XOR',
                        help='the method EDD use')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    path_now = os.path.dirname(os.path.abspath(__file__))
    args.data = os.path.join(path_now, args.data)
    args.save_dir = os.path.join(path_now, args.save_dir)
    args.log_dir = os.path.join(path_now, args.log_dir)
    args.teacher_pretrained = os.path.join(path_now, args.teacher_pretrained)
    if args.teacher_pretrained_base != 'None':
        args.student_pretrained_base = os.path.join(path_now, args.student_pretrained_base)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')
    
    if num_gpus > 1 and args.local_rank == 0:
        argsDict = args.__dict__
        with open(os.path.join(os.path.expanduser(args.save_dir), 'config.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


    return args


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, args=None, reduction='mean', **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        logger.info(f'OHEM thresh is {thresh}')

        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.reduction = reduction
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                                       reduction=self.reduction)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=self.reduction)
        self.cityscapepallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        self.args = args

    def OHEM_Mask_Visulize(self, masks, logits, names, label):
        # mask: [B, H , W]
        prediction = F.softmax(logits, dim=1)
        prediction = torch.argmax(prediction, dim=1).cpu().numpy()

        for i in range(len(names)):
            pred_img = Image.fromarray(prediction[i].astype('uint8'))
            pred_img.putpalette(self.cityscapepallete)

            masks_idx = masks[i].cpu().numpy().astype('uint8')
            mask = Image.fromarray(masks_idx)

            plt.figure(figsize=(20, 10))

            plt.subplot(1, 2, 1)
            plt.imshow(pred_img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.axis('off')
            plt.colorbar()

            outname = os.path.splitext(os.path.split(names[i])[-1])[0] + f'_OHEM_{label}'
            plt.savefig(os.path.join('vis', f'{outname}_pred.png'), bbox_inches='tight')
            plt.close()

    def get_mask(self, pred, target):
        n, c, h, w = pred.size()
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)  # prob: [B, 19, 512, 1024]
        prob = prob.transpose(0, 1).reshape(c, -1)  # prob: [19, B* 512*1024]
        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]

            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
        return valid_mask, kept_mask

    def visulize_mask(self, mask):
        mask = mask.cpu().numpy().astype('uint8')
        mask = Image.fromarray(mask)
        plt.imshow(mask)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        plt.savefig('fig.png')

    def forward_with_teacher_student(self, pred_student, pred_teacher, target):
        B, H, W = target.size()
        pred_teacher = F.interpolate(pred_teacher, (H, W), mode='bilinear', align_corners=True)
        pred_student = F.interpolate(pred_student, (H, W), mode='bilinear', align_corners=True)

        n, c, h, w = pred_teacher.size()
        target = target.view(-1)

        valid_mask_student, kept_mask_student = self.get_mask(pred_student.clone(), target.clone())
        valid_mask_teacher, kept_mask_teacher = self.get_mask(pred_teacher.clone(), target.clone())

        # visulize_mask(valid_mask_student.view(n, h, w)[0])
        # visulize_mask(valid_mask_teacher.view(n, h, w)[0])
        # visulize_mask(kept_mask_student.view(n, h, w)[0])
        # visulize_mask(kept_mask_teacher.view(n, h, w)[0])

        # mask_union = torch.logical_or(mask_student, mask_teacher)
        
        # import pdb
        # pdb.set_trace()
        
        if self.args.EDD_method == 'XOR':
            keep_mask_union = torch.logical_or(kept_mask_student, kept_mask_teacher)
            mask_diff = valid_mask_student ^ valid_mask_teacher

        elif self.args.EDD_method == 'OR':
            keep_mask_union = torch.logical_or(kept_mask_student, kept_mask_teacher)
            mask_diff = valid_mask_student | valid_mask_teacher

        elif self.args.EDD_method == 'AND':
            keep_mask_union = torch.logical_or(kept_mask_student, kept_mask_teacher)
            mask_diff = valid_mask_student & valid_mask_teacher

        elif self.args.EDD_method == 'OHEM':
            keep_mask_union = torch.logical_or(kept_mask_student, kept_mask_teacher)
            mask_diff = valid_mask_student 
        

        target = target * keep_mask_union.long()
        target = target.masked_fill_(~mask_diff, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred_student, target), mask_diff.view(n, h, w)

    def forward_with_teacher(self, pred_teacher, pred_student, target):
        B, H, W = target.size()
        pred_teacher = F.interpolate(pred_teacher, (H, W), mode='bilinear', align_corners=True)

        n, c, h, w = pred_teacher.size()
        target = target.view(-1)

        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        # softmax function to transfer the logits
        prob = F.softmax(pred_teacher, dim=1)  # prob: [B, 19, 512, 1024]
        prob = prob.transpose(0, 1).reshape(c, -1)  # prob: [19, B* 512*1024]
        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]

            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        pred_student = F.interpolate(pred_student, (H, W), mode='bilinear', align_corners=True)
        return self.criterion(pred_student, target)

    def forward(self, pred, target, mask_vis=False, names=None, label=None):
        B, H, W = target.size()
        pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)

        n, c, h, w = pred.size()
        target = target.view(-1)

        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        # softmax function to transfer the logits
        prob = F.softmax(pred, dim=1)  # prob: [B, 19, 512, 1024]
        prob = prob.transpose(0, 1).reshape(c, -1)  # prob: [19, B* 512*1024]
        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]

            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        if mask_vis is True:
            self.OHEM_Mask_Visulize(valid_mask.view(n, h, w), pred, names, label)

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return self.criterion(pred, target)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        train_dataset = CSTrainValSet(args.data,
                                      list_path='./dataset/list/cityscapes/train.lst',
                                      max_iters=args.max_iterations * args.batch_size,
                                      crop_size=args.crop_size, scale=True, mirror=True)
        val_dataset = CSTrainValSet(args.data,
                                    list_path='./dataset/list/cityscapes/val.lst',
                                    crop_size=(1024, 2048), scale=False, mirror=False)

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=False, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model,
                                              backbone=args.teacher_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base='None',
                                              pretrained=args.teacher_pretrained,
                                              aux=True,
                                              norm_layer=nn.BatchNorm2d,
                                              num_class=train_dataset.num_class).to(self.args.local_rank)

        self.s_model = get_segmentation_model(model=args.student_model,
                                              backbone=args.student_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base=args.student_pretrained_base,
                                              pretrained='None',
                                              aux=args.aux,
                                              norm_layer=BatchNorm2d,
                                              num_class=train_dataset.num_class).to(self.device)

        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.CEcriterion = SegCrossEntropyLoss(ignore_index=args.ignore_label, reduction='none').to(self.device)
        self.OHEMcriterion = OhemCrossEntropy2d(ignore_index=args.ignore_label, thresh=args.ohem_thresh, args=args).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)

        params_list = nn.ModuleList([])
        params_list.append(self.s_model)

        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model,
                                                               device_ids=[args.local_rank],
                                                               output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr * ((1 - float(iter) / max_iter) ** (power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))
        kl_distance = nn.KLDivLoss(reduction='none')
        sm = torch.nn.Softmax(dim=1)
        log_sm = torch.nn.LogSoftmax(dim=1)
        scaler = torch.cuda.amp.GradScaler()

        for load_iter in range(1, 26):
            dict_save = {}
            VGD_list = [0] * 20
            EDD_list = [0] * 20
            count = 0

            weight_path = f'/home/muyun99/github/CVPR/distillation/DCS/work_dirs/table7' \
                          f'/deeplabv3_res18_onlykd_TwoStage_diff_warmup_lamda_bs16_iteration_40000_warmupiter_0_EDD_method_XOR_repeat_1/kd_deeplabv3_resnet18_citys_{load_iter * 1600}.pth'
            save_path = f'{args.save_dir}/iteration_{load_iter * 1600}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.s_model.load_state_dict(torch.load(weight_path))

            self.s_model.eval()
            for iteration, (images, targets, name_images) in tqdm(enumerate(self.train_loader)):
                iteration = iteration + 1
                if iteration == 2:
                    break

                images = images.to(self.device)
                targets = targets.long().to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        t_outputs = self.t_model(images)
                        s_outputs = self.s_model(images)
                    # generate VGD Map
                    variance = torch.sum(kl_distance(log_sm(t_outputs[0]), sm(t_outputs[1])), dim=1)
                    exp_variance = torch.exp(-variance)
                    exp_variance_scale = torch.unsqueeze(exp_variance, 1)
                    exp_variance_scale = F.interpolate(exp_variance_scale, (512, 1024), mode='bilinear',
                                                        align_corners=True)
                    # [16,512,1024]
                    mask_VGD = exp_variance_scale.squeeze()
                    # [16,512,1024]
                    task_loss_EDD, mask_EDD = self.OHEMcriterion.forward_with_teacher_student(s_outputs[0], t_outputs[0], targets)
                    count += len(name_images)

                    for cls in range(20):
                        VGD_list[cls] += mask_VGD[targets == cls].sum() / (targets == cls).sum()
                        EDD_list[cls] += mask_EDD.int()[targets == cls].sum() / (targets == cls).sum()

            # VGD_array = np.array(VGD_list) / count
            # EDD_array = np.array(EDD_list) / count

            # VGD_array = np.array(VGD_list)
            # EDD_array = np.array(EDD_list)

            print(f'VGD: {VGD_list}')
            print(f'EDD: {EDD_list}')
                    # for img_id in range(images.size(0)):
                    #     dict_save[name_images[img_id]] = [
                    #         mask_EDD[img_id].cpu().numpy(),
                    #         mask_VGD[img_id].cpu().numpy(),
                    #         targets[img_id].cpu().numpy()
                    #     ]
                    #     np.save(f'{save_path}/{name_images[img_id]}_mask_VGD.npy', mask_VGD[img_id].cpu().numpy())
                    #     np.save(f'{save_path}/{name_images[img_id]}_mask_EDD.npy', mask_EDD[img_id].cpu().numpy())
                    #     np.save(f'{save_path}/{name_images[img_id]}_mask_label.npy', targets[img_id].cpu().numpy())
                    #
                    # debug = 1

            # np.save(f'{save_path}/dict_save.npy', dict_save)


    def validation(self, iteration):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                pixAcc.item() * 100, mIoU * 100))

            writer.add_scalar('Test/pixAcc', pixAcc.item() * 100, iteration)
            writer.add_scalar('Test/mIoU', mIoU * 100, iteration)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        writer.add_scalar('Test/mIoU_Best', self.best_pred * 100, iteration)

        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best)
        
        synchronize()


def save_checkpoint(model, args, is_best=False, iteration=None):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if iteration is None:
        filename = 'kd_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset)
    else:
        filename = 'kd_{}_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset, iteration)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if (args.distributed and args.local_rank == 0):
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
    writer = SummaryWriter(args.save_dir)

    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
