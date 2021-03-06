import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import random
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import modified_vgg
import importlib

import skimage.measure
from scipy import ndimage
importlib.reload(modified_vgg)


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, cwid, config, data_dir):
        self.cwid = cwid
        self.config = config
        self.data_dir = data_dir
        self.cw_num_instances = [len(c) for c in cwid]
        self.class_ids = range(config.NUM_CLASSES)
        # class_wise_iterators
        self.sampler = self.weighted_sampler()
        # self.cid = [item for sublist in cwid for item in sublist]
        # random.shuffle(self.cid)
    # regardless of instance_index, we give some shit.
    # shouldn't matter anyway because of blah blah

    def __getitem__(self, instance_index):
        class_id = next(self.sampler)
        while True:
            # !! no guarantees that we iterate all elements,
            # TODO: modify this iterate one by one? or shift to weighted random sampler? !!
            cwid = (instance_index + 1) % self.cw_num_instances[class_id]
            image, masks, is_crowd = self.load_image_gt(class_id, cwid)
            loaded_im, loaded_masks = image, masks
            image, impulse, gt_response, one_hot, is_bad_image = self.generate_targets(image, masks, class_id, is_crowd)
            if not is_bad_image:
                image = image / 256
                image -= self.config.MEAN_PIXEL
                image /= self.config.STD_PIXEL
                # channels first
                image = np.moveaxis(image, 2, 0)
                # impulse = np.moveaxis(np.expand_dims(impulse, -1), 2, 0)
                i1, j1, i2, j2 = self.extract_bbox(gt_response)
                # di = (i2 - i1) // 2; dj = (j2 - j1) // 2
                # ci = (i1 + i2) // 2; cj = (j1 + j2) // 2
                bbox = np.zeros_like(gt_response)
                bbox[i1:i2, j1:j2] = 1
                # bbox[max(ci - 2 * di, 0):min(ci + 2 * di, image.shape[1]), max(cj - 2 * dj, 0):min(cj + 2 * dj, image.shape[2])] = 1
                bbox = np.moveaxis(np.expand_dims(bbox, -1), 2, 0)
                gt_response = np.moveaxis(np.expand_dims(gt_response, -1), 2, 0)
                return torch.from_numpy(image), torch.from_numpy(impulse), torch.from_numpy(gt_response), torch.from_numpy(bbox), torch.tensor(one_hot)
            else:
                instance_index += 1

    def __len__(self):
        return sum(self.cw_num_instances)


    def visualize_data(self):
        n = len(self)
        for i in range(n):
            image, impulse, response, bbox, one_hot = [data.numpy() for data in self[i]]
            image = np.moveaxis(image, 0, -1)
            image *= self.config.STD_PIXEL
            image += self.config.MEAN_PIXEL
            image *= 255
            impulse = impulse * 255
            response = np.squeeze(response) * 255
            bbox = np.squeeze(bbox) * 255
            image[:,:,0] = np.where(impulse[0]==255,random.choice(range(50,150)),image[:,:,0])
            # image[:,:,1] = np.where(impulse[0]==255,random.choice(range(50,150)),image[:,:,1])
            # image[:,:,2] = np.where(impulse[0]==255,random.choice(range(50,150)),image[:,:,2])
            Image.fromarray(image.astype(np.uint8), "RGB").show()
            # for i in range(impulse.shape[0]):
            #     Image.fromarray(impulse[-1].astype(np.uint8), "L").show()
            # Image.fromarray(response.astype(np.uint8), "L").show()
            # Image.fromarray(bbox.astype(np.uint8), "L").show()
            print(self.config.CLASS_NAMES[np.argmax(one_hot)])
            input()

    def weighted_sampler(self):
        config = self.config
        # TODO: define weighted sampler weights based on data_order
        data_order = config.DATA_ORDER
        class_weighting = np.array(self.cw_num_instances)
        # class_weighting = np.log2(class_weighting)**2
        class_weighting = class_weighting**0.5
        # class_weighting[0] = 0
        class_weighting = class_weighting / np.sum(class_weighting)
        np.random.seed()
        while True:
            yield np.random.choice(self.class_ids, p=class_weighting)

    # i->y, j->x
    def extract_bbox(self, mask):
        m = np.where(mask != 0)
        # y1,x1,y2,x2. bottom right just outside of blah
        return np.min(m[0]), np.min(m[1]), np.max(m[0]) + 1, np.max(m[1]) + 1

    def random_crop(self, image_obj, umask_obj, mask_obj, b, bbox):
        w, h = umask_obj.size
        y1, x1, y2, x2 = bbox
        p = int(random.uniform(max(0, x2 - b), min(w - b, x1)))
        q = int(random.uniform(max(0, y2 - b), min(h - b, y1)))
        return image_obj.crop((p, q, p + b, q + b)), umask_obj.crop((p, q, p + b, q + b)), mask_obj.crop((p, q, p + b, q + b))

    def random_impulse(self, umask_obj):
        umask = np.array(umask_obj)
        w, h = umask.shape
        small_umask = ndimage.convolve(umask, np.ones((32,32)), mode='constant', cval=0.0)
        idx = np.where(small_umask==np.max(small_umask))
        impulse = np.zeros((1,) + umask.shape)
        locx,locy = random.choice(list(zip(idx[0],idx[1])))
        # dimensions of impulse
        l = 8
        n = 4
        # for i in range(4):
        #     L = (l // 2)*(2**i)
        #     impulse[i][max(locx - L, 0):min(locx + L, w), max(locy - L, 0):min(locy + L, h)] = 1
        impulse[0] = umask
        return impulse
    def perturbed_bbox(self,umask):
        y1,x1,y2,x2 = self.extract_bbox(umask)
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        # skew = random.uniform(0.7,0.9)
        skew = 1
        lx = int((x2-x1)*skew)//2
        ly = int((y2-y1)*skew)//2
        bbox_impulse = np.zeros_like(umask)
        bbox_impulse[cy-ly:cy+ly,cx-lx:cx+lx] = 1
        return bbox_impulse
    # unscaled image, masks

    def generate_targets(self, image, masks, class_id, is_crowd):

        config = self.config
        num_classes = self.config.NUM_CLASSES

        mask = masks[:, :, 0]
        umask = masks[:, :, 1]

        # very small objects. will be ignored now and retrained later
        # should probably keep crowds like oranges etc
        if is_crowd or np.sum(mask) < 900:
            return None, None, None, None, True
        if np.sum(np.array(umask)) / np.sum(np.array(mask)) < 0.3:
            umask = mask


        # umask_obj now denotes where it is the only object present
        # mask_obj is the ground truth annotation
        # using umask_obj we generate impulse
        image_obj = Image.fromarray(image, "RGB")
        mask_obj = Image.fromarray(mask, "L")
        umask_obj = Image.fromarray(umask, "L")


        # fliplr = (random.random() > 0.5)
        fliplr = False
        coeff = self.random_coeff(mask,0.08,0.12)

        image_obj = self.resize_image(image_obj, (448, 448), "RGB", coeff, fliplr)
        mask_obj = self.resize_image(mask_obj, (448, 448), "L", coeff, fliplr)
        umask_obj = self.resize_image(umask_obj, (448, 448), "L", coeff, fliplr)

        # # code to crop stuff y1, x1, y2, x2
        # bbox = self.extract_bbox(np.array(mask_obj))
        # y1, x1, y2, x2 = bbox
        # b = config.CROP_SIZE

        # # big object
        # if (x2 - x1) > 180 or (y2 - y1) > 180:
        #     image_obj = self.resize_image(image_obj, (b, b), "RGB")
        #     umask_obj = self.resize_image(umask_obj, (b, b), "L")
        #     mask_obj = self.resize_image(mask_obj, (b, b), "L")
        # # small object
        # else:
        #     image_obj, umask_obj, mask_obj = self.random_crop(image_obj, umask_obj, mask_obj, b, bbox)

        impulse = self.random_impulse(umask_obj)
        impulse[-1] = np.array(mask_obj)
        gt_response = mask_obj
        one_hot = np.zeros(81)
        one_hot[class_id] = 1
        return np.array(image_obj).astype(np.float32), np.array(impulse).astype(np.float32), np.array(gt_response).astype(np.float32), np.array(one_hot).astype(np.float32), False

    def read_image(self, image_id):
        image = Image.open(self.data_dir + image_id).convert("RGB")
        return np.array(image)

    def resize_image(self, image_obj, thumbnail_shape, mode, coeff, fliplr = False):

        interpolation = {"RGB":Image.BICUBIC,"L":Image.NEAREST}[mode]

        if fliplr == True:
            image_obj = image_obj.transpose(Image.FLIP_LEFT_RIGHT)

        image_obj = image_obj.transform(thumbnail_shape, Image.PERSPECTIVE, coeff, interpolation)

        # z = Image.new(mode, thumbnail_shape, "black")
        return image_obj

    def load_image_gt(self, class_id, instance_index):
        config = self.config
        cwid = self.cwid
        instance_info = cwid[class_id][instance_index]
        image_id = instance_info["image_id"]
        mask_obj = instance_info["mask_obj"]
        is_crowd = instance_info['is_crowd']
        image = self.read_image(image_id)
        masks = maskUtils.decode(mask_obj)

        # if random.random() > 0.5:
        #     image = np.fliplr(image)
        #     masks = np.fliplr(masks)

        return image, masks, is_crowd


    def distort_gen(self, min_dis, max_dis):
        while True:
            yield random.uniform(min_dis, max_dis)

    def random_coeff(self, mask, min_dis, max_dis):
        width,height = mask.shape
        y1,x1,y2,x2 = self.extract_bbox(mask)
        m = self.distort_gen(min_dis, max_dis)
        pi = [(0, 0), (width, 0), (0, height), (width, height)]
        pf = [(0 - next(m) * width, 0 - next(m) * height), (width + next(m) * width, 0 - next(m) * height),
              (0 - next(m) * width, height + next(m) * height), (width + next(m) * width, height + next(m) * height)]
 
        matrix = []
        for p1, p2 in zip(pf, pi):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -
                           p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -
                           p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pi).reshape(8)

        res = np.linalg.solve(A, B)
        return np.array(res).reshape(8)


def get_loader(cwid, config, data_dir):
    coco_dataset = CocoDataset(cwid, config, data_dir)
    data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              # collate_fn=_collate_fn,
                                              shuffle=True,
                                              pin_memory=config.PIN_MEMORY,
                                              num_workers=config.NUM_WORKERS
                                              )
    return data_loader


# proposes masks for single scale feature maps.
# for multi_scale supervision, use this multiple times


class MaskProp(nn.Module):

    def __init__(self, init_weights=True):
        super(MaskProp, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(640 + 640, 640, (3, 3), padding=(1, 1)), nn.BatchNorm2d(640), 
            self.relu,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(640 + 640, 320, (5, 5), padding=(2, 2)), nn.BatchNorm2d(320), 
            self.relu,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(320 + 320, 160, (7, 7), padding=(3, 3)), nn.BatchNorm2d(160), 
            self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(160 + 160, 80, (9, 9), padding=(4, 4)), nn.BatchNorm2d(80), 
            self.relu,
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(80 + 80, 1, (9, 9), padding=(4, 4)), nn.BatchNorm2d(1), 
            # self.relu,
        )
        # point wise max pool
        # self.pmp = nn.MaxPool3d((80,1,1),stride=1)

        if init_weights:
            for name, child in self.named_children():
                if name[:-1] == 'layer' or 'mask_layer':
                    for gc in child.children():
                        if isinstance(gc, nn.Conv2d):
                            nn.init.xavier_uniform_(gc.weight)

    def forward(self, x):
        c, m = x
        l1, l2, l3, l4, l5 = m

        c = F.upsample(c, scale_factor=2)
        y = self.layer5(torch.cat([c, l5], 1))

        y = self.upsample(y)
        y = self.layer4(torch.cat([y, l4], 1))

        y = self.upsample(y)
        y = self.layer3(torch.cat([y, l3], 1))

        y = self.upsample(y)
        y = self.layer2(torch.cat([y, l2], 1))

        y = self.upsample(y)
        y = self.layer1(torch.cat([y, l1], 1))

        # unsqueeze and max pool over the channels
        # y = torch.unsqueeze(y,1)
        # y = self.pmp(y)
        # y = torch.squeeze(y).unsqueeze(1)

        return y


# classifier takes a single level features and classifies


class Classifier(nn.Module):

    def __init__(self, init_weights=True):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(640, 512, (3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(512, 256, (1, 1))

        self.gap = nn.AvgPool2d((14, 14), stride=1)
        # self.gap = nn.MaxPool2d((14, 14), stride=1)
        self.fc = nn.Linear(256*14*14, 81)
        self.relu = nn.ReLU(inplace=True)
        if init_weights:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.gap(x)
        # x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MultiHGModel(nn.Module):

    def __init__(self):
        super(MultiHGModel, self).__init__()
        self.vgg0 = modified_vgg.split_vgg16_features(pre_trained_weights=False, d_in=1)
        # self.vgg1 = modified_vgg.split_vgg16_features(pre_trained_weights=False, d_in=4)
        # self.mp0 = MaskProp()
        # self.mp1 = MaskProp()
        self.class_predictor = Classifier()

    def forward(self, x):
        im, impulse = x

        inp = torch.cat([im, impulse], dim=1)
        class_features, mask_features = self.vgg0(inp)
        # m0 = self.mp0([class_features, mask_features])

        # impulse[:,-1,:,:] = F.threshold(F.sigmoid(m0.squeeze()),0.5,0)
        
        # inp = torch.cat([im, impulse], dim=1)
        # class_features, mask_features = self.vgg1(inp)
        # m1 = self.mp1([class_features, mask_features])
        
        c = self.class_predictor(class_features)

        return c, [0,0]


class SimpleHGModel(nn.Module):

    def __init__(self):
        super(SimpleHGModel, self).__init__()
        self.vgg0 = modified_vgg.split_vgg16_features(pre_trained_weights=False, d_in=4)
        # self.vgg1 = modified_vgg.split_vgg16_features(pre_trained_weights=False, d_in=32)
        self.mp0 = MaskProp()
        # self.mp1 = MaskProp()
        # self.mp2 = MaskProp()
        # self.mp3 = MaskProp()
        self.class_predictor = Classifier()

    def forward(self, x):
        im, base_impulse = x
        del x
        outs = []
        inp = torch.cat([im, base_impulse], dim=1)
        class_features, mask_features = self.vgg0(inp)
        m0 = self.mp0([class_features, mask_features])
        outs.append(m0)
        # gen_impulse = F.threshold(F.sigmoid(F.upsample(m0[-1], scale_factor=4)),0.5,0)

        # base_impulse = base_impulse[:,:-1,...]

        # inp = torch.cat([im, y], dim=1)
        # class_features, mask_features = self.vgg1(inp)
        # y, m1 = self.mp0([class_features, mask_features])
        # outs.append(m1)
        # gen_impulse = F.upsample(m1[-1], scale_factor=4)

        # inp = torch.cat([im, base_impulse,gen_impulse], dim=1)
        # class_features, mask_features = self.vgg(inp)
        # m2 = self.mp2([class_features, mask_features])
        # outs.append(m2)
        # gen_impulse = F.upsample(m2[-1], scale_factor=4)

        # inp = torch.cat([im, base_impulse,gen_impulse], dim=1)
        # class_features, mask_features = self.vgg(inp)
        # m3 = self.mp3([class_features, mask_features])
        # outs.append(m3)
        # gen_impulse = F.upsample(m3[-1], scale_factor=4)

        c = self.class_predictor(class_features)
        # return c, outs
        return c, m0


def loss_criterion(pred_class, gt_class, pred_masks, gt_mask, bbox):
    idx = gt_class[..., 0].nonzero()
    mask_weights = torch.cuda.FloatTensor(gt_class.shape[0]).fill_(1)
    mask_weights[idx] = 0
    mask_weights = mask_weights.view(-1, 1, 1, 1)
    loss1 = ce_class_loss(pred_class, gt_class)
    loss2 = mask_loss(pred_masks, gt_mask, mask_weights, bbox, 4)
    return loss1, loss2


def multi_mask_loss_criterion(pred_class, gt_class, pred_masks, gt_mask, bbox):
    idx = gt_class[..., 0].nonzero()
    mask_weights = torch.cuda.FloatTensor(gt_class.shape[0]).fill_(1)
    mask_weights[idx] = 0
    mask_weights = mask_weights.view(-1, 1, 1, 1)
    loss1 = ce_class_loss(pred_class, gt_class)
    # loss2 = mask_loss(pred_masks[1], gt_mask, mask_weights, bbox, 1)+0.2*mask_loss(pred_masks[0], gt_mask, mask_weights, bbox, 1) 
    # reg = soft_iou_reg(pred_masks[0],pred_masks[1],mask_weights,gt_mask,1)

    # return loss1, loss2+reg
    return loss1, loss1


# gt_mask: N,1,w,h
# pred_masks: N,1,w/scale_down,h/scale_down

def full_mask_loss(pred_masks, target):
    _loss = nn.BCEWithLogitsLoss(reduce=False)
    l = _loss(pred_masks, target)
    return l


def bbox_mask_loss(pred_masks, target, bbox):
    _loss = nn.BCEWithLogitsLoss(weight=bbox, reduce=False)
    l = _loss(pred_masks, target)
    return l


def mask_loss(pred_masks, gt_mask, mask_weights, bbox, scale_down):
    target = F.max_pool2d(gt_mask, (scale_down, scale_down), stride=scale_down)
    bbox = F.max_pool2d(bbox, (scale_down, scale_down), stride=scale_down)
    f = full_mask_loss(pred_masks, target)
    b = bbox_mask_loss(pred_masks, target, bbox)
    w_bbox = bbox.sum(-1).sum(-1).view(-1, 1, 1, 1)
    # print(w_bbox.squeeze())
    w_full = torch.cuda.FloatTensor(gt_mask.shape[0]).fill_(56 * 56).view(-1, 1, 1, 1)
    # b = b
    # f = f
    l = (b / w_bbox + f / w_full)
    # l = (b+f)/w_bbox
    # print(b.mean().item(),f.mean().item(),l.mean().item())
    l *= mask_weights
    # print(mask_weights.squeeze())
    return l.sum(-1).sum(-1).mean()
def soft_iou_loss(pred_masks, gt_mask, mask_weights, bbox, scale_down):
    target = F.max_pool2d(gt_mask, (scale_down, scale_down), stride=scale_down)
    bbox = F.max_pool2d(bbox, (scale_down, scale_down), stride=scale_down)
    pred_masks = F.sigmoid(pred_masks)
    i = (pred_masks*target).sum(-1).sum(-1)
    u = (pred_masks+target-pred_masks*target).sum(-1).sum(-1)
    l = 1-i/u
    # print(l.shape,mask_weights.shape)
    l *= mask_weights.view(-1,1)
    return l.mean()
def soft_iou_reg(m0, m1, mask_weights, gt_mask, scale_down):
    target = F.max_pool2d(gt_mask, (scale_down, scale_down), stride=scale_down)
    m0 = F.sigmoid(m0)
    m1 = F.sigmoid(m1)
    pi = (m0*m1).sum(-1).sum(-1)
    pu = (m0+m1-m0*m1).sum(-1).sum(-1)
    gi = (m0*target).sum(-1).sum(-1)
    gu = (m0+target-m0*target).sum(-1).sum(-1)
    # l = (gi/gu)/(pi/pu) 
    # l = (gi*pu)/(pi*gu) 
    l = (pi/pu)*(1-gi/gu)
    # print(l.shape,mask_weights.shape)
    l *= mask_weights.view(-1,1)
    return l.mean()
# def mask_loss(pred_masks, gt_mask, mask_weights, scale_down):
#     target = F.max_pool2d(gt_mask, (scale_down, scale_down), stride=scale_down)
#     fg_size = gt_mask.squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1) / (scale_down**2)
#     bg_size = (1 - gt_mask).squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1) / (scale_down**2)
#     # bgfg_weighting = (target == 1).float() + (target == 0).float()
#     bgfg_weighting = (target == 1).float() + (target == 0).float() * fg_size / bg_size
#     bgfg_weighting *= mask_weights
#     _loss = nn.BCEWithLogitsLoss(weight=bgfg_weighting, reduce=False)
#     l = _loss(pred_masks, target)
#     # l = l.squeeze().sum(-1).sum(-1)
#     l = l.mean()
#     return l


def ce_class_loss(pred_class, gt_class):
    _loss = nn.CrossEntropyLoss(reduce=False)
    labels = gt_class.nonzero()[:, 1]
    l = _loss(pred_class, labels)
    l = l.mean()
    return l


def bce_class_loss(pred_class, gt_class):
    idx = gt_class.nonzero()[:, 0]
    labels = gt_class.nonzero()[:, 1]
    w = torch.cuda.FloatTensor(gt_class.shape).fill_(1 / 160)
    w[idx, labels] = 1 / 2
    _loss = nn.BCEWithLogitsLoss(weight=w, reduce=False)
    l = _loss(pred_class, gt_class)
    return l.sum(-1).mean()


def mml_class_loss(pred_class, gt_class):
    _loss = nn.MultiMarginLoss(reduce=False)
    labels = gt_class.nonzero()[:, 1]
    l = _loss(pred_class, labels)
    l = l.mean()
    return l


def accuracy(pred_class, batch_one_hot, pred_masks, batch_gt_responses):
    idx = batch_one_hot[..., 0].nonzero()
    mask_weights = torch.cuda.FloatTensor(batch_one_hot.shape[0]).fill_(1)
    mask_weights[idx] = 0
    class_score = class_acc(pred_class, batch_one_hot)
    # mask_score = mask_acc(pred_masks, batch_gt_responses, mask_weights)
    return class_score, class_score

def class_acc(pred_class, batch_one_hot):
    with torch.no_grad():
        labels = batch_one_hot.nonzero()[:, 1]
        maxs, indices = torch.topk(pred_class, 5, -1)
        # print(labels,indices[:,0])
        # return (indices[:, 0] == labels).sum() / batch_one_hot.shape[0]
        return (indices[:,0]==labels).float().mean()

def mask_acc(pred_masks, batch_gt_responses,mask_weights):
    stride = 1
    with torch.no_grad():
        target = F.max_pool2d(batch_gt_responses, (stride, stride), stride=stride).float()
        pred_masks = F.sigmoid(pred_masks)
        pred_masks = F.threshold(pred_masks, 0.5, 0)
        pred_masks = (pred_masks > 0).float()
        # print(pred_masks)
        # print(target)
        union = (pred_masks + target) > 0
        union = union.sum(-1).sum(-1).float()
        # print(union)
        intersection = (pred_masks * target) > 0
        intersection = intersection.sum(-1).sum(-1).float()
        # print(intersection)
        iou = (intersection / union).squeeze()
        # print(iou.shape,mask_weights.shape)
        iou *= mask_weights
        iou = iou.sum()
        return iou / mask_weights.sum()
# TODO: modify dummy stub to train code or inference code


def main():
    return 0
if __name__ == '__main__':
    main()
