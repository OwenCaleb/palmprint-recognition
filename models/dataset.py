# -*- coding:utf-8 -*-
import torch
from torch.utils import data
from torchvision import transforms as T
import os
import cv2
import math
import time
import numpy as np
import torch.nn as nn
from Loss import _gather_feat
from Loss import _transpose_and_gather_feat
from PIL import Image
from models.backbone.dlanet_dcn import DlaNet

def mk_file(file_path: str):
    if os.path.exists(file_path):
        return
    os.makedirs(file_path)

# 未使用
def rotateImage(img, degree, x, y, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(
        width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(
        height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
    matRotation = cv2.getRotationMatrix2D((x, y), degree, 1)
    # matRotation[0, 2] += (widthNew - width) / 2
    # matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (2000, 2000), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgRotation = imgRotation[int(pt3[0]):int(pt1[0]), int(pt1[1]):int(pt3[1])]
    return imgRotation

def rotate_point(point, center, angle_rad):
    """Rotate a point around a center point by a given angle."""
    x, y = point
    cx, cy = center
    new_x = (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad) + cx
    new_y = (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad) + cy
    return new_x, new_y

# Non-Maximum Suppression keep max-value within margin in 3*3
def _nms(heat, kernel=3):
    # keep size
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    # (batch, cat, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # 确保有效范围
    topk_inds = topk_inds % (height * width)
    # 注意 图像坐标 PIL not numpy 坐标
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # (batch, K) 前c个最有可能的
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # 最可能class
    topk_clses = (topk_ind / K).int()
    # 前40个 的图像index value:0-wh-1
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    # 前40个的 y
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

# 功能函数
def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]  #原始图像的width 416 608
    dst_w = output_size[0] #、要是保持清晰度就裁剪，不进行缩放 scale就是原始 否则scale就是目标大小
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    # 这里在标签中应该处理了 角度问题
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    # 锚点，中心点不变
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b): #获得直角三角形第三个点
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        c, h, w = tensor.size()

        if c is not 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


class MyTestDataset(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''

    def __init__(self, txt,model,cudadevice, transforms=None, train=True, imside=128, outchannels=1):

        pth_name = 'result_img_'
        path_name_all = pth_name + 'img_output_all'
        mk_file(path_name_all + '/fail_images/')
        # 目前先用Tongji 为了方便
        # self.images_path_all = [
        #     "Tongji",
        # ]

        self.img_path = './Tongji/'
        self.model=model
        self.device=cudadevice
        self.train = train

        self.imside = imside  # 128, 224
        self.chs = outchannels  # 1, 3

        self.text_path = txt

        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)

                ])
            else:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        T.RandomResizedCrop(size=self.imside, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC,expand=False,
                                             center=(0.5 * self.imside, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False,
                                             center=(0.0, 0.5 * self.imside)),
                        ]),
                    ]),

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])

    def __getitem__(self, index):
        label = self.images_label[index]
        img_path1 = self.images_path[index]
        idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])
        if self.train == True:
            while (idx2 == index):
                idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])
                # img_path2 = self.images_path[idx2]
        else:
            idx2 = index
        img_path2 = self.images_path[idx2]
        # 问题1：在测试时，等于拿着同一张掌纹ROI进行测试？

        img1=self.read_numpy_image(img_path1)
        img2=self.read_numpy_image(img_path2)


        images1, meta1 = self.pre_process(img1)
        images2, meta2 = self.pre_process(img2)

        images1 = images1.to(self.device)
        images2 = images2.to(self.device)

        # model result + bbox
        output1, dets1 = self.process(images1)
        output2, dets2 = self.process(images2)

        dets1 = self.post_process(dets1, meta1)
        dets2 = self.post_process(dets2, meta2)

        # 保留置信度高的检测框,final bbox
        # print("dest1")
        # print(np.shape(dets1))
        # print(np.shape(dets2))

        ret1 = self.merge_outputs(dets1)
        ret2 = self.merge_outputs(dets2)

        res1 =self.getMax(ret1)
        res2 =self.getMax(ret2)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        data1 = self.draw(img_path1, res1,img1).convert('L')
        data2 = self.draw(img_path2, res2,img2).convert('L')

        # ./data_base/Tongji
        # if not res:  不再需要，直接取最大置信，如果0.3不行的话
        #     # 如果 `data2` 是 None，则使用以下方法
        #     data = Image.open(img_path).convert('L')
        # else:
        #     data=draw(img_path, file_name1, res).convert('L')

        data1 = self.transforms(data1)
        data2 = self.transforms(data2)

        data = [data1, data2]
        # min_shape = min([item.shape for item in data])
        data_resized = [item[:, :128, :128] for item in data]
        # 注意 涉及到stack DataLoader隐式处理
        return data_resized, int(label)

    def read_numpy_image(self,path):
        file_name = os.path.basename(path)
        if len(file_name.split()) < 3:
            img1 = cv2.imread(path)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = np.zeros_like(img1)
            img2[:, :, 0] = gray
            img2[:, :, 1] = gray
            img2[:, :, 2] = gray
            img = img2
        elif len(path.split()) > 3:
            img1 = cv2.imread(path)
            img2 = img1.convert('RGB')
            img = img2
        else:
            img = cv2.imread(path)
        return img

    def __len__(self):
        return len(self.images_path)

    def pre_process(self,image):
        height, width = image.shape[0:2]
        # 设置目标图像的尺寸512x512
        inp_height, inp_width = 512, 512

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height]) # for numpy
        inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

        mean = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)

        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 三维reshape到4维，（1，3，512，512）

        images = torch.from_numpy(images)

        # original
        meta = {'c': c, 's': s,
                'out_height': inp_height // 4,
                'out_width': inp_width // 4}
        return images, meta

    def process(self,images, return_time=False):
        with torch.no_grad():
            output = self.model(images)
            hm = output['hm'].sigmoid_()
            ang = output['ang'].relu_()
            wh = output['wh']
            reg = output['reg']
            # CUDA 操作同步 GPU 上的所有计算任务都已完成
            # torch.cuda.synchronize()
            # forward_time = time.time()
            dets = self.ctdet_decode(hm, wh, ang, reg=reg, K=100)
            return output, dets

    # 目标检测的解码函数
    def ctdet_decode(self , heat, wh, ang, reg=None, K=100):
        batch, cat, height, width = heat.size()
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=K)

        # get Top-K reg
        reg = _transpose_and_gather_feat(reg, inds)

        reg = reg.view(batch, K, 2)
        # 中心坐标+偏移量
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        ang = _transpose_and_gather_feat(ang, inds)
        ang = ang.view(batch, K, 1)

        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2,
                            ang], dim=2)
        # (batch_size, K, 7)
        detections = torch.cat([bboxes, scores, clses], dim=2)
        return detections

    def post_process(self,dets, meta):
        # bbox meta
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])

        num_classes = 1 #目标只有1个

        # 后处理 仿射回来
        dets = self.ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'],
                                  num_classes)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
            dets[0][j][:, :5] /= 1#看起来多余
        return dets[0]

    def ctdet_post_process(self,dets, c, s, h, w, num_classes):
        # dets: batch x max_dets x dim  4+ 2+ 1
        # return 1-based class det dict
        ret = []
        for i in range(dets.shape[0]):
            top_preds = {}
            dets[i, :, :2] = self.transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
            dets[i, :, 2:4] = self.transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
            classes = dets[i, :, -1]
            for j in range(num_classes):#只有一个类别
                inds = (classes == j)#只寻找 是不是 掌纹
                top_preds[j + 1] = np.concatenate([
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()#Python列表格式
            ret.append(top_preds)
        return ret

    # 仿射变换
    def transform_preds(self,coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        # 逆变，反转为正常框
        trans = get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        return target_coords

    def merge_outputs(self,detections):
        num_classes = 1
        max_obj_per_img = 100
        scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
        if len(scores) > max_obj_per_img:
            kth = len(scores) - max_obj_per_img
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, 2 + 1):
                keep_inds = (detections[j][:, 5] >= thresh)
                detections[j] = detections[j][keep_inds]
        return detections

    def getMax(self,ret):
        res = np.empty([1, 7])
        for i, c in ret.items():
            # 提取置信度大于0.3的检测结果
            tmp_s = ret[i][ret[i][:, 5] > 0.3]
            if len(tmp_s) == 0:
                # 如果没有置信度大于0.3的结果，则提取最大置信度的结果
                max_conf_idx = np.argmax(ret[i][:, 5])  # 获取最大置信度的索引
                max_conf_detection = ret[i][max_conf_idx:max_conf_idx + 1]  # 提取最大置信度的检测结果
                tmp_s = max_conf_detection

            tmp_c = np.ones(len(tmp_s)) * (i + 1)
            tmp = np.c_[tmp_c, tmp_s]
            res = np.append(res, tmp, axis=0)

        res = np.delete(res, 0, 0)
        res = res.tolist()
        return res

    def draw(self,path, res,img):
        image = Image.open(path)
        for class_name, lx, ly, rx, ry, ang, prob in res:
            result = [int((rx + lx) / 2), int((ry + ly) / 2), int(rx - lx), int(ry - ly), ang]
            result = np.array(result)
            x = int(result[0])
            y = int(result[1])
            height = int(result[2])
            width = int(result[3])
            angle = result[4]

            # 指定新的尺寸
            lenge = max(image.width, image.height)
            new_width = lenge * 3
            new_height = lenge * 3

            # 计算左上角的坐标，以在新尺寸内居中显示图像
            left = (new_width - image.width) // 2
            top = (new_height - image.height) // 2

            # 创建新的画布，填充为黑色
            padded_image = Image.new('RGB', (new_width, new_height), color='black')

            # 将原始图像粘贴到新的画布中
            padded_image.paste(image, (left, top))
            center_x = x + (new_width - image.width) // 2
            center_y = y + (new_height - image.height) // 2
            # 将角度转换为弧度
            angle_rad = math.radians(angle)

            # 计算旋转框的四个角点坐标
            top_left = (center_x - width / 2, center_y - height / 2)
            top_right = (center_x + width / 2, center_y - height / 2)
            bottom_left = (center_x - width / 2, center_y + height / 2)
            bottom_right = (center_x + width / 2, center_y + height / 2)

            # 得到新的旋转框的四个顶点坐标
            new_rect_points = [top_left, top_right, bottom_right, bottom_left]

            rotated_image = padded_image.rotate(-angle + 90, center=(center_x, center_y), resample=Image.BICUBIC,
                                                expand=False)

            # 得到新的旋转框的坐标范围
            min_x = min(point[0] for point in new_rect_points)
            max_x = max(point[0] for point in new_rect_points)
            min_y = min(point[1] for point in new_rect_points)
            max_y = max(point[1] for point in new_rect_points)

            # 裁剪旋转后的图像
            cropped_image = rotated_image.crop((min_x, min_y, max_x, max_y))
            return cropped_image
            break


class MyTrainDataset(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''

    def __init__(self, txt, transforms=None, train=True, imside=128, outchannels=1):

        self.train = train

        self.imside = imside  # 128, 224
        self.chs = outchannels  # 1, 3

        self.text_path = txt

        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)

                ])
            else:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        T.RandomResizedCrop(size=self.imside, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC,expand=False,
                                             center=(0.5 * self.imside, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False,
                                             center=(0.0, 0.5 * self.imside)),
                        ]),
                    ]),

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_label[index]
        # print(img_path)
        # print(img_path)

        idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])

        if self.train == True:
            while (idx2 == index):
                idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])
                # img_path2 = self.images_path[idx2]
        else:
            idx2 = index

        img_path2 = self.images_path[idx2]

        data = Image.open(img_path).convert('L')
        data = self.transforms(data)

        data2 = Image.open(img_path2).convert('L')
        data2 = self.transforms(data2)

        data = [data, data2]
        # print(data)
        # print(label)
        return data, int(label)  # , img_path

    def __len__(self):
        return len(self.images_path)