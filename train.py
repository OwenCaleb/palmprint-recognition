import os
# os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse
import time
import sys
import multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.backbone.dlanet_dcn import DlaNet
plt.switch_backend('agg')
from torch.optim import lr_scheduler
import pickle
import cv2 as cv
from loss import SupConLoss
from models.dataset import MyTrainDataset,MyTestDataset
from models.ccnet import ccnet
from utils import *
import numpy as np

def init_roi_model(map):
    # dlaNet
    gpus = [0,1,2,3]
    model = DlaNet(34)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    checkpoint_path = map['des_path']+'best_roi_model.pth'
    if os.path.exists(checkpoint_path):
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k.replace('module.', '')  # 去掉 'module.' 前缀
        #     new_state_dict[name] = v
        print("model '{}' exists.".format(checkpoint_path))
        # 注意，以相同的顺序加载保证环境一致
        checkpoint = torch.load(checkpoint_path)
        loaded_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        if loaded_keys == checkpoint_keys:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        else:
            print("Warning: Model keys do not match exactly. Exit")
            sys.exit(1)
    else:
        print("model '{}' does not exist. Exit.".format(checkpoint_path))
        sys.exit(1)
    return model.eval()
def arg():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="CO3Net for Palmprint Recfognition")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_num", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--weight1", type=float, default=0.8)
    parser.add_argument("--weight2", type=float, default=0.2)
    parser.add_argument("--com_weight", type=float, default=0.8)
    parser.add_argument("--id_num", type=int, default=600,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    parser.add_argument("--gpu_id", type=str, default='0,1,2,3')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--redstep", type=int, default=500)
    parser.add_argument("--test_interval", type=str, default=1000)
    parser.add_argument("--save_interval", type=str, default=500)  ## 200 for Multi-spec 500 for RED
    ##Training Path
    parser.add_argument("--train_set_file", type=str, default='./data/train_Tongji.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/test_Tongji.txt')
    parser.add_argument("--inference_train_set_file", type=str, default='./data/val_train_Tongji.txt')
    parser.add_argument("--inference_test_set_file", type=str, default='./data/val_test_Tongji.txt')
    ##Store Path
    parser.add_argument("--des_path", type=str, default='./results/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/rst_test/')
    parser.add_argument("--dataset_name", type=str, default='Tongji')
    return parser
def set_arg(parser):
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    params_map = {
        'batch_size': args.batch_size,
        'epoch_num': args.epoch_num,
        'temp': args.temp,
        'weight1': args.weight1,
        'weight2': args.weight2,
        'com_weight': args.com_weight,
        'id_num': args.id_num,
        'gpu_id': args.gpu_id,
        'lr': args.lr,
        'redstep': args.redstep,
        'test_interval': args.test_interval,
        'save_interval': args.save_interval,
        'train_set_file': args.train_set_file,
        'test_set_file': args.test_set_file,
        'inference_train_set_file':args.inference_train_set_file,
        'inference_test_set_file':args.inference_test_set_file,
        'des_path': args.des_path,
        'path_rst': args.path_rst,
        'dataset_name': args.dataset_name
    }
    # 根据键获取值并打印
    print("Batch Size:", params_map['batch_size'])
    print("Epoch Number:", params_map['epoch_num'])
    print("Temp:", params_map['temp'])
    print("Weight 1:", params_map['weight1'])
    print("Weight 2:", params_map['weight2'])
    print("Component Weight:", params_map['com_weight'])
    print("Destination Path:", params_map['des_path'])
    print("Result Path:", params_map['path_rst'])
    print("Train Set File:", params_map['train_set_file'])
    print("Test Set File:", params_map['test_set_file'])
    print("Inference Train Set File:", params_map['inference_train_set_file'])
    print("Inference Test Set File:", params_map['inference_test_set_file'])
    print("Dataset Name:", params_map['dataset_name'])
    if not os.path.exists(params_map['des_path']):
        os.makedirs(params_map['des_path'])
    if not os.path.exists(params_map['path_rst']):
        os.makedirs(params_map['path_rst'])
    # output dir
    # ./results/rst_test/
    if not os.path.exists(params_map['path_rst']):
        os.makedirs(params_map['path_rst'])
    # ./results/rst_test/rank1_hard
    path_hard = os.path.join(params_map['path_rst'], 'rank1_hard')
    if not os.path.exists(path_hard):
        os.makedirs(path_hard)
    # ./results/rst_test/veriEER
    if not os.path.exists(params_map['path_rst']+'veriEER'):
        os.makedirs(params_map['path_rst']+'veriEER')
    # ./results/rst_test/veriEER/rank1_hard/
    if not os.path.exists(params_map['path_rst']+'veriEER/rank1_hard/'):
        os.makedirs(params_map['path_rst']+'veriEER/rank1_hard/')
    return params_map
def init_ccnet_model(map,Train):
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    print('------Init Model------')
    # set CCNET model
    net = ccnet(num_classes=map['id_num'], weight=map['com_weight'])
    net = net.cuda()
    if Train:
        checkpoint_path = map['des_path'] + 'net_train_ccnet_best.pth'
    else :
        checkpoint_path = map['des_path'] + 'net_ccnet_best.pth'
        net.eval()
    if os.path.exists(checkpoint_path):
        print("Model ccnet '{}' exists.".format(checkpoint_path))
        # 注意，以相同的顺序加载保证环境一致
        checkpoint = torch.load(checkpoint_path)
        loaded_keys = set(net.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        if loaded_keys == checkpoint_keys:
            net.load_state_dict(checkpoint)
            print("Model ccnet loaded successfully.")
        else:
            print("Warning: Model ccnet keys do not match exactly.")
    else:
        print("model ccnet '{}' does not exist. Start form scratch.".format(checkpoint_path))
    return net
def get_my_Dataset(train_set_file,test_set_file,transforms,imside,batch_size,num_workers,shuffle):
    # dataset use Demo
    # trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    # testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)
    trainset = MyTrainDataset(txt=train_set_file, transforms=transforms, train=True, imside=imside, outchannels=1)
    testset = MyTrainDataset(txt=test_set_file, transforms=transforms, train=False, imside=imside, outchannels=1)
    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    data_loader_test = DataLoader(dataset=testset, batch_size=128, num_workers=num_workers, shuffle=shuffle)
    return data_loader_train,data_loader_test
def get_Opt_Sch(map):
    optimizer = optim.Adam(ccnet_model.parameters(), lr=map['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=map['redstep'], gamma=0.8)
    checkpoint_path = map['des_path']+'{}_optimizer_scheduler.pth'.format(map['dataset_name'])
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Checkpoint file '{}' exists.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print("Checkpoint file '{}' does not exist. Start form scratch.".format(checkpoint_path))
    return optimizer,scheduler,start_epoch
def fit(epoch, model, data_loader,optimizer,criterion,con_criterion,map,phase='training'):
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')
    if phase == 'training':
        model.train()
    if phase == 'testing':
        model.eval()
    running_loss = 0
    running_correct = 0
    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0]
        data = data.cuda()
        data_con = datas[1]
        data_con = data_con.cuda()
        target = target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
            output, fe1 = model(data, target)
            output2, fe2 = model(data_con, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                output2, fe2 = model(data_con, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)
        loss = map['weight1']*ce+map['weight2']*ce2
        ## log
        running_loss += loss.data.cpu().numpy()
        preds = output.data.max(dim=1, keepdim=True)[1]  # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()
        ## update
        if phase == 'training':
            loss.backward(retain_graph=None)  #
            optimizer.step()
    ## log info of this epoch
    total = len(data_loader.dataset)
    loss = running_loss / total
    accuracy = (100.0 * running_correct) / total
    print('epoch %d: \t%s average loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
    epoch, phase, loss, phase, running_correct, total, accuracy))
    return loss, accuracy
def train_model(ccnet_model,map):
    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=map['temp'], base_temperature=map['temp'])
    optimizer,scheduler,start_epoch=get_Opt_Sch(map)
    data_loader_train, data_loader_test=get_my_Dataset(map['train_set_file'],map['test_set_file'],None,128,map['batch_size'],2,True)
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    train_bestacc = 0
    val_bestacc = 0
    epoch_num=map['epoch_num']
    for epoch in range(start_epoch,epoch_num):
        epoch_loss, epoch_accuracy = fit(epoch, ccnet_model, data_loader_train, optimizer,criterion,con_criterion,map,phase='training')
        scheduler.step()
        # ------------------------logs----------------------
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # val_losses.append(val_epoch_loss)
        # val_accuracy.append(val_epoch_accuracy)
        if(epoch%5==0):
            val_epoch_loss, val_epoch_accuracy = fit(epoch, ccnet_model, data_loader_train, optimizer, criterion, con_criterion,map, phase='testing')
            val_losses.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)
            if val_epoch_accuracy >= val_bestacc:
                val_bestacc=val_epoch_accuracy
                torch.save(ccnet_model.state_dict(), map['des_path'] + 'net_ccnet_best.pth')
        # save the best model
        if epoch_accuracy >= train_bestacc:
            train_bestacc = epoch_accuracy
            torch.save(ccnet_model.state_dict(), map['des_path'] + 'net_train_ccnet_best.pth')
            print("Saving net_ccnet_best successfully!")
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, map['des_path']+'{}_optimizer_scheduler.pth'.format(map['dataset_name']))
    print('Finish Trainning------------\n')
    return
def test_model(ccnet_model,map):
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    roi_model = init_roi_model(map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test是要匹配的人脸，train是人脸数据库
    trainset = MyTestDataset(txt=map['inference_train_set_file'], transforms=None, train=False, imside=128, outchannels=1, model=roi_model,cudadevice=device)
    testset = MyTestDataset(txt=map['inference_test_set_file'], transforms=None, train=False, imside=128, outchannels=1, model=roi_model,cudadevice=device)
    # 如果想匹配更多图片，就把batch_size设置小一点
    data_loader_train = DataLoader(dataset=trainset, batch_size=512, num_workers=2, shuffle=True)
    data_loader_test = DataLoader(dataset=testset, batch_size=512, num_workers=2, shuffle=True)

    # 特征提取
    featDB_train = []
    iddb_train = []
    featDB_test = []
    iddb_test = []
    for batch_id, (datas, target) in enumerate(data_loader_train):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()
        codes = ccnet_model.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()
        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))
    for batch_id, (datas, target) in enumerate(data_loader_test):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()
        codes = ccnet_model.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()
        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))
    print('Completed feature extraction!')
    print('featDB_train.shape: ', featDB_train.shape)
    print('iddb_train.shape: ', iddb_train.shape)
    print('featDB_test.shape: ', featDB_test.shape)
    print('iddb_test.shape: ', iddb_test.shape)
    trainclassNumber = len(set(iddb_train))
    testclassNumber = len(set(iddb_test))
    trainsampleNumber = featDB_train.shape[0]
    testsampleNumber = featDB_test.shape[0]
    # trainNum = num_training_samples // classNumel ???
    print('Start Feature Matching and Verification EER')
    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching->label
    ntest  = testsampleNumber
    ntrain = trainsampleNumber
    for i in range(ntest):
        feat1 = featDB_test[i]
        for j in range(ntrain):
            feat2 = featDB_train[j]
            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            s.append(dis)
            if iddb_test[i] == iddb_train[j]:
                l.append(1)
            else:
                l.append(-1)
    with open(map['path_rst'] + 'veriEER/scores_VeriEER.txt', 'w') as f:
        # i*j
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')
    sys.stdout.flush()
    # 在 Python 脚本中执行系统命令。它可以调用外部程序或脚本，在系统环境中运行命令行指令。
    os.system('python ./getGI.py' + '  ' + map['path_rst'] + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + map['path_rst'] + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    print('\n------------------')

    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'
    # ./Tongji/session1/00001.tiff 0 -> [./Tongji/session1/00001.tiff]
    fileDB_train = getFileNames(map['inference_train_set_file'])
    fileDB_test = getFileNames(map['inference_test_set_file'])
    for i in range(ntest):
        probeID = iddb_test[i]
        # ntrain行，1列 这里相当于把score拿回来
        dis = np.zeros((ntrain, 1))
        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1
        idx = np.argmin(dis[:])
        galleryID = iddb_train[idx]
        if probeID == galleryID:
            corr += 1
        else:#选择
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            # 用于处理和存储那些在模型匹配任务中被错误识别的“困难样本”
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            # 将 im_train 图像放在 im_test 图像的右侧，形成一张宽度更大的图像，用于直观对比。
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(map['path_rst'] + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), os.path.basename(testname).split('.')[0],os.path.basename(trainname).split('.')[0]), img)
    rankacc = corr / ntest * 100
    print('Rank-1 acc: %.3f%%' % rankacc)
    with open(map['path_rst'] + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')
    print('Real EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    # 计算上三角 已经等价于两两对比
    for i in range(n - 1):
        feat1 = featDB_test[i]
        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]
            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            s.append(dis)
            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)
    with open(map['path_rst'] + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')
    print('Feature extraction about real EER done!\n')
    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + map['path_rst'] + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + map['path_rst'] + 'veriEER/scores_EER_test.txt scores_EER_test')
    return
if __name__== "__main__" :
    parser = arg()
    map = set_arg(parser)
    # ccnet_model = init_ccnet_model(map,True)
    # train_model(ccnet_model,map)
    ccnet_model = init_ccnet_model(map,False)
    test_model(ccnet_model,map)
# saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc,path_rst)???









