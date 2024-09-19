import os
import numpy as np
path1 = '../Tongji/session1/'
path2 = '../Tongji/session2/'
file_path1 = './Tongji/session1/'
file_path2= './Tongji/session2/'
root = './'

with open(os.path.join(root, 'train_Tongji.txt'), 'w') as ofs:
    files = os.listdir(path1)
    files.sort()
    for filename in files:
        userID = int(filename[:5])
        sampleID = userID % 10
        userID = int((userID-1)/10)
        imagePath = os.path.join(file_path1, filename)
        # if sampleID == ID_number:
        ofs.write('%s %d\n'%(imagePath, userID))
            # ID_number = 0
with open(os.path.join(root, 'val_train_Tongji.txt'), 'w') as ofs:
    files = os.listdir('../Val/Tongji/session1/')
    files.sort()
    for filename in files:
        userID = int(filename[:5])
        sampleID = userID % 10
        userID = int((userID-1)/10)
        imagePath = os.path.join('./Val/Tongji/session1/', filename)
        # if sampleID == ID_number:
        ofs.write('%s %d\n'%(imagePath, userID))
with open(os.path.join(root, 'test_Tongji.txt'), 'w') as ofs:
    files = os.listdir(path2)
    files.sort()
    for filename in files:
        # print(filename)
        userID = int(filename[:5])
        userID = int((userID-1)/10)
        # print(userID)
        imagePath = os.path.join(file_path2, filename)
        # if userID % 2 != 0:
        ofs.write('%s %d\n'%(imagePath,userID))
with open(os.path.join(root, 'val_test_Tongji.txt'), 'w') as ofs:
    files = os.listdir('../Val/Tongji/session2/')
    files.sort()
    for filename in files:
        # print(filename)
        userID = int(filename[:5])
        userID = int((userID-1)/10)
        # print(userID)
        imagePath = os.path.join('./Val/Tongji/session2/', filename)
        # if userID % 2 != 0:
        ofs.write('%s %d\n'%(imagePath,userID))