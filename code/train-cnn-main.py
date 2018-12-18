# -*- coding: utf-8 -*-

import argparse
import os
import time
import datetime
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from cnn_finetune import make_model
from gen_label_csv import gen_data

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='train_cnn')
parser.add_argument('--test', default=True, help='test model on test set')

def main():
    global args

    args = parser.parse_args()
    # print(args.test)
    if not os.path.isfile('../data/train.csv'):
        print('Generate csv_data from data...')
        gen_data()
        print('Generate data done.')
    else:
        print('Data have been generated.')
    # random seed
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)

    # get the file name, to save models and test-result and log
    # model_name = 'resnet18'
    model_name = 'nasnetalarge'
    batch_size = 8
    resize_shape = 550 #345,400
    crop_shape = 512 #331,384
    file_name = model_name + '-' + str(crop_shape) + '-' + str(batch_size)
    print(os.path.basename(__file__) + '\t' + file_name)
    # model path
    if not os.path.exists('../model/%s' % file_name):
        os.makedirs('../model/%s' % file_name)
    if not os.path.exists('../submit/%s' % file_name):
        os.makedirs('../submit/%s' % file_name)
    if not os.path.exists('../log/%s' % file_name):
        os.makedirs('../log/%s' % file_name)
    # log file
    if not os.path.exists('../log/%s.txt' % file_name):
        with open('../log/%s.txt' % file_name, 'w') as acc_file:
            pass
    with open('../log/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

    # PIL to load image
    def default_loader(path):
        # return Image.open(path)
        return Image.open(path).convert('RGB')

    # train-data
    class TrainDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    # val-data
    class ValDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    # test-data
    class TestDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, filename

        def __len__(self):
            return len(self.imgs)

    # data-augment：trans in angles
    class FixedRotation(object):
        def __init__(self, angles):
            self.angles = angles

        def __call__(self, img):
            return fixed_rotate(img, self.angles)

    def fixed_rotate(img, angles):
        angles = list(angles)
        angles_num = len(angles)
        index = random.randint(0, angles_num - 1)
        return img.rotate(angles[index])


    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        # training
        for i, (images, target) in enumerate(train_loader):
            # time
            data_time.update(time.time() - end)
          
            image_var = torch.tensor(images).cuda(async=True)
            label = torch.tensor(target).cuda(async=True)

            y_pred = model(image_var)
            loss = criterion(y_pred, label)
            losses.update(loss.item(), images.size(0))

            prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
            acc.update(prec, PRED_COUNT)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

    # val
    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            image_var = torch.tensor(images).cuda(async=True)
            target = torch.tensor(labels).cuda(async=True)

            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)

            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

        print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        return acc.avg, losses.avg

    # test
    def test(test_loader, model, save_path):
        csv_map = OrderedDict({'filename': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            filepath = [os.path.basename(i) for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)

            with torch.no_grad():
                y_pred = model(image_var)
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
            
            csv_map['filename'].extend(filepath)
            for output in smax_out:
                prob = ';'.join([str(i) for i in output.data.tolist()])
                csv_map['probability'].append(prob)

        result = pd.DataFrame(csv_map)
        result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

        # submisiion-file
        sub_filename, sub_label = [], []
        for index, row in result.iterrows():
            sub_filename.append(row['filename'])
            pred_label = np.argmax(row['probability'])
            if pred_label == 0:
                sub_label.append('norm')
            else:
                sub_label.append('defect%d' % pred_label)

        submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
        submission.to_csv(save_path, header=None, index=False)
        return

    # save_model
    def save_checkpoint(state, is_best, is_lowest_loss, filename='../model/%s/checkpoint.pth.tar' % file_name):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '../model/%s/model_best.pth.tar' % file_name)
        if is_lowest_loss:
            shutil.copyfile(filename, '../model/%s/lowest_loss.pth.tar' % file_name)

    class AverageMeter(object):
        """Computes and stores the average and current value"""
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

    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        prob, pred = y_pred.topk(maxk, 1, True, True)
        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT
    
    # main

    # GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    batch_size = batch_size
    workers = 4
    # epoch
    stage_epochs = [20, 10, 10]  
    lr = 1e-4
    lr_decay = 5
    weight_decay = 1e-4
    stage = 0
    start_epoch = 0
    total_epochs = sum(stage_epochs)
    best_precision = 0
    lowest_loss = 100

    print_freq = 1
    evaluate = False
    resume = False

    ##get model 
    model = make_model(
        model_name,
        pretrained=True, #from scratch
        # input_size=(128, 128),
        num_classes=12,
        # dropout_p=args.dropout_p,
    )
    original_model_info = model.original_model_info
    # print('image_size',original_model_info.input_size)

    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if resume:
        checkpoint_path = '../model/%s/checkpoint.pth.tar' % file_name
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            stage = checkpoint['stage']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            if start_epoch in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('../model/%s/model_best.pth.tar' % file_name)['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # load train/val data list
    train_data_list = pd.read_csv('../data/train.csv')
    val_data_list = pd.read_csv('../data/train.csv')
    # load test_a data list
    test_a_data_list = pd.read_csv('../data/test_a.csv')
    # load test_b data list
    test_b_data_list = pd.read_csv('../data/test_b.csv')

    # pic normalize,the mean and std inherit from pretrain-model
    normalize = transforms.Normalize(mean=original_model_info.mean, std=original_model_info.std)
    
    # load train image for train
    train_data = TrainDataset(train_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((resize_shape, resize_shape)),
                                  transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomGrayscale(),
                                  # transforms.RandomRotation(20),
                                  FixedRotation([0, 90, 180, 270]),
                                  transforms.RandomCrop(crop_shape),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

    # load val-data
    val_data = ValDataset(val_data_list,
                          transform=transforms.Compose([
                              transforms.Resize((resize_shape, resize_shape)),
                              transforms.CenterCrop(crop_shape),
                              transforms.ToTensor(),
                              normalize,
                          ]))

    # load test-a
    test_a_data = TestDataset(test_a_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((resize_shape, resize_shape)),
                                transforms.CenterCrop(crop_shape),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    # load test-a
    test_b_data = TestDataset(test_b_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((resize_shape, resize_shape)),
                                transforms.CenterCrop(crop_shape),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=workers)
    test_a_loader = DataLoader(test_a_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=workers)
    test_b_loader = DataLoader(test_b_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=workers)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    if args.test==True:
        # load the pretrain model to get test_prediction
        print('Choose the test option and Get test prediction from pretrain model.')
        best_model = torch.load('../model/pretrain/model_best.pth.tar')
        model.load_state_dict(best_model['state_dict'])
        submit_path = '../submit/pretrain' 
        if not os.path.isdir(submit_path):
            os.makedirs(submit_path)
        test_a_path = submit_path + '/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_test_a.csv'
        test_b_path = submit_path + '/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_test_b.csv'
        test(test_loader=test_a_loader, model=model, save_path=test_a_path)
        test(test_loader=test_b_loader, model=model, save_path=test_b_path)
    else:
        # training
        print('Choose the train option and train model.')
        for epoch in range(start_epoch, total_epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            precision, avg_loss = validate(val_loader, model, criterion)

            with open('../log/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))

            # record loss
            is_best = (precision >= best_precision)
            is_lowest_loss = (avg_loss <= lowest_loss)
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
                'stage': stage,
                'lr': lr,
            }
            save_checkpoint(state, is_best, is_lowest_loss)

            # step into next
            if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('../model/%s/model_best.pth.tar' % file_name)['state_dict'])
                print('Step into next stage')
                with open('../log/%s.txt' % file_name, 'a') as acc_file:
                    acc_file.write('---------------Step into next stage----------------\n')

        with open('../log/%s.txt' % file_name, 'a') as acc_file:
            acc_file.write('* best acc: %.8f  %s\n' % (best_precision, os.path.basename(__file__)))
        with open('../log/best_acc.txt', 'a') as acc_file:
            acc_file.write('%s  * best acc: %.8f  %s\n' % (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision, os.path.basename(__file__) + '\t' + file_name))

        # load the best model to get test_prediction
        best_model = torch.load('../model/%s/model_best.pth.tar' % file_name)
        model.load_state_dict(best_model['state_dict'])
        test_a_path = '../submit/' + file_name + '/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_test_a.csv'
        test_b_path = '../submit/' + file_name + '/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_test_b.csv'
        test(test_loader=test_a_loader, model=model, save_path=test_a_path)
        test(test_loader=test_b_loader, model=model, save_path=test_b_path)

    # release GPU
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
