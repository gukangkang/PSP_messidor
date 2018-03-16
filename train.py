import os
import numpy as np
import torch
from torch.autograd import Variable
from plainbox.impl import logging
from torch import nn
from torch.optim import Adam
from pspnet import PSPNet
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize,Scale
from torchvision.transforms import ToTensor, ToPILImage
from transform import Relabel, ToLabel, Colorize
# from Dataset import OrigaDataset
from Dataset_mse import OrigaDataset
from utils import Constants


import scipy.misc as misc

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[back/end]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def prepared_train_data():

    input_transform = Compose([
        ToTensor(),
    ])
    target_transform = Compose([
        ToLabel(),
    ])

    glaucoma = OrigaDataset(
                            input_transform=input_transform,
                            target_transform=target_transform
                            )

    loader = DataLoader(glaucoma, num_workers=4, batch_size=12)

    return loader


def prepared_test_data():


    input_transform = Compose([
        ToTensor(),
    ])
    target_transform = Compose([
        ToLabel(),
    ])

    glaucoma = OrigaDataset(data_type=Constants.TYPE_TEST,
                            input_transform=input_transform,
                            target_transform=target_transform
                            )

    loader = DataLoader(glaucoma, num_workers=4, batch_size=1)

    return loader


def train():
    print(torch.cuda.device_count())
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

    net = PSPNet(n_classes=1, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34',
                 pretrained=False)
    net = nn.DataParallel(net)
    net = net.cuda()

    weight = torch.ones(3)
    weight[0] = 0

    # print(net)
    optimizer = Adam(net.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()

    criterion = nn.MSELoss()

    loaders = prepared_train_data()
    test_loaders = prepared_test_data()
    # print(len(loaders))
    # print(loaders)
    
    for epoch in range(1, 100):
        print('Training................')
        epoch_loss = []
        iteration = 0

        net.train(mode=True)

        for step, sample_batch in enumerate(loaders):

            # print("Iter:"+str(iteration))
            iteration = iteration + 1
            images = sample_batch['image'].cuda()
            # masks = torch.transpose(sample_batch['mask'], 1, 3)
            masks = sample_batch['mask'].cuda()

            # print(images.size())
            # print(masks.size())

            inputs = Variable(images)
            targets = Variable(masks)

            # print(targets.size())

            outputs = net(inputs)
            outputs = torch.clamp(outputs, 0., 255.)
            # results = outputs.cpu().data.numpy()
            #
            # # print(np.shape(results))
            #
            # map = np.squeeze(results, axis=[0, 1])
            #
            # misc.imsave('./test_images/test_image_' + str(iteration) + '.png', map)
            # print(outputs)
            # print(outputs)

            # print(outputs.size())
            optimizer.zero_grad()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            # if iteration % 10 == 0:
            #     print("Epoch:{}, iteration:{}, loss:{}".format(epoch,
            #                                                    step,
            #                                                    loss))

            # if iteration % 10 == 0:
            #     results = outputs.cpu().data.numpy()
            #     # print(np.shape(results))
            #     map = np.squeeze(results, axis=1)
            #     # print(np.shape(map))
            #     # map = np.transpose(map, [0, 2, 3, 1])
            #
            #     misc.imsave('./train_image/test_image'+str(iteration)+'.png', map[0, :, :])

            # if iteration % 400 == 0:
            #     torch.save(net, 'Models/modol-'+str(epoch)+'-'+str(iteration)+'.pth')
        #
        torch.save(net, 'Models/modol-' + str(epoch)+ '.pth')
        print('Testing........................')
        net.train(mode=False)
        total_m1 = 0
        disc = 44
        for iteration, item in enumerate(test_loaders):

            images = item['image'].cuda()
            # masks = torch.transpose(sample_batch['mask'], 1, 3)
            masks = item['mask'].numpy()
            name = item['name']
     
            masks = np.squeeze(masks, axis=0)


            test_image = Variable(images).cuda()

            predict_label = net(test_image)
            predict_label = torch.clamp(predict_label, 0., 255.)

            results = predict_label.cpu().data.numpy()

            map = np.squeeze(results, axis=[0, 1])

            gt = np.zeros(shape=np.shape(masks))

            gt[masks>200] = 1

            prediction = np.zeros(shape=np.shape(map))
            prediction[map>disc] = 1

            overlap = gt + prediction
            # print(overlap.max(), overlap.min())
            # print(np.shape(overlap))

            image_inter = np.zeros(shape=np.shape(overlap))
            image_inter[overlap > 1] = 1
            num_inter = sum(sum(image_inter))

            # print(np.shape(num_inter))

            image_union = np.zeros(shape=np.shape(overlap))
            image_union[overlap > 0] = 1
            num_union = sum(sum(image_union))
            # print(np.shape(num_union))


            m_1 = (1 - num_inter / num_union)
            print('Image name is {}, and m1 is {}'.format(name[0], m_1))
            total_m1 = total_m1 + m_1

            map[map > disc] = 255

            #
            # misc.imsave('./test_images/test_image_' + str(iteration) + '.png', map)

            misc.imsave('./test_image/' + name[0], map)
        print('m1 is {}'.format(total_m1 / 200))
            # if iteration % 10 == 0:
            #     net.train(mode=False)
            #
            #     image_name = '/home/imed/workspace/PycharmProjects/pspnet-pytorch-master/' \
            #                  'Source/rename_image/AGLAIA_GT_006.jpg'
            #     image = misc.imread(image_name)
            #
            #     # image_data = np.transpose(np.divide(image, 255.), (2, 0, 1))
            #
            #     image_data = np.transpose(image, (2, 0, 1))
            #
            #     image_data = np.expand_dims(image_data, axis=0)
            #     print(np.shape(image_data))
            #     image_data = torch.from_numpy(image_data)
            #
            #     image_data = image_data.float().div(255)
            #
            #     test_image = Variable(image_data.cuda())
            #
            #     predict_label = net(test_image)
            #
            #     results = predict_label.cpu().data.numpy()
            #     print(np.shape(results))
            #     map = np.argmax(results, axis=1)
            #     # print(np.shape(map))
            #     # map = np.transpose(map, [0, 2, 3, 1])
            #     print(np.shape(map))
            #     print(np.max(map), np.min(map))
            #     print(map[0, :, :])
            #     misc.imsave('test_image.png', map[0, :, :])

                # predict_label = net(test_image)
                # print(predict_label.size())
                #
                # predict_label = np.transpose(np.squeeze(predict_label.cpu().data.numpy(), axis=0),
                #                              (1, 2, 0))
                #
                # predict_label = np.argmax(predict_label, axis=-1) * 128
                # # print(image_data.size())
                #
                # print(np.shape(predict_label))
                #
                # misc.imsave('/home/imed/workspace/PycharmProjects/pspnet-pytorch-master/test_image.png', predict_label)
            # torch.save(net, )

#
# @click.command()
# @click.option('--data-path', type=str, help='Path to dataset folder')
# @click.option('--models-path', type=str, help='Path for storing model snapshots')
# @click.option('--backend', type=str, default='resnet34', help='Feature extractor')
# @click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
# @click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
# @click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
# @click.option('--batch-size', type=int, default=16)
# @click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
# @click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
# @click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
# @click.option('--start-lr', type=float, default=0.001)
# @click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
# def train(data_path, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#     net, starting_epoch = build_network(snapshot, backend)
#     data_path = os.path.abspath(os.path.expanduser(data_path))
#     models_path = os.path.abspath(os.path.expanduser(models_path))
#     os.makedirs(models_path, exist_ok=True)
#
#     '''
#         To follow this training routine you need a DataLoader that yields the tuples of the following format:
#         (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
#         x - batch of input images,
#         y - batch of groung truth seg maps,
#         y_cls - batch of 1D tensors of dimensionality N: N total number of classes,
#         y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
#     '''
#     train_loader, class_weights, n_images = None, None, None
#
#     optimizer = optim.Adam(net.parameters(), lr=start_lr)
#     scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
#
#     for epoch in range(starting_epoch, starting_epoch + epochs):
#         seg_criterion = nn.NLLLoss2d(weight=class_weights)
#         cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
#         epoch_losses = []
#         train_iterator = tqdm(loader, total=max_steps // batch_size + 1)
#         net.train()
#         for x, y, y_cls in train_iterator:
#             steps += batch_size
#             optimizer.zero_grad()
#             x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
#             out, out_cls = net(x)
#             seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
#             loss = seg_loss + alpha * cls_loss
#             epoch_losses.append(loss.data[0])
#             status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {5:0.7f}'.format(
#                 epoch + 1, loss.data[0], np.mean(epoch_losses), scheduler.get_lr()[0])
#             train_iterator.set_description(status)
#             loss.backward()
#             optimizer.step()
#         scheduler.step()
#         torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
#         train_loss = np.mean(epoch_losses)

        
if __name__ == '__main__':
    train()
