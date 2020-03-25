from model import ResidualBlock, ResNet
from train import train, infer
from utils import select_n_random, matplotlib_imshow

from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import matplotlib.pyplot as plt
import numpy as np

import pickle
import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)

import glob
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    num_epochs = 20
    save_model_str = './models/'
    logging.info(f"{torch.cuda.is_available()}")
    cudnn.benchmark = True
    cudnn.enabled=True
    gpu = 'cuda:0'
    torch.cuda.set_device(gpu)
    model = ResNet(ResidualBlock, [1, 1, 1], initial_depth=64)
    model.cuda()
    
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    summary_dir = os.mkdir(f'{save_model_str}_summary')
    writer = SummaryWriter(f'{summary_dir}/summary_model')
    
    c = datetime.datetime.now()
    for epoch in range(num_epochs):
            logging.info('epoch %d lr %e', epoch, lr_scheduler.get_lr()[0])

            train_acc, train_obj = train(trainloader, model, criterion, optimizer, writer)
            logging.info('train_acc %f', train_acc)
            lr_scheduler.step()

            test_acc, test_obj = infer(testloader, model, criterion)
            logging.info('test_acc %f', test_acc)
    a = datetime.datetime.now() - c

    if save_model_str:
        # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
        if not os.path.exists(save_model_str):
            os.mkdir(save_model_str)
        save_model_str += f'_({datetime.datetime.now()})'
        os.mkdir(save_model_str)
        
        torch.save(model.state_dict(), f'{save_model_str}_state')
        # get some random training images
        model.eval()
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda(non_blocking=True)
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        # show images
        matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('cifar', img_grid)
        writer.add_graph(model, images)
        # images, labels = select_n_random(trainset.data, trainset.targets)

        # # get the class labels for each image
        # class_labels = [classes[lab] for lab in labels]

        # # log embeddings
        # features = images.view(-1, 28 * 28)
        # writer.add_embedding(features,
        #                     metadata=class_labels,
        #                     label_img=images.unsqueeze(1))
        writer.close()

    logging.info(f'test_acc: {test_acc}, save_model_str:{save_model_str}, total time :{a.total_seconds()}')



