import torch 
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import logging
logging.basicConfig(level=logging.INFO)

# def preprocess(dataset, transforms):
#     dataset = copy.copy(dataset) #shallow copy
#     for transform in transforms:
#         dataset['data'] = transform(dataset['data'])
#     return dataset

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)
    for transform in reversed(transforms):
        dataset.data = transform(dataset.data)
    return dataset

 
class AccuracyMeter(object):
    
    def __init__(self, name, model_dir):
        self.name = name
        self.file = open(f'{model_dir}/{name}_stats.txt', 'a')
        self.reset()
    
    def reset(self):
        self.stats = {'acc': [], 'loss': []}
        self.cnt = 0
        self.time = 0.0
    
    def update(self, stats, time, n=1):
        self.stats['acc'].append(stats['acc'])
        self.stats['loss'].append(stats['loss'])
        self.cnt += n
        self.time += time
        logging.info(f"{self.name}_acc: {stats['acc']}, time for step{self.cnt}: {round(time, 2)}, total {self.name} time: {round(self.time, 2)}")
        self.file.write(f"{self.name}_acc: {stats['acc']}, time for step{self.cnt}: {round(time, 2)}, total {self.name} time: {round(self.time, 2)}\n")
    def get(self):
        return self.stats, self.cnt, self.time
    def plot(self, out_dir):
        out_dir = os.path.join(out_dir, 'plots')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for key in self.stats.keys():
            plt.plot(self.stats[key])
            plt.title(f'Model {self.name}_{key}')
            plt.ylabel(f'Loss {self.name}_{key}')
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, self.cnt, 5))
            plt.savefig(f"{os.path.join(out_dir, f'{self.name}_{key}')}.png")
            plt.close()
        self.file.close()
        
def write_to_file(save_dict, file_name):
    fo = open(file_name, "a")
    for k, v in save_dict.items():
        fo.write(str(k) + ': '+ str(v) + '\t')
    fo.write('\n')
    fo.close()
    
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True)
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True)
#     print(trainset.data[0].shape)
#     # indices = list(range(int(split*len(trainset))))
#     # valid_indices =  list(range(int(split*len(trainset)), len(trainset)))
#     logging.info(f"Training size= {len(trainset)}")
#     # training_sampler = SubsetRandomSampler(indices)
#     # valid_sampler = SubsetRandomSampler(valid_indices)
#     trainloader = torch.utils.data.DataLoader(dataset=transform.Transform(trainset, train_transform),
#                                             batch_size=batch_size) #,
#                                             #sampler=training_sampler) 

#     # validloader = torch.utils.data.DataLoader(dataset=transform.Transform(trainset, test_transform), 
#     #                                         batch_size=batch_size, 
#     #                                         sampler=valid_sampler)                
    
#     testloader = torch.utils.data.DataLoader(transform.Transform(testset, test_transform),
#                                             batch_size=batch_size,
#                                             shuffle=False)

#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')